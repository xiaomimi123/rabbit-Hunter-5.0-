"""ML model registry — discover, describe, activate, and roll back models.

The models/ directory is append-only: every training run produces one new
`ml_model_v<timestamp>/` and never overwrites an old one. The "active"
model is whichever path is written into `configs/strategies/ml_scoring.yaml`.

This module provides the four operations you'd otherwise do by hand:

  list_models(models_root) — scan models/, return each version's metrics
  get_active_model(config)  — read which model MLScoring is pointed at
  promote(config, path)     — atomically switch to a new model, record
                              the previous active so rollback works
  rollback(config)          — restore the previously-active model

Promote/rollback rewrite `ml_scoring.yaml` using atomic tempfile+rename
so a crashed edit can't leave a half-written config on disk. The previous
active path is stashed at `configs/.ml_previous_active` so recovering
from `rabbit ml rollback` works even after a process restart.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


PREV_ACTIVE_MARKER = "configs/.ml_previous_active"


@dataclass
class ModelDescription:
    """What we know about one on-disk model."""
    path: Path                    # path to the model.pkl
    version: str
    dir: Path                     # path to ml_model_v.../
    trained_at: str
    train_auc: float
    test_auc: float
    train_accuracy: float
    test_accuracy: float
    n_train: int
    n_test: int
    features_used: list[str] = field(default_factory=list)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    is_active: bool = False


def _load_training_result(model_dir: Path) -> dict[str, Any] | None:
    """Parse training_result.json in a model dir. Returns None if the
    JSON is missing or unreadable — we still list the model dir but
    with placeholder metrics so a corrupted result file doesn't hide
    a real .pkl."""
    p = model_dir / "training_result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _describe(model_dir: Path, active_path: Path | None) -> ModelDescription:
    result = _load_training_result(model_dir) or {}
    version = result.get("model_version") or model_dir.name.replace("ml_model_v", "")
    model_path = model_dir / "model.pkl"
    return ModelDescription(
        path=model_path,
        version=version,
        dir=model_dir,
        trained_at=result.get("trained_at", ""),
        train_auc=float(result.get("train_auc", 0.0)),
        test_auc=float(result.get("test_auc", 0.0)),
        train_accuracy=float(result.get("train_accuracy", 0.0)),
        test_accuracy=float(result.get("test_accuracy", 0.0)),
        n_train=int(result.get("n_train", 0)),
        n_test=int(result.get("n_test", 0)),
        features_used=list(result.get("features_used", [])),
        hyperparameters=dict(result.get("hyperparameters", {})),
        is_active=(active_path is not None and model_path == active_path),
    )


def list_models(
    models_root: Path,
    active_path: Path | None = None,
) -> list[ModelDescription]:
    """Return every ml_model_v*/ under models_root, sorted by version.

    `active_path` marks which model is currently active (comparison by
    resolved absolute path). Missing model.pkl entries are skipped —
    we won't advertise a directory that can't be loaded.
    """
    if not models_root.exists():
        return []
    active_resolved: Path | None = None
    if active_path is not None:
        try:
            active_resolved = active_path.resolve()
        except OSError:
            active_resolved = None
    out: list[ModelDescription] = []
    for d in sorted(models_root.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.startswith("ml_model_v"):
            continue
        model_pkl = d / "model.pkl"
        if not model_pkl.exists():
            continue
        try:
            resolved = model_pkl.resolve()
        except OSError:
            resolved = model_pkl
        active_flag = (active_resolved is not None
                       and resolved == active_resolved)
        desc = _describe(d, active_path=(resolved if active_flag else None))
        desc.is_active = active_flag
        out.append(desc)
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML via tempfile + os.replace so a crash mid-write can't
    leave a truncated config on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, sort_keys=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def get_active_model(ml_config_path: Path) -> Path | None:
    """Read the active model_path from configs/strategies/ml_scoring.yaml.
    Returns None if the config or the model_path field is missing."""
    if not ml_config_path.exists():
        return None
    data = _load_yaml(ml_config_path)
    params = data.get("params") or {}
    p = params.get("model_path")
    return Path(p) if p else None


def promote(
    ml_config_path: Path,
    new_model_path: Path,
    prev_marker_path: Path | None = None,
) -> Path | None:
    """Atomically point the ML config at a new model. Records the
    previously-active path in `prev_marker_path` so rollback() can
    restore it. Returns the previously-active path (or None if there
    wasn't one)."""
    if not new_model_path.exists():
        raise FileNotFoundError(f"model does not exist: {new_model_path}")
    prev_active = get_active_model(ml_config_path)
    data = _load_yaml(ml_config_path)
    params = dict(data.get("params") or {})
    params["model_path"] = str(new_model_path)
    data["params"] = params
    _atomic_write_yaml(ml_config_path, data)

    if prev_marker_path is None:
        prev_marker_path = ml_config_path.parent.parent / ".ml_previous_active"
    if prev_active is not None:
        prev_marker_path.parent.mkdir(parents=True, exist_ok=True)
        prev_marker_path.write_text(str(prev_active), encoding="utf-8")
    return prev_active


def rollback(
    ml_config_path: Path,
    prev_marker_path: Path | None = None,
) -> Path:
    """Restore the previously-active model path. Raises if there's no
    recorded previous or the previous path no longer exists on disk."""
    if prev_marker_path is None:
        prev_marker_path = ml_config_path.parent.parent / ".ml_previous_active"
    if not prev_marker_path.exists():
        raise FileNotFoundError(
            f"no previous-active marker at {prev_marker_path}. "
            "Did you promote at least once since the last rollback?"
        )
    prev_path = Path(prev_marker_path.read_text(encoding="utf-8").strip())
    if not prev_path.exists():
        raise FileNotFoundError(
            f"marker points to {prev_path} but that file is gone. "
            "Cannot roll back to a missing model."
        )
    current = get_active_model(ml_config_path)
    data = _load_yaml(ml_config_path)
    params = dict(data.get("params") or {})
    params["model_path"] = str(prev_path)
    data["params"] = params
    _atomic_write_yaml(ml_config_path, data)
    # Swap the marker: what was current is now the "previous" for a
    # subsequent rollback (so rollback → rollback returns you to where
    # you were before the first rollback).
    if current is not None:
        prev_marker_path.write_text(str(current), encoding="utf-8")
    else:
        try:
            prev_marker_path.unlink()
        except OSError:
            pass
    return prev_path
