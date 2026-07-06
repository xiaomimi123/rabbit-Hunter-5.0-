"""Tests for the ML model registry — every user-visible operation
(list/get_active/promote/rollback) is pinned including atomic-write
safety and marker-file semantics.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest
import yaml

from rabbit_hunter.ml.registry import (
    get_active_model, list_models, promote, rollback,
)


def _mk_model(root: Path, version: str, test_auc: float = 0.55,
              train_auc: float = 0.65) -> Path:
    """Create a fake ml_model_v<version>/ dir with model.pkl + result."""
    d = root / f"ml_model_v{version}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "model.pkl").open("wb") as f:
        pickle.dump({"fake": "model", "version": version}, f)
    (d / "training_result.json").write_text(json.dumps({
        "model_version": version,
        "trained_at": "2026-01-01T00:00:00Z",
        "train_auc": train_auc,
        "test_auc": test_auc,
        "train_accuracy": 0.65,
        "test_accuracy": 0.55,
        "n_train": 100, "n_test": 30,
        "features_used": ["ema20_t0"],
        "hyperparameters": {"model": "test"},
    }), encoding="utf-8")
    return d / "model.pkl"


def _mk_config(root: Path, model_path: Path | None = None) -> Path:
    """Create a minimal ml_scoring.yaml pointing at a model (or nothing)."""
    d = root / "configs" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "ml_scoring.yaml"
    data: dict = {"name": "ml_scoring", "version": "0.1.0", "params": {}}
    if model_path is not None:
        data["params"]["model_path"] = str(model_path)
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ============================================================
# list_models
# ============================================================

def test_list_models_empty_root(tmp_path):
    assert list_models(tmp_path) == []


def test_list_models_returns_sorted_versions(tmp_path):
    _mk_model(tmp_path, "20260101-000000", test_auc=0.55)
    _mk_model(tmp_path, "20260102-000000", test_auc=0.60)
    _mk_model(tmp_path, "20260103-000000", test_auc=0.52)
    models = list_models(tmp_path)
    assert [m.version for m in models] == [
        "20260101-000000", "20260102-000000", "20260103-000000",
    ]


def test_list_models_reads_metrics_from_json(tmp_path):
    _mk_model(tmp_path, "v1", test_auc=0.62, train_auc=0.78)
    models = list_models(tmp_path)
    m = models[0]
    assert m.test_auc == 0.62
    assert m.train_auc == 0.78
    assert m.n_train == 100
    assert "ema20_t0" in m.features_used


def test_list_models_skips_dirs_without_model_pkl(tmp_path):
    """A dir starting with ml_model_v but missing model.pkl is not listed
    — otherwise we'd advertise something that can't be loaded."""
    empty = tmp_path / "ml_model_vBROKEN"
    empty.mkdir()
    _mk_model(tmp_path, "v_good")
    models = list_models(tmp_path)
    assert len(models) == 1
    assert models[0].version == "v_good"


def test_list_models_marks_active(tmp_path):
    p1 = _mk_model(tmp_path, "v1")
    p2 = _mk_model(tmp_path, "v2")
    models = list_models(tmp_path, active_path=p2)
    by_ver = {m.version: m for m in models}
    assert by_ver["v2"].is_active is True
    assert by_ver["v1"].is_active is False


def test_list_models_survives_corrupt_training_json(tmp_path):
    """A corrupt training_result.json shouldn't hide the model — we
    still list it, just with placeholder metrics."""
    _mk_model(tmp_path, "v_ok")
    d = tmp_path / "ml_model_vCORRUPT"
    d.mkdir()
    (d / "model.pkl").write_bytes(b"pickle")
    (d / "training_result.json").write_text("not json", encoding="utf-8")
    models = list_models(tmp_path)
    versions = {m.version for m in models}
    assert "v_ok" in versions
    assert "CORRUPT" in versions


# ============================================================
# get_active_model
# ============================================================

def test_get_active_reads_config_path(tmp_path):
    model = _mk_model(tmp_path, "v1")
    cfg = _mk_config(tmp_path, model)
    assert get_active_model(cfg) == model


def test_get_active_returns_none_when_config_missing(tmp_path):
    assert get_active_model(tmp_path / "no.yaml") is None


def test_get_active_returns_none_when_no_model_path(tmp_path):
    cfg = _mk_config(tmp_path, None)   # config without model_path
    assert get_active_model(cfg) is None


# ============================================================
# promote — atomic + records previous
# ============================================================

def test_promote_updates_config_and_records_previous(tmp_path):
    old = _mk_model(tmp_path, "old", test_auc=0.55)
    new = _mk_model(tmp_path, "new", test_auc=0.60)
    cfg = _mk_config(tmp_path, old)
    marker = tmp_path / "configs" / ".ml_previous_active"

    prev = promote(cfg, new, prev_marker_path=marker)
    assert prev == old
    assert get_active_model(cfg) == new
    assert marker.exists()
    assert Path(marker.read_text().strip()) == old


def test_promote_preserves_other_config_fields(tmp_path):
    old = _mk_model(tmp_path, "old")
    new = _mk_model(tmp_path, "new")
    cfg = _mk_config(tmp_path, old)
    # Add an extra field the user might have edited
    data = yaml.safe_load(cfg.read_text())
    data["params"]["prob_threshold"] = 0.55
    data["params"]["side_mode"] = "both"
    cfg.write_text(yaml.safe_dump(data), encoding="utf-8")

    promote(cfg, new, prev_marker_path=tmp_path / "marker")

    result = yaml.safe_load(cfg.read_text())
    assert result["params"]["model_path"] == str(new)
    assert result["params"]["prob_threshold"] == 0.55
    assert result["params"]["side_mode"] == "both"
    assert result["name"] == "ml_scoring"


def test_promote_raises_when_target_model_missing(tmp_path):
    old = _mk_model(tmp_path, "old")
    cfg = _mk_config(tmp_path, old)
    with pytest.raises(FileNotFoundError):
        promote(cfg, tmp_path / "no_such_model.pkl",
                prev_marker_path=tmp_path / "marker")
    # Config unchanged
    assert get_active_model(cfg) == old


def test_promote_no_previous_when_config_had_none(tmp_path):
    """Promoting when no model was active is fine — previous returned
    is None and no marker is written."""
    new = _mk_model(tmp_path, "new")
    cfg = _mk_config(tmp_path, None)
    marker = tmp_path / "marker"
    prev = promote(cfg, new, prev_marker_path=marker)
    assert prev is None
    assert not marker.exists()
    assert get_active_model(cfg) == new


# ============================================================
# rollback
# ============================================================

def test_rollback_restores_previous(tmp_path):
    old = _mk_model(tmp_path, "old")
    new = _mk_model(tmp_path, "new")
    cfg = _mk_config(tmp_path, old)
    marker = tmp_path / "marker"
    promote(cfg, new, prev_marker_path=marker)
    assert get_active_model(cfg) == new

    restored = rollback(cfg, prev_marker_path=marker)
    assert restored == old
    assert get_active_model(cfg) == old
    # The marker now points to "new" (so rollback → rollback goes forward)
    assert Path(marker.read_text().strip()) == new


def test_rollback_raises_when_no_marker(tmp_path):
    old = _mk_model(tmp_path, "old")
    cfg = _mk_config(tmp_path, old)
    with pytest.raises(FileNotFoundError):
        rollback(cfg, prev_marker_path=tmp_path / "no_marker")


def test_rollback_raises_when_marker_points_to_missing_file(tmp_path):
    old = _mk_model(tmp_path, "old")
    new = _mk_model(tmp_path, "new")
    cfg = _mk_config(tmp_path, new)
    marker = tmp_path / "marker"
    marker.write_text(str(old), encoding="utf-8")
    # Delete the old model
    old.unlink()
    with pytest.raises(FileNotFoundError):
        rollback(cfg, prev_marker_path=marker)
    # Config unchanged
    assert get_active_model(cfg) == new
