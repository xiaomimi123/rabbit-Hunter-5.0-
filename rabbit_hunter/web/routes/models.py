"""ML model registry API."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


@router.get("")
def list_models_api(request: Request) -> dict:
    """Every trained model + which one is active."""
    from rabbit_hunter.ml.registry import list_models, get_active_model
    paths = _paths(request)
    active = get_active_model(paths.ml_config)
    out = []
    for m in list_models(paths.models, active_path=active):
        out.append({
            "version": m.version,
            "path": str(m.path),
            "is_active": m.is_active,
            "trained_at": m.trained_at,
            "train_auc": m.train_auc,
            "test_auc": m.test_auc,
            "n_train": m.n_train,
            "n_test": m.n_test,
        })
    return {"models": out}


@router.get("/retrain-log")
def retrain_log(request: Request, limit: int = 100) -> dict:
    """Audit log from models/retrain_log.jsonl, most recent first."""
    paths = _paths(request)
    log_p = paths.models / "retrain_log.jsonl"
    if not log_p.exists():
        return {"entries": []}
    entries: list[dict] = []
    for line in log_p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    entries = list(reversed(entries))[:max(1, limit)]
    return {"entries": entries}
