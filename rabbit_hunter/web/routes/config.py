"""Config API — current active config + snapshot history + diffs."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


@router.get("/current")
def current_config(request: Request) -> dict:
    """The current default.yaml as raw text (frontend renders in a
    monospace block)."""
    p = _paths(request).config_default
    if not p.exists():
        return {"path": str(p), "content": None}
    return {"path": str(p), "content": p.read_text(encoding="utf-8")}


@router.get("/history")
def config_history_api(request: Request) -> dict:
    """Every recorded snapshot — used by the config drift page."""
    from rabbit_hunter.config.history import history
    paths = _paths(request)
    entries = history(history_dir=paths.config_history)
    return {"entries": [{
        "index": i,
        "timestamp_utc": e.timestamp_utc,
        "config_hash": e.config_hash,
        "source_path": e.source_path,
        "snapshot_path": e.snapshot_path,
        "note": e.note,
    } for i, e in enumerate(entries)]}


@router.get("/diff")
def config_diff(
    request: Request,
    rev_a: str = "previous",
    rev_b: str = "latest",
) -> dict:
    """Unified diff between two snapshots. Returns empty string when
    identical (frontend shows "no changes")."""
    from rabbit_hunter.config.history import diff
    paths = _paths(request)
    try:
        text = diff(rev_a, rev_b, history_dir=paths.config_history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"rev_a": rev_a, "rev_b": rev_b, "diff": text}
