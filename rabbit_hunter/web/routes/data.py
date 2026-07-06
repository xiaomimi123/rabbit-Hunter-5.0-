"""Data-pipeline health API."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


@router.get("/health")
def data_health(
    request: Request,
    intervals: str = "1H",
    grace_bars: int = 2,
) -> dict:
    """Per-(symbol, interval) health snapshot."""
    from rabbit_hunter.data_engine.health import check_all, summarize
    from rabbit_hunter.config.loader import load_config
    paths = _paths(request)
    if not paths.config_default.exists():
        return {"reports": [], "summary": {"total": 0, "healthy": 0,
                                            "unhealthy": 0, "by_status": {}}}
    cfg = load_config(paths.config_default)
    interval_list = [x.strip() for x in intervals.split(",") if x.strip()]
    reports = check_all(
        root=paths.data, symbols=list(cfg.data.symbols),
        intervals=interval_list, freshness_grace_bars=grace_bars,
    )
    return {
        "reports": [r.to_row() for r in reports],
        "summary": summarize(reports),
    }
