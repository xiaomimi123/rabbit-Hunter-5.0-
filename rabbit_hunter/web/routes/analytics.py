"""Analytics API — cluster performance + drift + feature stability."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


def _load_ledger(state_dir: Path):
    p = state_dir / "state" / "ledger.pkl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


@router.get("/clusters/shadow")
def cluster_shadow(request: Request) -> dict:
    """Per-cluster performance over shadow closed_trades."""
    from rabbit_hunter.analytics.cluster_performance import analyze
    from rabbit_hunter.analytics.clustering import CLUSTER_DESCRIPTIONS
    paths = _paths(request)
    ledger = _load_ledger(paths.shadow)
    if ledger is None or not ledger.closed_trades:
        return {"clusters": [], "total_trades": 0}
    df = pd.DataFrame(ledger.closed_trades)
    report = analyze(df, schema="shadow")
    out = []
    for s in report.stats:
        out.append({
            "cluster": s.cluster,
            "description": CLUSTER_DESCRIPTIONS.get(s.cluster, ""),
            "n": s.n,
            "winrate": s.winrate,
            "total_pnl": s.total_pnl,
            "avg_pnl": s.avg_pnl,
            "profit_factor": (s.profit_factor
                              if s.profit_factor != float("inf") else None),
            "sharpe_est": s.sharpe_est,
            "symbols_touched": s.symbols_touched,
        })
    return {"clusters": out, "total_trades": report.total_trades}


@router.get("/clusters/report/{name}")
def cluster_report(request: Request, name: str) -> dict:
    """Per-cluster performance over a backtest report's trades."""
    from rabbit_hunter.analytics.cluster_performance import analyze
    from rabbit_hunter.analytics.clustering import CLUSTER_DESCRIPTIONS
    paths = _paths(request)
    trades_p = paths.reports / name / "trades.parquet"
    if not trades_p.exists():
        raise HTTPException(status_code=404, detail=f"no such report: {name}")
    df = pd.read_parquet(trades_p)
    report = analyze(df, schema="backtest")
    out = []
    for s in report.stats:
        out.append({
            "cluster": s.cluster,
            "description": CLUSTER_DESCRIPTIONS.get(s.cluster, ""),
            "n": s.n,
            "winrate": s.winrate,
            "total_pnl": s.total_pnl,
            "avg_pnl": s.avg_pnl,
            "profit_factor": (s.profit_factor
                              if s.profit_factor != float("inf") else None),
            "sharpe_est": s.sharpe_est,
            "symbols_touched": s.symbols_touched,
        })
    return {"clusters": out, "total_trades": report.total_trades}


@router.get("/drift")
def drift(request: Request, baseline: str) -> dict:
    """Trade-outcome drift: shadow vs a named baseline JSON."""
    from rabbit_hunter.analytics.baseline import load
    from rabbit_hunter.analytics.cluster_performance import analyze
    from rabbit_hunter.analytics.drift import compare
    paths = _paths(request)
    baseline_p = paths.baselines / baseline
    if not baseline_p.exists():
        raise HTTPException(status_code=404,
                             detail=f"no baseline: {baseline}")
    ledger = _load_ledger(paths.shadow)
    if ledger is None or not ledger.closed_trades:
        return {"ok": True, "findings": [],
                "message": "no shadow trades yet"}
    bl = load(baseline_p)
    live = analyze(pd.DataFrame(ledger.closed_trades), schema="shadow")
    report = compare(bl, live)
    findings = [{
        "cluster": f.cluster,
        "baseline_n": f.baseline_n,
        "live_n": f.live_n,
        "baseline_winrate": f.baseline_winrate,
        "live_winrate": f.live_winrate,
        "winrate_delta": f.winrate_delta,
        "avg_pnl_zscore": f.avg_pnl_zscore,
        "triggered": f.triggered,
        "reason": f.reason,
    } for f in report.findings]
    return {"ok": report.ok, "findings": findings,
            "baseline_only": report.baseline_only,
            "live_only": report.live_only}


@router.get("/baselines")
def baselines(request: Request) -> dict:
    """List available baseline files under baselines/."""
    paths = _paths(request)
    out = []
    if paths.baselines.exists():
        for p in sorted(paths.baselines.iterdir()):
            if p.is_file() and p.suffix == ".json":
                out.append({
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                })
    return {"baselines": out}
