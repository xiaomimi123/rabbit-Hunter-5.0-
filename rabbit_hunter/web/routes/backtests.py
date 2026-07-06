"""Backtest-report API — list + drill-down + compare."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


def _list_report_dirs(reports_root: Path) -> list[Path]:
    if not reports_root.exists():
        return []
    return sorted(
        [d for d in reports_root.iterdir()
         if d.is_dir() and (d / "trades.parquet").exists()],
        key=lambda d: d.name,
        reverse=True,
    )


@router.get("")
def list_backtests(request: Request) -> dict:
    """Every reports/*/ that has trades.parquet, most-recent first."""
    from rabbit_hunter.backtest.compare import _compute_metrics
    reports_root = _paths(request).reports
    out = []
    for d in _list_report_dirs(reports_root):
        try:
            m = _compute_metrics(d.name, d)
        except Exception:
            continue
        out.append({
            "name": d.name,
            "path": str(d),
            "n_trades": m.n_trades,
            "winrate": m.winrate,
            "total_pnl": m.total_pnl,
            "profit_factor": m.profit_factor if m.profit_factor != float("inf") else None,
            "sharpe_est": m.sharpe_est,
            "max_drawdown": m.max_drawdown,
        })
    return {"reports": out}


@router.get("/{name}")
def get_backtest(request: Request, name: str) -> dict:
    """Full metrics + top trades for a single report."""
    from rabbit_hunter.backtest.compare import _compute_metrics
    reports_root = _paths(request).reports
    d = reports_root / name
    if not (d / "trades.parquet").exists():
        raise HTTPException(status_code=404, detail=f"no such report: {name}")
    m = _compute_metrics(name, d)
    # Top 20 by |PnL| — same convention as the compare tool
    df = m.trades_df.copy()
    pnl_col = "pnl_after_fees" if "pnl_after_fees" in df.columns else "pnl"
    df = df.reindex(df[pnl_col].abs().sort_values(ascending=False).index).head(20)
    top: list[dict] = []
    for _, r in df.iterrows():
        top.append({
            "symbol": r.get("symbol"),
            "side": r.get("side"),
            "pnl": float(r[pnl_col]),
            "exit_reason": r.get("exit_reason"),
            "entry_time_ms": int(r.get("entry_time") or 0),
            "exit_time_ms": int(r.get("exit_time") or 0),
        })
    return {
        "name": m.label,
        "n_trades": m.n_trades,
        "winners": m.winners,
        "losers": m.losers,
        "winrate": m.winrate,
        "total_pnl": m.total_pnl,
        "avg_pnl": m.avg_pnl,
        "best_pnl": m.best_pnl,
        "worst_pnl": m.worst_pnl,
        "profit_factor": (m.profit_factor
                          if m.profit_factor != float("inf") else None),
        "sharpe_est": m.sharpe_est,
        "max_drawdown": m.max_drawdown,
        "max_drawdown_pct": m.max_drawdown_pct,
        "top_trades": top,
    }
