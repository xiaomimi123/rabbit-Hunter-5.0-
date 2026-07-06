"""Shadow-mode API — the frontend's primary data source."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request


router = APIRouter()


def _paths(request: Request):
    return request.app.state.paths


def _load_ledger(state_dir: Path):
    p = state_dir / "state" / "ledger.pkl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


def _load_metrics(state_dir: Path) -> pd.DataFrame:
    p = state_dir / "state" / "metrics_history.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


@router.get("/state")
def shadow_state(request: Request) -> dict:
    """Point-in-time summary — what the header cards show."""
    p = _paths(request)
    ledger = _load_ledger(p.shadow)
    hist = _load_metrics(p.shadow)
    if ledger is None:
        return {"has_ledger": False, "message": "no shadow ledger yet"}
    latest = hist.iloc[-1].to_dict() if not hist.empty else {}
    initial = float(ledger.initial_capital)
    equity = float(ledger.equity)
    return {
        "has_ledger": True,
        "equity": equity,
        "initial_capital": initial,
        "total_pnl": equity - initial,
        "pnl_pct": (equity - initial) / initial if initial else 0.0,
        "peak_equity": float(latest.get("peak_equity") or equity),
        "drawdown_pct": float(latest.get("drawdown_from_peak_pct") or 0.0),
        "open_positions": len(ledger.open_positions),
        "closed_trades": len(ledger.closed_trades),
        "winrate": float(latest.get("winrate") or 0.0),
        "profit_factor": float(latest.get("profit_factor") or 0.0),
        "last_tick_ms": int(latest["timestamp_ms"]) if latest else None,
        "minutes_since_last_bar": float(
            latest.get("minutes_since_last_bar") or 0.0
        ),
        "consecutive_errors": int(latest.get("consecutive_errors") or 0),
    }


@router.get("/metrics-history")
def shadow_metrics_history(request: Request, hours: int = 24) -> dict:
    """Time series for the equity + drawdown charts. `hours` window."""
    hist = _load_metrics(_paths(request).shadow)
    if hist.empty:
        return {"points": []}
    if hours > 0:
        cutoff = int(hist["timestamp_ms"].max()) - hours * 3_600_000
        hist = hist[hist["timestamp_ms"] >= cutoff]
    cols = ["timestamp_ms", "equity", "total_pnl", "drawdown_from_peak_pct",
            "open_positions", "winrate", "alert_count"]
    cols = [c for c in cols if c in hist.columns]
    return {"points": hist[cols].to_dict(orient="records")}


@router.get("/positions")
def shadow_positions(request: Request) -> dict:
    """Current open positions with entry price, stop, take profit."""
    ledger = _load_ledger(_paths(request).shadow)
    if ledger is None:
        return {"positions": []}
    out = []
    for sym, pos in ledger.open_positions.items():
        out.append({
            "symbol": sym,
            "side": pos.side,
            "size": float(pos.size),
            "entry_price": float(pos.entry_price),
            "stop": float(pos.stop),
            "take_profit": float(pos.take_profit),
            "entry_time_ms": int(pos.entry_time),
            "bars_held": int(pos.bars_held),
            "funding_accum": float(pos.funding_accum),
        })
    return {"positions": out}


@router.get("/trades")
def shadow_trades(request: Request, limit: int = 100) -> dict:
    """Recent closed trades — most recent first."""
    ledger = _load_ledger(_paths(request).shadow)
    if ledger is None or not ledger.closed_trades:
        return {"trades": []}
    trades = list(reversed(ledger.closed_trades))[:max(1, limit)]
    out = []
    for t in trades:
        out.append({
            "symbol": t.get("symbol"),
            "side": t.get("side"),
            "pnl": float(t.get("pnl_after_fees") or 0.0),
            "exit_reason": t.get("exit_reason"),
            "exit_time_ms": int(t.get("exit_time") or 0),
            "entry_time_ms": int(t.get("entry_time") or 0),
            "bars_held": int(t.get("bars_held") or 0),
        })
    return {"trades": out}


@router.get("/alerts")
def shadow_alerts(request: Request, hours: int = 24, limit: int = 50) -> dict:
    """Recent alerts (from metrics_history), most recent first."""
    hist = _load_metrics(_paths(request).shadow)
    if hist.empty or "alerts" not in hist.columns:
        return {"alerts": []}
    now_ms = int(hist["timestamp_ms"].max())
    cutoff = now_ms - hours * 3_600_000
    recent = hist[(hist["timestamp_ms"] >= cutoff)
                    & (hist["alert_count"] > 0)]
    out = []
    for _, r in recent.tail(limit).iloc[::-1].iterrows():
        out.append({
            "timestamp_ms": int(r["timestamp_ms"]),
            "alerts": str(r["alerts"]),
            "count": int(r["alert_count"]),
        })
    return {"alerts": out}
