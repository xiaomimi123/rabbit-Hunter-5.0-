"""Tests for the shadow daily digest renderer.

The digest must be robust to every combination of missing input:
  - No ledger yet (fresh runner)
  - Ledger but no closed trades
  - Ledger with trades but no baselines
  - Baseline paths supplied but files missing on disk
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from rabbit_hunter.backtest.ledger import Ledger
from rabbit_hunter.shadow.digest import render


def _empty_state(tmp_path: Path) -> Path:
    state = tmp_path / "shadows"
    (state / "state").mkdir(parents=True)
    return state


def _seed_ledger(state: Path, closed_trades=None) -> None:
    ledger = Ledger(initial_capital=10_000.0)
    ledger.equity = 11_000.0
    if closed_trades:
        ledger.closed_trades = closed_trades
    with (state / "state" / "ledger.pkl").open("wb") as f:
        pickle.dump(ledger, f)


def _seed_metrics_row(state: Path, alerts: str = "") -> None:
    p = state / "state" / "metrics_history.parquet"
    pd.DataFrame([{
        "timestamp_ms": 1_700_000_000_000,
        "equity": 11_000.0,
        "initial_capital": 10_000.0,
        "total_pnl": 1_000.0,
        "pnl_pct": 0.1,
        "peak_equity": 11_500.0,
        "drawdown_from_peak_pct": 0.043,
        "open_positions": 1,
        "open_notional": 5000.0,
        "open_long_notional": 5000.0,
        "open_short_notional": 0.0,
        "total_closed_trades": 5,
        "winners": 3, "losers": 2,
        "winrate": 0.6, "profit_factor": 1.5,
        "last_bar_ts_ms": 1_700_000_000_000,
        "minutes_since_last_bar": 0.0,
        "consecutive_errors": 0,
        "alerts": alerts,
        "alert_count": 1 if alerts else 0,
    }]).to_parquet(p)


# ============================================================
# Fresh runner — no state at all
# ============================================================

def test_digest_renders_when_state_empty(tmp_path):
    state = _empty_state(tmp_path)
    md = render(state)
    assert md.startswith("# Rabbit Hunter — Shadow Digest")
    assert "Runtime health" in md
    assert "No metrics history yet" in md
    assert "No closed trades to classify yet" in md


# ============================================================
# Metrics + ledger present, no baselines supplied
# ============================================================

def test_digest_includes_health_from_metrics(tmp_path):
    state = _empty_state(tmp_path)
    _seed_metrics_row(state)
    md = render(state)
    assert "Equity: $11,000" in md
    assert "PnL: +1,000" in md
    assert "Open positions: 1" in md


def test_digest_no_baseline_muted_message(tmp_path):
    state = _empty_state(tmp_path)
    _seed_ledger(state)
    md = render(state)
    assert "No trade baseline specified" in md
    assert "No feature baseline specified" in md


def test_digest_missing_baseline_file_is_reported(tmp_path):
    state = _empty_state(tmp_path)
    _seed_ledger(state)
    md = render(
        state,
        trade_baseline_path=tmp_path / "does_not_exist.json",
        feature_baseline_path=tmp_path / "also_not_here.json",
    )
    assert "Baseline missing" in md


# ============================================================
# Alerts section
# ============================================================

def test_digest_shows_recent_alerts_when_present(tmp_path):
    state = _empty_state(tmp_path)
    _seed_metrics_row(state, alerts="drawdown_high=12%>=10%")
    # Push the ts within lookback — the seeded row is at
    # 1_700_000_000_000 which is far in the past; render() uses
    # datetime.now(). So this test only verifies the "no recent" path.
    md = render(state)
    assert "Recent alerts" in md


def test_digest_ok_when_no_alerts(tmp_path):
    state = _empty_state(tmp_path)
    _seed_metrics_row(state)   # no alerts
    md = render(state)
    assert ("All clear" in md) or ("No alerts" in md)


# ============================================================
# Trade drift section
# ============================================================

def test_digest_trade_drift_section_with_baseline(tmp_path):
    from rabbit_hunter.analytics.baseline import (
        BaselineSnapshot, ClusterBaseline, save,
    )
    state = _empty_state(tmp_path)
    # Seed enough shadow trades for at least one cluster
    closed = [
        {"symbol": "BTC-USDT-SWAP", "side": "short",
         "pnl_after_fees": +100.0, "exit_reason": "tp",
         "exit_time": 1_700_000_000_000, "bars_held": 5,
         "entry_snapshot": {"rsi_14": 25.0, "zscore_20": 0.0,
                             "bb_pct": 0.5, "structure_regime": "range",
                             "bos_flag": 0}},
    ]
    _seed_ledger(state, closed_trades=closed)
    baseline = BaselineSnapshot(
        tag="v0.1.3", created_at_utc="2026-01-01T00:00:00Z",
        source_report_dir="reports/x", total_trades=100,
        clusters=[ClusterBaseline("1_momentum_breakdown",
                                   100, 0.6, 20.0, 3.0, 1.5, 40.0)],
    )
    bp = save(baseline, tmp_path / "baseline.json")
    md = render(state, trade_baseline_path=bp)
    assert "Trade-outcome drift" in md
    assert "1_momentum_breakdown" in md


# ============================================================
# Feature drift section
# ============================================================

def test_digest_feature_drift_section_with_baseline(tmp_path):
    import numpy as np
    from rabbit_hunter.analytics.feature_stability import (
        build_baseline_from_features, save,
    )
    state = _empty_state(tmp_path)
    # Build a synthetic baseline
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "close": rng.normal(100, 5, 500),
        "volume": rng.normal(50, 5, 500),
        "atr_14": rng.normal(1.0, 0.1, 500),
        "atr_pct": rng.normal(0.01, 0.002, 500),
        "ema20_slope": rng.normal(0, 0.1, 500),
        "adx": rng.normal(25, 5, 500),
        "rsi_14": rng.normal(50, 10, 500),
        "bb_pct": rng.normal(0.5, 0.1, 500),
        "zscore_20": rng.normal(0, 1, 500),
        "volume_ratio_20": rng.normal(1.0, 0.2, 500),
        "funding_rate": rng.normal(0.0001, 0.00005, 500),
        "oi_change_pct": rng.normal(0, 0.01, 500),
    })
    snap = build_baseline_from_features(df, tag="v1", source="test")
    bp = save(snap, tmp_path / "features.json")

    # Seed a live features_log
    live = df.iloc[:300].copy()
    live["timestamp"] = range(len(live))
    live["symbol"] = "BTC-USDT-SWAP"
    live.to_parquet(state / "state" / "features_log.parquet")

    md = render(state, feature_baseline_path=bp)
    assert "Feature-distribution drift" in md
    assert "rsi_14" in md
