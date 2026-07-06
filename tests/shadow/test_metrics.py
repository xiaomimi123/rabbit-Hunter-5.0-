"""Unit tests for ShadowMetrics — the per-tick snapshot + alerts collector."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rabbit_hunter.backtest.ledger import Ledger, Position
from rabbit_hunter.shadow.metrics import (
    AlertThresholds, MetricsSnapshot, ShadowMetrics,
)


def _fresh_ledger(initial: float = 10_000.0) -> Ledger:
    return Ledger(initial_capital=initial)


def _open_position(
    ledger: Ledger, symbol: str = "BTC-USDT-SWAP",
    side: str = "long", entry_price: float = 50_000.0,
    size: float = 0.1, entry_time_ms: int = 1_700_000_000_000,
) -> None:
    ledger.open_positions[symbol] = Position(
        symbol=symbol, side=side, entry_time=entry_time_ms,
        entry_price=entry_price, size=size, fees_paid=0.0,
        stop=entry_price * 0.98, take_profit=entry_price * 1.04,
        entry_snapshot={"close": entry_price},
        strategy_scores={"trend_following": {"long": 0.7, "short": 0.0}},
    )


def _closed_trade(pnl: float) -> dict:
    return {"pnl_after_fees": pnl, "exit_reason": "take_profit"}


# ============================================================
# Basic snapshot correctness
# ============================================================

def test_snapshot_flat_ledger_no_alerts():
    m = ShadowMetrics()
    ledger = _fresh_ledger()
    snap = m.snapshot(ledger, prices={}, last_seen_ts={}, now_ms=1_700_000_000_000)

    assert snap.equity == 10_000.0
    assert snap.total_pnl == 0.0
    assert snap.pnl_pct == 0.0
    assert snap.drawdown_from_peak_pct == 0.0
    assert snap.open_positions == 0
    assert snap.total_closed_trades == 0
    assert snap.winrate == 0.0
    assert snap.profit_factor == 0.0
    assert snap.alerts == []


def test_snapshot_counts_open_positions_and_notional():
    m = ShadowMetrics()
    ledger = _fresh_ledger()
    _open_position(ledger, "BTC-USDT-SWAP", "long", 50_000.0, 0.1)
    _open_position(ledger, "ETH-USDT-SWAP", "short", 3_000.0, 2.0)

    snap = m.snapshot(
        ledger,
        prices={"BTC-USDT-SWAP": 50_000.0, "ETH-USDT-SWAP": 3_000.0},
        last_seen_ts={"BTC-USDT-SWAP": 1_700_000_000_000},
        now_ms=1_700_000_000_000,
    )
    assert snap.open_positions == 2
    assert snap.open_long_notional == 50_000.0 * 0.1   # 5000
    assert snap.open_short_notional == 3_000.0 * 2.0   # 6000
    assert snap.open_notional == 11_000.0


def test_winrate_and_profit_factor():
    m = ShadowMetrics()
    ledger = _fresh_ledger()
    ledger.closed_trades = [
        _closed_trade(+100.0),
        _closed_trade(+50.0),
        _closed_trade(-30.0),
        _closed_trade(-20.0),
    ]
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert snap.winners == 2
    assert snap.losers == 2
    assert snap.winrate == 0.5
    assert snap.profit_factor == pytest.approx(150.0 / 50.0)  # 3.0


def test_profit_factor_infinite_when_no_losses():
    m = ShadowMetrics()
    ledger = _fresh_ledger()
    ledger.closed_trades = [_closed_trade(+100.0), _closed_trade(+50.0)]
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert snap.profit_factor == float("inf")


def test_profit_factor_zero_when_no_wins():
    m = ShadowMetrics()
    ledger = _fresh_ledger()
    ledger.closed_trades = [_closed_trade(-100.0)]
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert snap.profit_factor == 0.0


# ============================================================
# Peak equity + drawdown
# ============================================================

def test_peak_equity_only_ratchets_up():
    m = ShadowMetrics()
    ledger = _fresh_ledger(initial=10_000.0)
    ledger.equity = 12_000.0
    m.snapshot(ledger, prices={}, last_seen_ts={})
    assert m.peak_equity == 12_000.0
    ledger.equity = 11_000.0
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    # peak did not fall; drawdown from peak = (12k - 11k)/12k
    assert m.peak_equity == 12_000.0
    assert snap.drawdown_from_peak_pct == pytest.approx(1_000.0 / 12_000.0)


def test_drawdown_zero_when_at_peak():
    m = ShadowMetrics()
    ledger = _fresh_ledger()
    ledger.equity = 11_000.0
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert snap.drawdown_from_peak_pct == 0.0


# ============================================================
# Alerts — one per rule, plus the "no alert" boundary
# ============================================================

def test_alert_drawdown_high():
    m = ShadowMetrics(AlertThresholds(max_drawdown_pct=0.10))
    ledger = _fresh_ledger(initial=10_000.0)
    ledger.equity = 12_000.0
    m.snapshot(ledger, prices={}, last_seen_ts={})   # sets peak
    ledger.equity = 10_500.0                          # 12.5% drawdown
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert any("drawdown_high" in a for a in snap.alerts)


def test_no_drawdown_alert_at_threshold_boundary():
    """Threshold is >=; drawdown just below must NOT alert."""
    m = ShadowMetrics(AlertThresholds(max_drawdown_pct=0.10))
    ledger = _fresh_ledger()
    ledger.equity = 10_000.0
    m.snapshot(ledger, prices={}, last_seen_ts={})   # peak
    ledger.equity = 9_100.0    # 9% drawdown < 10%
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert not any("drawdown_high" in a for a in snap.alerts)


def test_alert_position_stuck():
    m = ShadowMetrics(AlertThresholds(stuck_position_hours=24.0))
    ledger = _fresh_ledger()
    now_ms = 1_700_000_000_000
    # Position opened 30 hours ago
    _open_position(ledger, entry_time_ms=now_ms - 30 * 3_600_000)
    snap = m.snapshot(ledger, prices={"BTC-USDT-SWAP": 50_000.0},
                       last_seen_ts={}, now_ms=now_ms)
    assert any("position_stuck:BTC-USDT-SWAP" in a for a in snap.alerts)


def test_alert_stale_data():
    m = ShadowMetrics(AlertThresholds(stale_data_minutes=60.0))
    ledger = _fresh_ledger()
    now_ms = 1_700_000_000_000
    stale_ts = now_ms - 120 * 60_000  # 2 hours ago
    snap = m.snapshot(ledger, prices={},
                       last_seen_ts={"BTC-USDT-SWAP": stale_ts}, now_ms=now_ms)
    assert any("stale_data" in a for a in snap.alerts)


def test_alert_consecutive_errors():
    m = ShadowMetrics(AlertThresholds(consecutive_error_limit=3))
    ledger = _fresh_ledger()
    m.note_tick_error()
    m.note_tick_error()
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert not any("consecutive_errors" in a for a in snap.alerts)
    m.note_tick_error()  # now at 3
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert any("consecutive_errors" in a for a in snap.alerts)


def test_note_success_resets_error_streak():
    m = ShadowMetrics()
    m.note_tick_error()
    m.note_tick_error()
    m.note_tick_success()
    assert m.consecutive_errors == 0


# ============================================================
# Persistence & startup hydration
# ============================================================

def test_append_history_writes_parquet(tmp_path: Path):
    m = ShadowMetrics(state_dir=tmp_path)
    ledger = _fresh_ledger()
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    path = m.append_history(snap)
    assert path is not None
    assert path.exists()
    hist = pd.read_parquet(path)
    assert len(hist) == 1
    assert "equity" in hist.columns
    assert "alert_count" in hist.columns


def test_append_history_no_op_without_state_dir():
    m = ShadowMetrics(state_dir=None)
    ledger = _fresh_ledger()
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    assert m.append_history(snap) is None


def test_hydrates_peak_from_prior_history(tmp_path: Path):
    """A restart in the same state_dir must recover peak_equity so
    drawdown reference isn't reset."""
    m1 = ShadowMetrics(state_dir=tmp_path)
    ledger = _fresh_ledger()
    ledger.equity = 12_500.0
    snap = m1.snapshot(ledger, prices={}, last_seen_ts={})
    m1.append_history(snap)

    m2 = ShadowMetrics(state_dir=tmp_path)
    assert m2.peak_equity == 12_500.0


def test_corrupt_history_does_not_crash_startup(tmp_path: Path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "metrics_history.parquet").write_bytes(b"not parquet")
    # Must not raise
    m = ShadowMetrics(state_dir=tmp_path)
    assert m.peak_equity == 0.0


def test_snapshot_to_dict_includes_alerts_and_count():
    m = ShadowMetrics(AlertThresholds(max_drawdown_pct=0.05))
    ledger = _fresh_ledger()
    ledger.equity = 12_000.0
    m.snapshot(ledger, prices={}, last_seen_ts={})
    ledger.equity = 10_000.0
    snap = m.snapshot(ledger, prices={}, last_seen_ts={})
    d = snap.to_dict()
    assert "alerts" in d
    assert "alert_count" in d
    assert d["alert_count"] >= 1
    assert isinstance(d["alerts"], str)
