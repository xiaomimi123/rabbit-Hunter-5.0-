"""Unit tests for v0.1.3 chop-market kill switch (BacktestEngine).

Rather than construct a full backtest that happens to lose 65% of 10 trades
(fragile and slow), we drive the engine's `_on_trade_closed` hook directly
and inspect the internal state. The gate logic in `run()` is a single
`ts < paused_until_ts` compare, so once the state transitions are proven
here, the runtime behavior follows.
"""
from __future__ import annotations

from rabbit_hunter.config.schema import (
    AppConfig, DataConfig, FeatureEngineConfig, StrategyRouterConfig,
    StrategyEntry, RiskConfig, ExecutionConfig, FeeConfig, BacktestConfig,
    ReportConfig, ChopKillSwitchConfig,
)
from rabbit_hunter.backtest.engine import BacktestEngine


def _cfg(chop: ChopKillSwitchConfig) -> AppConfig:
    return AppConfig(
        data=DataConfig(exchange="okx", symbols=["BTC-USDT-SWAP"],
                        main_interval="1H", confirm_interval="15m",
                        history_window_days=30),
        feature_engine=FeatureEngineConfig(version="0.1.0"),
        strategy_router=StrategyRouterConfig(
            composer="weighted_avg",
            enabled_strategies={
                "trend_following": StrategyEntry(
                    weight=1.0, config_file="strategies/trend_following.yaml",
                ),
            },
        ),
        risk=RiskConfig(risk_per_trade_pct=1.0, atr_stop_multiplier=1.5,
                        reward_risk_ratio=2.0, max_leverage=3,
                        daily_max_loss_pct=3.0, hold_timeout_bars=48),
        execution=ExecutionConfig(fees=FeeConfig(maker=0.0002, taker=0.0005),
                                  slippage_atr_multiplier=0.1,
                                  funding_settlement=True),
        backtest=BacktestConfig(start="2025-01-01", end="2025-02-01",
                                initial_capital=10_000.0),
        report=ReportConfig(),
        chop_kill_switch=chop,
    )


def _engine(**chop_overrides) -> BacktestEngine:
    chop = ChopKillSwitchConfig(**{
        "enabled": True, "window": 4, "wr_threshold": 0.5, "pause_bars": 10,
        **chop_overrides,
    })
    return BacktestEngine(_cfg(chop), strategies=[])  # strategies unused here


BAR_MS = 3_600_000
BASE_TS = 1_700_000_000_000


def _close(engine, pnl: float, ts: int | None = None) -> None:
    """Feed one closed-trade event into the kill switch."""
    engine._on_trade_closed({"pnl_after_fees": pnl}, ts or BASE_TS)


def test_disabled_switch_never_pauses():
    e = _engine(enabled=False)
    for _ in range(10):
        _close(e, -100.0)
    assert e._paused_until_ts is None


def test_no_pause_before_window_full():
    """A single loser must not trip the switch — otherwise the first
    bad trade of every session pauses the engine."""
    e = _engine(window=4, wr_threshold=0.5)
    _close(e, -100.0)
    assert e._paused_until_ts is None
    _close(e, -100.0)
    assert e._paused_until_ts is None
    _close(e, -100.0)
    assert e._paused_until_ts is None


def test_pauses_when_window_full_and_wr_below_threshold():
    """Four consecutive losses fills the window with WR=0% < 50% → pause."""
    e = _engine(window=4, wr_threshold=0.5, pause_bars=10)
    for _ in range(4):
        _close(e, -100.0, ts=BASE_TS)
    expected_pause_end = BASE_TS + 10 * BAR_MS
    assert e._paused_until_ts == expected_pause_end


def test_does_not_pause_when_wr_meets_threshold():
    """3 winners + 1 loser = 75% WR ≥ 50% → do not pause."""
    e = _engine(window=4, wr_threshold=0.5)
    _close(e, +100.0)
    _close(e, +100.0)
    _close(e, +100.0)
    _close(e, -100.0)
    assert e._paused_until_ts is None


def test_wr_boundary_at_threshold_does_not_pause():
    """WR exactly at threshold must not pause (strict less-than semantics)."""
    e = _engine(window=4, wr_threshold=0.5)
    _close(e, +100.0)
    _close(e, +100.0)
    _close(e, -100.0)
    _close(e, -100.0)   # WR = 0.5, exactly threshold
    assert e._paused_until_ts is None


def test_window_resets_after_pause_engaged():
    """Once paused, the rolling window clears — otherwise the same losers
    re-trigger the pause on every subsequent close and it never releases."""
    e = _engine(window=4, wr_threshold=0.5, pause_bars=10)
    for _ in range(4):
        _close(e, -100.0)
    assert e._paused_until_ts is not None
    assert len(e._recent_pnl) == 0


def test_pause_reengages_on_new_losing_streak():
    """After the window resets, a fresh 4-loser streak must re-engage
    the switch (with a later paused_until_ts)."""
    e = _engine(window=4, wr_threshold=0.5, pause_bars=10)
    for _ in range(4):
        _close(e, -100.0, ts=BASE_TS)
    first_pause = e._paused_until_ts
    later_ts = BASE_TS + 20 * BAR_MS
    for _ in range(4):
        _close(e, -100.0, ts=later_ts)
    assert e._paused_until_ts == later_ts + 10 * BAR_MS
    assert e._paused_until_ts > first_pause


def test_rolling_window_evicts_oldest():
    """With window=4, adding a 5th trade should evict the first —
    proving deque(maxlen=window) semantics. Three winners + one recent
    loser must NOT trip even if the evicted trade was a loser."""
    e = _engine(window=4, wr_threshold=0.5)
    _close(e, -100.0)   # will be evicted
    _close(e, +100.0)
    _close(e, +100.0)
    _close(e, +100.0)   # window: [-, +, +, +] → WR 75% (does not pause)
    assert e._paused_until_ts is None
    _close(e, -100.0)   # evict oldest -, window becomes [+, +, +, -] → WR 75%
    assert e._paused_until_ts is None


def test_zero_pnl_counts_as_loser():
    """A trade with pnl_after_fees == 0 is not a winner. Boundary against
    a strict `>0` check silently reclassifying breakevens as winners."""
    e = _engine(window=4, wr_threshold=0.5)
    _close(e, 0.0)
    _close(e, 0.0)
    _close(e, 0.0)
    _close(e, 0.0)  # WR = 0%
    assert e._paused_until_ts is not None
