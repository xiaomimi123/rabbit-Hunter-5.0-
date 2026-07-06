"""Unit tests for v0.1.3 BtcCrashBooster.

The booster is a narrow, deliberate size uplift on short-side orders when
BTC itself is in a systemic-crash bar. Every branch that decides "boost"
vs "no boost" must be pinned by a test so the invariant survives future
refactors.
"""
from __future__ import annotations

import math

from rabbit_hunter.config.schema import BtcCrashBoostConfig
from rabbit_hunter.risk_engine.btc_crash_booster import BtcCrashBooster
from rabbit_hunter.risk_engine.position_sizing import Order


def _order(side: str = "short", size: float = 10.0, price: float = 100.0) -> Order:
    return Order(
        symbol="ETH-USDT-SWAP",
        side=side,
        entry_price=price,
        stop_price=price * (1.02 if side == "short" else 0.98),
        take_profit_price=price * (0.94 if side == "short" else 1.06),
        size=size,
        leverage=1.0,
    )


def _cfg(**overrides) -> BtcCrashBoostConfig:
    kw = dict(enabled=True, btc_symbol="BTC-USDT-SWAP",
              zscore_threshold=2.0, boost_multiplier=1.5)
    kw.update(overrides)
    return BtcCrashBoostConfig(**kw)


def _crash_row(z: float = -2.5, close: float = 50_000.0) -> dict:
    return {"zscore_20": z, "close": close}


def test_boost_applies_on_crash_short():
    """Short order + BTC z ≤ -2 + BTC falling → boost applied."""
    b = BtcCrashBooster(_cfg())
    result = b.evaluate(
        candidate=_order("short", size=10.0),
        btc_row=_crash_row(z=-2.5, close=50_000.0),
        btc_prev_close=51_000.0,  # BTC fell
        equity=10_000.0,
    )
    assert result.boosted is True
    assert result.multiplier == 1.5
    assert math.isclose(result.adjusted_order.size, 15.0)  # 10 * 1.5
    assert "btc_crash" in result.reason


def test_no_boost_when_disabled():
    b = BtcCrashBooster(_cfg(enabled=False))
    result = b.evaluate(
        candidate=_order("short"),
        btc_row=_crash_row(z=-3.0),
        btc_prev_close=51_000.0,
        equity=10_000.0,
    )
    assert result.boosted is False
    assert result.multiplier == 1.0


def test_no_boost_for_long_orders():
    """The booster is deliberately short-side-only — the cluster study
    only measured the edge on shorts."""
    b = BtcCrashBooster(_cfg())
    result = b.evaluate(
        candidate=_order("long"),
        btc_row=_crash_row(z=-3.0),
        btc_prev_close=51_000.0,
        equity=10_000.0,
    )
    assert result.boosted is False


def test_no_boost_when_z_above_threshold():
    """z = -1.5 is below threshold |z|=2.0 → skip."""
    b = BtcCrashBooster(_cfg(zscore_threshold=2.0))
    result = b.evaluate(
        candidate=_order("short"),
        btc_row=_crash_row(z=-1.5),
        btc_prev_close=51_000.0,
        equity=10_000.0,
    )
    assert result.boosted is False


def test_no_boost_when_btc_rebounding():
    """z-score still deeply negative BUT BTC's current close is above
    prior close = rebound bar, not a fresh crash → skip. Prevents
    late-entry on the recovery leg."""
    b = BtcCrashBooster(_cfg())
    result = b.evaluate(
        candidate=_order("short"),
        btc_row=_crash_row(z=-2.5, close=51_500.0),
        btc_prev_close=51_000.0,  # BTC rose
        equity=10_000.0,
    )
    assert result.boosted is False


def test_no_boost_when_btc_row_missing():
    """BTC not in the symbol basket → booster is a no-op, not a crash."""
    b = BtcCrashBooster(_cfg())
    result = b.evaluate(
        candidate=_order("short"),
        btc_row=None,
        btc_prev_close=None,
        equity=10_000.0,
    )
    assert result.boosted is False


def test_no_boost_when_prev_close_missing():
    """First bar has no prior close → skip (can't verify direction)."""
    b = BtcCrashBooster(_cfg())
    result = b.evaluate(
        candidate=_order("short"),
        btc_row=_crash_row(z=-3.0),
        btc_prev_close=None,
        equity=10_000.0,
    )
    assert result.boosted is False


def test_nan_zscore_handled_as_neutral():
    """NaN z-score must not trigger a boost."""
    b = BtcCrashBooster(_cfg())
    result = b.evaluate(
        candidate=_order("short"),
        btc_row={"zscore_20": float("nan"), "close": 50_000.0},
        btc_prev_close=51_000.0,
        equity=10_000.0,
    )
    assert result.boosted is False


def test_leverage_recomputed_after_boost():
    """Boosted size must produce a matching leverage figure — otherwise
    the risk snapshot lies about exposure."""
    b = BtcCrashBooster(_cfg(boost_multiplier=1.2))
    order = _order("short", size=100.0, price=50.0)  # 5000 notional
    result = b.evaluate(
        candidate=order,
        btc_row=_crash_row(z=-2.5),
        btc_prev_close=51_000.0,
        equity=10_000.0,
    )
    # new size = 120, new notional = 6000, leverage = 0.6
    assert math.isclose(result.adjusted_order.size, 120.0)
    assert math.isclose(result.adjusted_order.leverage, 0.6)


def test_boost_at_exact_threshold_triggers():
    """z-score at exactly -threshold must still fire (inclusive)."""
    b = BtcCrashBooster(_cfg(zscore_threshold=2.0))
    result = b.evaluate(
        candidate=_order("short"),
        btc_row=_crash_row(z=-2.0),
        btc_prev_close=51_000.0,
        equity=10_000.0,
    )
    assert result.boosted is True
