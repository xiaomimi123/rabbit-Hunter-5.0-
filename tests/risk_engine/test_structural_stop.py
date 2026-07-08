"""Tests for structural stop placement (stop_mode="structural").

The stop goes just beyond the structure-invalidation level (swing low
for longs / swing high for shorts) + buffer, clamped to [min,max]×ATR,
with ATR fallback whenever the swing level is unusable. Sizing must
stay risk-normalized: wider stop → smaller position, same $ risk.
"""
from __future__ import annotations

import pytest

from rabbit_hunter.config.schema import RiskConfig
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext
from rabbit_hunter.strategy_router.router import Intent


def _cfg(**over) -> RiskConfig:
    kw = dict(risk_per_trade_pct=1.0, atr_stop_multiplier=2.5,
              reward_risk_ratio=2.5, max_leverage=10,
              daily_max_loss_pct=3.0, hold_timeout_bars=96,
              stop_mode="structural",
              structural_buffer_atr_mult=0.25,
              structural_min_atr_mult=1.0,
              structural_max_atr_mult=4.0)
    kw.update(over)
    return RiskConfig(**kw)


def _intent(action: str, **feats) -> Intent:
    return Intent(symbol="BTC-USDT-SWAP", action=action, conviction=0.8,
                   features_snapshot=feats)


def _ctx(price=100.0, atr=2.0) -> RiskContext:
    return RiskContext(equity=10_000.0, atr=atr, price=price,
                        daily_realized_pnl=0.0, initial_capital=10_000.0,
                        open_positions_count=0)


# ============================================================
# Structural placement
# ============================================================

def test_long_stop_below_swing_low_with_buffer():
    """price 100, swing_low 95, atr 2, buffer 0.25×atr=0.5
    → distance = 5 + 0.5 = 5.5 → stop at 94.5 (within 4×atr=8 cap)."""
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=95.0), _ctx())
    assert order is not None
    assert order.stop_price == pytest.approx(100.0 - 5.5)


def test_short_stop_above_swing_high_with_buffer():
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_short", swing_high_last=104.0), _ctx())
    # distance = 4 + 0.5 = 4.5 → stop 104.5
    assert order.stop_price == pytest.approx(100.0 + 4.5)


def test_sizing_stays_risk_normalized():
    """Wider structural stop → smaller size, same $ risk (1% = $100)."""
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=95.0), _ctx())
    stop_distance = 100.0 - order.stop_price
    assert order.size * stop_distance == pytest.approx(100.0)  # $100 risk


def test_take_profit_scales_with_structural_distance():
    """TP = entry + RR × stop_distance — RR logic unchanged."""
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=95.0), _ctx())
    dist = 100.0 - order.stop_price
    assert order.take_profit_price == pytest.approx(100.0 + 2.5 * dist)


# ============================================================
# Clamps
# ============================================================

def test_too_tight_swing_clamped_to_min():
    """swing_low at 99.9 (0.1 away) → clamp to 1×atr=2.0 min distance."""
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=99.9), _ctx())
    assert order.stop_price == pytest.approx(100.0 - 2.0)


def test_too_wide_swing_clamped_to_max():
    """swing_low at 80 (20 away) → clamp to 4×atr=8 max distance."""
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=80.0), _ctx())
    assert order.stop_price == pytest.approx(100.0 - 8.0)


# ============================================================
# ATR fallback
# ============================================================

def test_missing_swing_falls_back_to_atr():
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long"), _ctx())
    # legacy: 2.5 × 2.0 = 5.0
    assert order.stop_price == pytest.approx(100.0 - 5.0)


def test_nan_swing_falls_back_to_atr():
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=float("nan")), _ctx())
    assert order.stop_price == pytest.approx(100.0 - 5.0)


def test_wrong_side_swing_falls_back_to_atr():
    """swing_low ABOVE price (structure broken) → fallback."""
    eng = RiskEngine(_cfg())
    order = eng.size(_intent("open_long", swing_low_last=101.0), _ctx())
    assert order.stop_price == pytest.approx(100.0 - 5.0)


# ============================================================
# Legacy mode untouched
# ============================================================

def test_atr_mode_ignores_swing_features():
    eng = RiskEngine(_cfg(stop_mode="atr"))
    order = eng.size(_intent("open_long", swing_low_last=95.0), _ctx())
    assert order.stop_price == pytest.approx(100.0 - 5.0)
