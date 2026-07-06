import math
from rabbit_hunter.config.schema import RiskConfig
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext, Order
from rabbit_hunter.strategy_router.router import Intent


def _cfg():
    return RiskConfig(
        risk_per_trade_pct=1.0,
        atr_stop_multiplier=1.5,
        reward_risk_ratio=2.0,
        max_leverage=3,
        daily_max_loss_pct=3.0,
        hold_timeout_bars=48,
    )


def _intent(action="open_long"):
    return Intent(symbol="BTC-USDT-SWAP", action=action, conviction=0.7)


def _ctx(**kw):
    base = dict(equity=10_000.0, atr=100.0, price=50_000.0,
                daily_realized_pnl=0.0, initial_capital=10_000.0, open_positions_count=0)
    base.update(kw)
    return RiskContext(**base)


def test_long_order_sizing():
    e = RiskEngine(_cfg())
    order = e.size(_intent("open_long"), _ctx())
    assert isinstance(order, Order)
    assert order.side == "long"
    assert order.entry_price == 50_000.0
    assert math.isclose(order.stop_price, 50_000.0 - 1.5 * 100.0)
    assert math.isclose(order.take_profit_price, 50_000.0 + 2.0 * 1.5 * 100.0)
    # uncapped size = risk / stop_distance = 10000 * 0.01 / 150 ≈ 0.6667
    # notional = 0.6667 * 50000 ≈ 33333.33 > equity * max_leverage (30000) → capped
    # capped size = equity * max_leverage / price = 10000 * 3 / 50000 = 0.6
    assert math.isclose(order.size, 10_000.0 * 3 / 50_000.0, rel_tol=1e-6)
    assert order.leverage <= 3


def test_short_order_sizing():
    e = RiskEngine(_cfg())
    order = e.size(_intent("open_short"), _ctx())
    assert order.side == "short"
    assert math.isclose(order.stop_price, 50_000.0 + 1.5 * 100.0)
    assert math.isclose(order.take_profit_price, 50_000.0 - 2.0 * 1.5 * 100.0)


def test_wait_returns_none():
    e = RiskEngine(_cfg())
    assert e.size(_intent("wait"), _ctx()) is None


def test_daily_circuit_blocks():
    e = RiskEngine(_cfg())
    ctx = _ctx(daily_realized_pnl=-400.0)  # 4% loss on 10000
    assert e.size(_intent("open_long"), ctx) is None


def test_leverage_cap():
    e = RiskEngine(_cfg())
    # ATR 很小 → 名义仓位 = size * price 可能 > equity * max_leverage → 应该被 cap
    order = e.size(_intent("open_long"), _ctx(atr=1.0))
    assert order.leverage <= 3
    assert order.size * order.entry_price <= 3 * 10_000.0 + 1e-6
