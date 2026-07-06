import math
from rabbit_hunter.config.schema import ExecutionConfig, FeeConfig
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor, Fill
from rabbit_hunter.risk_engine.position_sizing import Order


def _cfg():
    return ExecutionConfig(
        fees=FeeConfig(maker=0.0002, taker=0.0005),
        slippage_atr_multiplier=0.1,
        funding_settlement=True,
    )


def _order(side="long", price=50_000.0, size=0.1):
    stop = price - 150 if side == "long" else price + 150
    tp = price + 300 if side == "long" else price - 300
    return Order(symbol="BTC-USDT-SWAP", side=side, entry_price=price,
                 stop_price=stop, take_profit_price=tp, size=size, leverage=1.0)


def test_long_fill_price_includes_slippage():
    e = BacktestExecutor(_cfg())
    next_bar = {"timestamp": 1, "open": 50_000.0, "high": 50_100.0, "low": 49_900.0, "close": 50_050.0}
    fill = e.submit(_order("long"), next_bar, atr=100.0)
    # 多头买入 → 成交价 = open + slippage_atr_multiplier * atr
    assert math.isclose(fill.fill_price, 50_000.0 + 0.1 * 100.0)
    assert fill.side == "long"
    # taker 费率
    assert math.isclose(fill.fees, 50_010.0 * 0.1 * 0.0005, rel_tol=1e-6)


def test_short_fill_price_slips_down():
    e = BacktestExecutor(_cfg())
    next_bar = {"timestamp": 1, "open": 50_000.0, "high": 50_100.0, "low": 49_900.0, "close": 49_950.0}
    fill = e.submit(_order("short"), next_bar, atr=100.0)
    assert math.isclose(fill.fill_price, 50_000.0 - 0.1 * 100.0)


def test_apply_funding_long_pays_when_positive():
    e = BacktestExecutor(_cfg())
    # 多头持仓，funding 为正 → 多头付钱（负 pnl）
    delta = e.apply_funding(position_size=0.1, price=50_000.0, funding_rate=0.0001)
    assert delta < 0
    assert math.isclose(delta, -0.1 * 50_000.0 * 0.0001)


def test_apply_funding_short_receives_when_positive():
    e = BacktestExecutor(_cfg())
    delta = e.apply_funding(position_size=-0.1, price=50_000.0, funding_rate=0.0001)
    assert delta > 0
    assert math.isclose(delta, 0.1 * 50_000.0 * 0.0001)
