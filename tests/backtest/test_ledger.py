import math
from rabbit_hunter.backtest.ledger import Ledger, Position
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor
from rabbit_hunter.config.schema import ExecutionConfig, FeeConfig
from rabbit_hunter.execution_engine.base import Fill


def _ex():
    return BacktestExecutor(ExecutionConfig(
        fees=FeeConfig(maker=0.0002, taker=0.0005),
        slippage_atr_multiplier=0.1,
        funding_settlement=True,
    ))


def test_open_and_close_long_produces_trade():
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=25.0, slippage=10.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={"ema20": 100.0}, strategy_scores={"trend_following": 0.7},
                        stop=49_800.0, take_profit=50_400.0)
    assert "BTC-USDT-SWAP" in ledger.open_positions

    exit_ = Fill("BTC-USDT-SWAP", "long", 50_300.0, 0.01, 2000, fees=25.15, slippage=10.0, reason="take_profit")
    trade = ledger.record_exit(exit_, exit_snapshot={"ema20": 105.0})
    assert trade is not None
    assert trade["side"] == "long"
    assert trade["exit_reason"] == "take_profit"
    assert math.isclose(trade["pnl_raw"], (50_300.0 - 50_000.0) * 0.01)
    assert math.isclose(trade["pnl_after_fees"], trade["pnl_raw"] - trade["fees"])
    assert "BTC-USDT-SWAP" not in ledger.open_positions


def test_stop_loss_triggers_on_long():
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=25.0, slippage=10.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_800.0, take_profit=50_400.0)
    ex = _ex()
    bar = {"timestamp": 2000, "open": 50_000.0, "high": 50_050.0, "low": 49_700.0, "close": 49_750.0}
    trades = ledger.check_exits("BTC-USDT-SWAP", bar, atr=50.0, executor=ex, hold_timeout_bars=10, exit_snapshot_fn=lambda: {})
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "stop_loss"


def test_timeout_exit_at_close():
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=25.0, slippage=10.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_000.0, take_profit=51_000.0)
    ex = _ex()
    for i in range(11):
        bar = {"timestamp": 2000 + i, "open": 50_000.0, "high": 50_010.0, "low": 49_990.0, "close": 50_005.0}
        trades = ledger.check_exits("BTC-USDT-SWAP", bar, atr=10.0, executor=ex, hold_timeout_bars=10, exit_snapshot_fn=lambda: {})
    assert any(t["exit_reason"] == "timeout" for t in ledger.closed_trades)


def test_funding_not_double_counted_on_close():
    """Funding realized via apply_funding must not be added again on close."""
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=10.0, slippage=0.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={}, stop=49_000.0, take_profit=51_000.0)
    # equity: 10_000 - 10 (fees) = 9_990
    assert ledger.equity == 9_990.0

    # Apply funding twice (positive rates → long pays, negative delta).
    # BacktestExecutor.apply_funding: delta = -size * price * funding_rate
    # size=0.01, price=50_000.0 → delta1 = -0.01*50_000*0.0001 = -0.05, delta2 = -0.01*50_000*0.0002 = -0.1
    ex = _ex()
    ledger.apply_funding("BTC-USDT-SWAP", 50_000.0, 0.0001, ex)  # delta = -0.05
    ledger.apply_funding("BTC-USDT-SWAP", 50_000.0, 0.0002, ex)  # delta = -0.1
    # equity: 9_990 - 0.05 - 0.1 = 9_989.85
    assert math.isclose(ledger.equity, 9_989.85, abs_tol=1e-9)

    # Close: exit price 50_100, exit fee 5.0 → pnl_raw = 1.0
    exit_ = Fill("BTC-USDT-SWAP", "long", 50_100.0, 0.01, 5000, fees=5.0, slippage=0.0, reason="take_profit")
    trade = ledger.record_exit(exit_, exit_snapshot={})
    # pnl_after_fees in Trade dict = pnl_raw - total_fees + funding = 1.0 - 15.0 + (-0.15) = -14.15
    assert math.isclose(trade["pnl_after_fees"], -14.15, abs_tol=1e-9)
    # equity change from exit = pnl_raw - exit_fee = 1.0 - 5.0 = -4.0
    # final equity = 9_989.85 - 4.0 = 9_985.85 (NOT 9_985.70 which would be double-counting-fix-not-applied)
    assert math.isclose(ledger.equity, 9_985.85, abs_tol=1e-9)
