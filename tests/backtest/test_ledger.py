import math
from rabbit_hunter.backtest.ledger import Ledger, Position, TrailingConfig
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


def test_trailing_activates_at_1r_profit_and_moves_stop_up_only():
    """Long position: after price reaches 1R profit, trailing stop must
    kick in and move upward; a subsequent lower high must NOT drop it."""
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=1.0, slippage=0.0, reason="entry")
    # Initial stop $400 below entry → 1R = $400. TP way above so it doesn't fire.
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_600.0, take_profit=55_000.0)
    ex = _ex()
    trailing = TrailingConfig(enabled=True, activation_r=1.0, atr_multiplier=1.0)
    # atr = $400 → trail distance = $400

    # Bar 1: high $50_200 = 0.5R profit → trailing NOT active yet
    bar1 = {"timestamp": 1001, "open": 50_000.0, "high": 50_200.0, "low": 49_900.0, "close": 50_150.0}
    ledger.check_exits("BTC-USDT-SWAP", bar1, atr=400.0, executor=ex,
                       hold_timeout_bars=100, exit_snapshot_fn=lambda: {}, trailing=trailing)
    pos = ledger.open_positions["BTC-USDT-SWAP"]
    assert pos.trailing_active is False
    assert pos.stop == 49_600.0
    assert pos.max_favorable_price == 50_200.0

    # Bar 2: high $50_500 = 1.25R profit → trailing MUST activate
    # new_trail = 50_500 - 400 = 50_100 → pos.stop = max(49_600, 50_100) = 50_100
    bar2 = {"timestamp": 1002, "open": 50_150.0, "high": 50_500.0, "low": 50_150.0, "close": 50_400.0}
    ledger.check_exits("BTC-USDT-SWAP", bar2, atr=400.0, executor=ex,
                       hold_timeout_bars=100, exit_snapshot_fn=lambda: {}, trailing=trailing)
    pos = ledger.open_positions["BTC-USDT-SWAP"]
    assert pos.trailing_active is True
    assert pos.max_favorable_price == 50_500.0
    assert math.isclose(pos.stop, 50_100.0)

    # Bar 3: high $50_400 (LOWER than prev HW), close $50_350
    #   HW should stay 50_500 (not overwritten by lower high)
    #   trail_stop would be 50_500 - 400 = 50_100 → no change (not lower)
    bar3 = {"timestamp": 1003, "open": 50_400.0, "high": 50_400.0, "low": 50_200.0, "close": 50_350.0}
    ledger.check_exits("BTC-USDT-SWAP", bar3, atr=400.0, executor=ex,
                       hold_timeout_bars=100, exit_snapshot_fn=lambda: {}, trailing=trailing)
    pos = ledger.open_positions["BTC-USDT-SWAP"]
    assert pos.max_favorable_price == 50_500.0  # unchanged
    assert math.isclose(pos.stop, 50_100.0)     # unchanged (never moves down)


def test_trailing_stop_hit_marked_as_trailing_stop_reason():
    """When a trailing-active position is stopped out, the trade's exit_reason
    must be `trailing_stop` (not the pre-trailing `stop_loss`)."""
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=1.0, slippage=0.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_600.0, take_profit=55_000.0)
    ex = _ex()
    trailing = TrailingConfig(enabled=True, activation_r=1.0, atr_multiplier=1.0)

    # Push up to activate trailing at $50_500 → stop lifts to $50_100.
    # `low` must stay ABOVE the new trailing stop so this bar itself doesn't
    # exit — we want to check the next-bar dump triggers `trailing_stop`.
    bar_up = {"timestamp": 1001, "open": 50_000.0, "high": 50_500.0, "low": 50_200.0, "close": 50_400.0}
    ledger.check_exits("BTC-USDT-SWAP", bar_up, atr=400.0, executor=ex,
                       hold_timeout_bars=100, exit_snapshot_fn=lambda: {}, trailing=trailing)
    assert ledger.open_positions["BTC-USDT-SWAP"].trailing_active is True

    # Next bar dumps below the trailing stop $50_100 → must exit as trailing_stop
    bar_dump = {"timestamp": 1002, "open": 50_400.0, "high": 50_400.0, "low": 50_050.0, "close": 50_090.0}
    trades = ledger.check_exits("BTC-USDT-SWAP", bar_dump, atr=400.0, executor=ex,
                                hold_timeout_bars=100, exit_snapshot_fn=lambda: {}, trailing=trailing)
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "trailing_stop"


def test_trailing_disabled_preserves_prior_behavior():
    """When trailing.enabled=False (default), Position.stop is never mutated
    and exit_reason falls back to `stop_loss` — behavior identical to
    pre-v0.2.0 Ledger."""
    ledger = Ledger(initial_capital=10_000.0)
    entry = Fill("BTC-USDT-SWAP", "long", 50_000.0, 0.01, 1000, fees=1.0, slippage=0.0, reason="entry")
    ledger.record_entry(entry, entry_snapshot={}, strategy_scores={},
                        stop=49_600.0, take_profit=51_000.0)
    ex = _ex()

    # Big move up that WOULD have triggered trailing if enabled, then stop-out.
    # No trailing arg = defaults to TRAILING_OFF.
    bar_up = {"timestamp": 1001, "open": 50_000.0, "high": 50_800.0, "low": 49_900.0, "close": 50_700.0}
    ledger.check_exits("BTC-USDT-SWAP", bar_up, atr=400.0, executor=ex,
                       hold_timeout_bars=100, exit_snapshot_fn=lambda: {})
    pos = ledger.open_positions["BTC-USDT-SWAP"]
    assert pos.trailing_active is False
    assert pos.stop == 49_600.0  # never mutated

    # Dump below original stop → exit as stop_loss (not trailing_stop)
    bar_dump = {"timestamp": 1002, "open": 50_700.0, "high": 50_700.0, "low": 49_500.0, "close": 49_550.0}
    trades = ledger.check_exits("BTC-USDT-SWAP", bar_dump, atr=400.0, executor=ex,
                                hold_timeout_bars=100, exit_snapshot_fn=lambda: {})
    assert len(trades) == 1
    assert trades[0]["exit_reason"] == "stop_loss"
