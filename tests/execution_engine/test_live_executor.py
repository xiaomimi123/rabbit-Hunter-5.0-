"""Tests for LiveExecutor — the safety envelope MUST hold.

Every branch that could reach a real exchange is covered:
  - enabled=False → simulated fallback runs, no exchange call
  - enabled=True + missing env → refuses to construct
  - enabled=True + oversized notional → refuses to submit
  - enabled=True + normal path → calls the (mocked) exchange with the
    expected side/size/params and returns a Fill with real fields
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rabbit_hunter.config.schema import (
    ExecutionConfig, FeeConfig, LiveExecutionConfig,
)
from rabbit_hunter.execution_engine.live_executor import (
    LiveExecutor, LiveExecutionError,
)
from rabbit_hunter.risk_engine.position_sizing import Order


def _exec_cfg() -> ExecutionConfig:
    return ExecutionConfig(
        fees=FeeConfig(maker=0.0002, taker=0.0005),
        slippage_atr_multiplier=0.1, funding_settlement=True,
    )


def _live_cfg(**overrides) -> LiveExecutionConfig:
    kw = dict(enabled=False, exchange="okx", testnet=True,
              max_notional_per_order=1_000.0)
    kw.update(overrides)
    return LiveExecutionConfig(**kw)


def _order(side: str = "long", size: float = 0.01,
           price: float = 50_000.0) -> Order:
    return Order(
        symbol="BTC-USDT-SWAP", side=side, entry_price=price,
        stop_price=price * 0.98 if side == "long" else price * 1.02,
        take_profit_price=price * 1.04 if side == "long" else price * 0.96,
        size=size, leverage=1.0,
    )


def _next_bar(price: float = 50_000.0, ts: int = 1_700_000_000_000) -> dict:
    return {"timestamp": ts, "open": price, "high": price + 100,
            "low": price - 100, "close": price + 50}


# ============================================================
# Disabled path — pure paper behavior
# ============================================================

def test_disabled_submit_returns_simulated_fill():
    ex = LiveExecutor(_exec_cfg(), _live_cfg(enabled=False))
    fill = ex.submit(_order("long", size=0.01), _next_bar(), atr=100.0)
    # slippage = 0.1 × 100 = 10 → long fills open + 10 = 50010
    assert fill.fill_price == pytest.approx(50_010.0)
    assert fill.reason == "entry"
    assert ex._exchange is None       # never built


def test_disabled_close_returns_simulated_fill():
    ex = LiveExecutor(_exec_cfg(), _live_cfg(enabled=False))
    fill = ex.close_at(
        symbol="BTC-USDT-SWAP", side="long", size=0.01,
        price=50_000.0, timestamp=1, atr=100.0,
        reason="stop_loss",
    )
    # closing long → sell → fill = price − slip = 49990
    assert fill.fill_price == pytest.approx(49_990.0)
    assert fill.reason == "stop_loss"


def test_apply_funding_matches_backtest_math():
    ex = LiveExecutor(_exec_cfg(), _live_cfg(enabled=False))
    # size=+1, price=100, funding=+0.001 → -0.1 for the long
    assert ex.apply_funding(1.0, 100.0, 0.001) == pytest.approx(-0.1)


# ============================================================
# Enabled path — construction safeguards
# ============================================================

def test_enabled_without_env_credentials_refuses(monkeypatch):
    monkeypatch.delenv("OKX_API_KEY", raising=False)
    monkeypatch.delenv("OKX_API_SECRET", raising=False)
    monkeypatch.delenv("OKX_API_PASSPHRASE", raising=False)
    with pytest.raises(LiveExecutionError) as ei:
        LiveExecutor(_exec_cfg(), _live_cfg(enabled=True))
    assert "empty" in str(ei.value)


def test_enabled_with_credentials_builds_via_factory():
    """When a factory is injected, __init__ uses it instead of touching
    ccxt — proves construction wiring works without hitting the network."""
    mock_ex = MagicMock()
    ex = LiveExecutor(
        _exec_cfg(), _live_cfg(enabled=True),
        exchange_factory=lambda: mock_ex,
    )
    assert ex._exchange is mock_ex


# ============================================================
# Enabled path — notional cap
# ============================================================

def test_oversized_order_refused_before_hitting_exchange():
    mock_ex = MagicMock()
    ex = LiveExecutor(
        _exec_cfg(),
        _live_cfg(enabled=True, max_notional_per_order=1_000.0),
        exchange_factory=lambda: mock_ex,
    )
    # notional = 60_000 × 0.05 = 3_000, above the 1_000 cap
    with pytest.raises(LiveExecutionError) as ei:
        ex.submit(_order(size=0.05, price=60_000.0), _next_bar(), atr=100.0)
    assert "exceeds" in str(ei.value)
    mock_ex.create_market_order.assert_not_called()


# ============================================================
# Enabled path — real order placement (via mock exchange)
# ============================================================

def test_enabled_submit_calls_exchange_with_correct_side_and_size():
    """The mock stands in for an ExchangeAdapter now (LiveExecutor
    delegates symbol conversion to the adapter), so create_market_order
    receives the internal-form symbol."""
    mock_ex = MagicMock()
    mock_ex.create_market_order.return_value = {
        "average": 50_010.0, "filled": 0.01, "fee": {"cost": 0.25},
    }
    ex = LiveExecutor(
        _exec_cfg(),
        _live_cfg(enabled=True, max_notional_per_order=10_000.0),
        exchange_factory=lambda: mock_ex,
    )
    fill = ex.submit(_order("long", size=0.01, price=50_000.0),
                     _next_bar(), atr=100.0)
    mock_ex.create_market_order.assert_called_once()
    args = mock_ex.create_market_order.call_args
    assert args.kwargs["symbol"] == "BTC-USDT-SWAP"    # internal form
    assert args.kwargs["side"] == "buy"
    assert args.kwargs["amount"] == 0.01
    assert args.kwargs["params"]["tdMode"] == "cross"
    assert fill.fill_price == 50_010.0
    assert fill.size == 0.01
    assert fill.fees == 0.25
    assert fill.reason == "entry_live"


def test_enabled_short_submit_sends_sell_side():
    mock_ex = MagicMock()
    mock_ex.create_market_order.return_value = {
        "average": 49_990.0, "filled": 0.01, "fee": {"cost": 0.25},
    }
    ex = LiveExecutor(
        _exec_cfg(),
        _live_cfg(enabled=True, max_notional_per_order=10_000.0),
        exchange_factory=lambda: mock_ex,
    )
    ex.submit(_order("short", size=0.01), _next_bar(), atr=100.0)
    assert mock_ex.create_market_order.call_args.kwargs["side"] == "sell"


def test_enabled_close_sends_reduce_only():
    mock_ex = MagicMock()
    mock_ex.create_market_order.return_value = {
        "average": 49_990.0, "fee": {"cost": 0.25},
    }
    ex = LiveExecutor(
        _exec_cfg(),
        _live_cfg(enabled=True, max_notional_per_order=10_000.0),
        exchange_factory=lambda: mock_ex,
    )
    ex.close_at(symbol="BTC-USDT-SWAP", side="long", size=0.01,
                price=50_000.0, timestamp=1, atr=100.0, reason="stop_loss")
    params = mock_ex.create_market_order.call_args.kwargs["params"]
    assert params.get("reduceOnly") is True


# ============================================================
# fetch_exchange_positions — parses ccxt response into flat dict
# ============================================================

def test_fetch_exchange_positions_flattens_ccxt_response():
    """Adapter's from_native_symbol handles the reverse mapping now."""
    mock_ex = MagicMock()
    mock_ex.fetch_positions.return_value = [
        {"symbol": "BTC/USDT:USDT", "side": "long",
         "contracts": 0.01, "entryPrice": 50_000.0},
        {"symbol": "ETH/USDT:USDT", "side": "short",
         "contracts": 2.0, "entryPrice": 3_000.0},
        # Zero-contract entry should be filtered out
        {"symbol": "SOL/USDT:USDT", "side": "long",
         "contracts": 0.0, "entryPrice": 0.0},
    ]
    # Adapter's from_native_symbol maps ccxt form → internal form
    mock_ex.from_native_symbol.side_effect = lambda s: (
        s.replace("/", "-").replace(":USDT", "-SWAP")
    )
    ex = LiveExecutor(
        _exec_cfg(), _live_cfg(enabled=True),
        exchange_factory=lambda: mock_ex,
    )
    positions = ex.fetch_exchange_positions()
    assert set(positions.keys()) == {"BTC-USDT-SWAP", "ETH-USDT-SWAP"}
    assert positions["BTC-USDT-SWAP"] == {
        "side": "long", "size": 0.01, "entry_price": 50_000.0,
    }
    assert positions["ETH-USDT-SWAP"]["side"] == "short"


def test_fetch_exchange_positions_returns_empty_when_disabled():
    ex = LiveExecutor(_exec_cfg(), _live_cfg(enabled=False))
    assert ex.fetch_exchange_positions() == {}
