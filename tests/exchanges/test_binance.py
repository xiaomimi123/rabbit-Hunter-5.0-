"""Tests for BinanceAdapter — mirror-image of OKXAdapter tests.

Adds enough coverage that a Binance-based deployment isn't a leap of
faith. Every method that touches the exchange goes through an injected
mock client — no network + no real credentials.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rabbit_hunter.exchanges.binance import BinanceAdapter
from rabbit_hunter.exchanges.base import get_exchange


# ============================================================
# Factory registration
# ============================================================

def test_factory_returns_binance_adapter():
    adapter = get_exchange("binance")
    assert isinstance(adapter, BinanceAdapter)
    assert adapter.name == "binance"


# ============================================================
# Symbol translation
# ============================================================

def test_to_native_symbol_matches_ccxt_form():
    a = BinanceAdapter()
    assert a.to_native_symbol("BTC-USDT-SWAP") == "BTC/USDT:USDT"
    assert a.to_native_symbol("SOL-USDT-SWAP") == "SOL/USDT:USDT"


def test_from_native_symbol_reverses_translation():
    a = BinanceAdapter()
    assert a.from_native_symbol("BTC/USDT:USDT") == "BTC-USDT-SWAP"


# ============================================================
# authenticated property
# ============================================================

def test_authenticated_false_by_default():
    assert BinanceAdapter().authenticated is False


def test_authenticated_true_with_key_and_secret():
    """Binance needs only two credentials (unlike OKX's three-part)."""
    a = BinanceAdapter(api_key="k", api_secret="s")
    assert a.authenticated is True


def test_authenticated_ignores_passphrase_arg():
    """Passphrase is accepted for signature parity with OKX but ignored."""
    a = BinanceAdapter(api_key="k", api_secret="s", api_passphrase="ignored")
    assert a.authenticated is True


# ============================================================
# Read-only market data — via injected client
# ============================================================

def test_fetch_orderbook_top_returns_bid_ask_mid_spread():
    client = MagicMock()
    client.fetch_order_book.return_value = {
        "bids": [[50_000.0, 1.0]],
        "asks": [[50_010.0, 1.0]],
    }
    a = BinanceAdapter(client=client)
    ob = a.fetch_orderbook_top("BTC-USDT-SWAP")
    assert ob == {"bid": 50_000.0, "ask": 50_010.0,
                  "mid": 50_005.0, "spread": 10.0}


def test_fetch_orderbook_top_returns_none_on_exception():
    client = MagicMock()
    client.fetch_order_book.side_effect = RuntimeError("net flake")
    a = BinanceAdapter(client=client)
    assert a.fetch_orderbook_top("BTC-USDT-SWAP") is None


def test_fetch_current_funding_rate_returns_rate_and_ts():
    client = MagicMock()
    client.fetch_funding_rate.return_value = {
        "fundingRate": 0.00025, "fundingTimestamp": 1_700_000_000_000,
    }
    a = BinanceAdapter(client=client)
    fr = a.fetch_current_funding_rate("BTC-USDT-SWAP")
    assert fr == {"rate": 0.00025, "next_funding_time_ms": 1_700_000_000_000}


def test_fetch_funding_rate_history_returns_typed_frame():
    client = MagicMock()
    client.fetch_funding_rate_history.return_value = [
        {"timestamp": 1000, "fundingRate": 0.0001},
        {"timestamp": 2000, "fundingRate": 0.0002},
        {"timestamp": 3000, "fundingRate": 0.0003},
    ]
    a = BinanceAdapter(client=client)
    df = a.fetch_funding_rate_history("BTC-USDT-SWAP", 0, 5000)
    assert len(df) == 3
    assert df["timestamp"].dtype == "int64"
    assert df["funding_rate"].dtype == "float64"


def test_fetch_funding_rate_history_empty_returns_typed_empty_frame():
    client = MagicMock()
    client.fetch_funding_rate_history.return_value = []
    a = BinanceAdapter(client=client)
    df = a.fetch_funding_rate_history("BTC-USDT-SWAP", 0, 5000)
    assert df.empty
    assert df["timestamp"].dtype == "int64"
    assert df["funding_rate"].dtype == "float64"


def test_fetch_ohlcv_pages_and_deduplicates():
    """OHLCV pagination — same batch semantics as OKX, uses Binance's
    larger 1500-row limit."""
    client = MagicMock()
    start_ms = 1_700_000_000_000
    client.fetch_ohlcv.side_effect = [
        [[start_ms, 100, 101, 99, 100.5, 10],
         [start_ms + 3_600_000, 101, 102, 100, 101.5, 12]],
        [[start_ms + 7_200_000, 102, 103, 101, 102.5, 15]],
        [],  # exhausted
    ]
    a = BinanceAdapter(client=client)
    df = a.fetch_ohlcv("BTC-USDT-SWAP", "1H",
                       start_ms, start_ms + 3 * 3_600_000)
    assert len(df) == 3
    assert list(df.columns) == ["timestamp", "open", "high",
                                 "low", "close", "volume"]
    assert df["timestamp"].is_monotonic_increasing


# ============================================================
# Authenticated / private
# ============================================================

def test_create_market_order_forwards_side_and_amount():
    client = MagicMock()
    client.create_market_order.return_value = {"filled": 0.01}
    a = BinanceAdapter(api_key="k", api_secret="s", client=client)
    result = a.create_market_order("BTC-USDT-SWAP", "buy", 0.01)
    assert result == {"filled": 0.01}
    kwargs = client.create_market_order.call_args.kwargs
    assert kwargs["symbol"] == "BTC/USDT:USDT"
    assert kwargs["side"] == "buy"
    assert kwargs["amount"] == 0.01


def test_create_market_order_refuses_when_unauthenticated():
    a = BinanceAdapter()
    with pytest.raises(NotImplementedError):
        a.create_market_order("BTC-USDT-SWAP", "buy", 0.01)


def test_fetch_positions_delegates_to_client():
    client = MagicMock()
    client.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT"}]
    a = BinanceAdapter(api_key="k", api_secret="s", client=client)
    assert a.fetch_positions() == [{"symbol": "BTC/USDT:USDT"}]


def test_fetch_positions_returns_empty_when_none():
    client = MagicMock()
    client.fetch_positions.return_value = None
    a = BinanceAdapter(api_key="k", api_secret="s", client=client)
    assert a.fetch_positions() == []


def test_fetch_positions_refuses_when_unauthenticated():
    a = BinanceAdapter()
    with pytest.raises(NotImplementedError):
        a.fetch_positions()


# ============================================================
# Backwards-compat shim
# ============================================================

def test_data_engine_binance_funding_delegates_to_adapter():
    """The legacy fetch_funding_rate_history_binance() must route through
    the same BinanceAdapter, so a bug in Binance-specific code has one
    fix location, not two."""
    from unittest.mock import patch
    from rabbit_hunter.data_engine import binance_funding

    client = MagicMock()
    client.fetch_funding_rate_history.return_value = [
        {"timestamp": 1000, "fundingRate": 0.0001},
    ]
    adapter = BinanceAdapter(client=client)
    with patch.object(binance_funding, "_adapter", return_value=adapter):
        df = binance_funding.fetch_funding_rate_history_binance(
            "BTC-USDT-SWAP", 0, 5000,
        )
    assert len(df) == 1
    assert df["funding_rate"].iloc[0] == 0.0001
