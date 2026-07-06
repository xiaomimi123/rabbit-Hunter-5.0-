"""Tests for OKXAdapter — every method that touches the exchange goes
through an injectable mock client, so no network + no real credentials."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rabbit_hunter.exchanges.okx import OKXAdapter


# ============================================================
# Symbol conversion
# ============================================================

def test_to_native_symbol_matches_ccxt_form():
    a = OKXAdapter()
    assert a.to_native_symbol("BTC-USDT-SWAP") == "BTC/USDT:USDT"
    assert a.to_native_symbol("ETH-USDT-SWAP") == "ETH/USDT:USDT"


def test_from_native_symbol_reverses_translation():
    a = OKXAdapter()
    assert a.from_native_symbol("BTC/USDT:USDT") == "BTC-USDT-SWAP"


# ============================================================
# authenticated property
# ============================================================

def test_authenticated_false_by_default():
    assert OKXAdapter().authenticated is False


def test_authenticated_true_when_all_three_credentials_set():
    a = OKXAdapter(api_key="k", api_secret="s", api_passphrase="p")
    assert a.authenticated is True


def test_authenticated_false_if_any_credential_missing():
    a = OKXAdapter(api_key="k", api_secret="s")   # missing passphrase
    assert a.authenticated is False


# ============================================================
# Read-only market data — via injected client
# ============================================================

def test_fetch_orderbook_top_returns_bid_ask_mid_spread():
    client = MagicMock()
    client.fetch_order_book.return_value = {
        "bids": [[50_000.0, 1.0]],
        "asks": [[50_010.0, 1.0]],
    }
    a = OKXAdapter(client=client)
    ob = a.fetch_orderbook_top("BTC-USDT-SWAP")
    assert ob == {"bid": 50_000.0, "ask": 50_010.0,
                  "mid": 50_005.0, "spread": 10.0}


def test_fetch_orderbook_top_returns_none_on_exception():
    client = MagicMock()
    client.fetch_order_book.side_effect = RuntimeError("net flake")
    a = OKXAdapter(client=client)
    assert a.fetch_orderbook_top("BTC-USDT-SWAP") is None


def test_fetch_orderbook_top_returns_none_on_empty_book():
    client = MagicMock()
    client.fetch_order_book.return_value = {"bids": [], "asks": []}
    a = OKXAdapter(client=client)
    assert a.fetch_orderbook_top("BTC-USDT-SWAP") is None


def test_fetch_current_funding_rate_returns_rate_and_ts():
    client = MagicMock()
    client.fetch_funding_rate.return_value = {
        "fundingRate": 0.00025, "fundingTimestamp": 1_700_000_000_000,
    }
    a = OKXAdapter(client=client)
    fr = a.fetch_current_funding_rate("BTC-USDT-SWAP")
    assert fr == {"rate": 0.00025, "next_funding_time_ms": 1_700_000_000_000}


def test_fetch_current_funding_rate_returns_none_on_exception():
    client = MagicMock()
    client.fetch_funding_rate.side_effect = RuntimeError("boom")
    a = OKXAdapter(client=client)
    assert a.fetch_current_funding_rate("BTC-USDT-SWAP") is None


def test_fetch_funding_rate_history_returns_typed_frame():
    client = MagicMock()
    client.fetch_funding_rate_history.return_value = [
        {"timestamp": 1000, "fundingRate": 0.0001},
        {"timestamp": 2000, "fundingRate": 0.0002},
    ]
    a = OKXAdapter(client=client)
    df = a.fetch_funding_rate_history("BTC-USDT-SWAP", 0, 5000)
    assert len(df) == 2
    assert df["timestamp"].dtype == "int64"
    assert df["funding_rate"].dtype == "float64"


def test_fetch_funding_rate_history_returns_typed_empty_frame():
    client = MagicMock()
    client.fetch_funding_rate_history.return_value = []
    a = OKXAdapter(client=client)
    df = a.fetch_funding_rate_history("BTC-USDT-SWAP", 0, 5000)
    assert df.empty
    assert df["timestamp"].dtype == "int64"
    assert df["funding_rate"].dtype == "float64"


# ============================================================
# Authenticated / private
# ============================================================

def test_create_market_order_forwards_side_and_amount():
    client = MagicMock()
    client.create_market_order.return_value = {"filled": 0.01}
    a = OKXAdapter(api_key="k", api_secret="s", api_passphrase="p",
                    client=client)
    result = a.create_market_order("BTC-USDT-SWAP", "buy", 0.01)
    assert result == {"filled": 0.01}
    client.create_market_order.assert_called_once()
    kwargs = client.create_market_order.call_args.kwargs
    assert kwargs["symbol"] == "BTC/USDT:USDT"
    assert kwargs["side"] == "buy"
    assert kwargs["amount"] == 0.01
    assert kwargs["params"] == {"tdMode": "cross"}


def test_create_market_order_forwards_custom_params():
    client = MagicMock()
    client.create_market_order.return_value = {}
    a = OKXAdapter(api_key="k", api_secret="s", api_passphrase="p",
                    client=client)
    a.create_market_order("BTC-USDT-SWAP", "sell", 0.01,
                          params={"tdMode": "cross", "reduceOnly": True})
    kwargs = client.create_market_order.call_args.kwargs
    assert kwargs["params"]["reduceOnly"] is True


def test_create_market_order_refuses_when_unauthenticated():
    a = OKXAdapter()
    with pytest.raises(NotImplementedError):
        a.create_market_order("BTC-USDT-SWAP", "buy", 0.01)


def test_fetch_positions_delegates_to_client():
    client = MagicMock()
    client.fetch_positions.return_value = [{"symbol": "BTC/USDT:USDT"}]
    a = OKXAdapter(api_key="k", api_secret="s", api_passphrase="p",
                    client=client)
    assert a.fetch_positions() == [{"symbol": "BTC/USDT:USDT"}]


def test_fetch_positions_returns_empty_when_none():
    client = MagicMock()
    client.fetch_positions.return_value = None
    a = OKXAdapter(api_key="k", api_secret="s", api_passphrase="p",
                    client=client)
    assert a.fetch_positions() == []
