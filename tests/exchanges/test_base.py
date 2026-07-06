"""Tests for the exchange abstraction interface + factory."""
from __future__ import annotations

import pytest

from rabbit_hunter.exchanges.base import ExchangeAdapter, get_exchange
from rabbit_hunter.exchanges.okx import OKXAdapter


def test_get_exchange_okx_returns_adapter():
    adapter = get_exchange("okx")
    assert isinstance(adapter, OKXAdapter)
    assert adapter.name == "okx"


def test_get_exchange_is_case_insensitive():
    assert isinstance(get_exchange("OKX"), OKXAdapter)


def test_get_exchange_unknown_raises_with_registered_list():
    with pytest.raises(ValueError) as ei:
        get_exchange("kraken")
    assert "kraken" in str(ei.value)
    assert "okx" in str(ei.value)


def test_get_exchange_passes_kwargs_through():
    adapter = get_exchange("okx", testnet=True)
    assert isinstance(adapter, OKXAdapter)
    assert adapter._testnet is True


def test_default_from_native_symbol_reverses_ccxt_form():
    class _MinimalAdapter(ExchangeAdapter):
        name = "test"
        def to_native_symbol(self, symbol: str) -> str:
            return symbol
        def fetch_ohlcv(self, s, i, a, b): raise NotImplementedError
        def fetch_orderbook_top(self, s): raise NotImplementedError
        def fetch_current_funding_rate(self, s): raise NotImplementedError
        def fetch_funding_rate_history(self, s, a, b): raise NotImplementedError
        def fetch_open_interest_history(self, s, a, b): raise NotImplementedError
    a = _MinimalAdapter()
    assert a.from_native_symbol("BTC/USDT:USDT") == "BTC-USDT-SWAP"


def test_default_from_native_symbol_pass_through_when_no_slash():
    class _MinimalAdapter(ExchangeAdapter):
        name = "test"
        def to_native_symbol(self, symbol: str) -> str:
            return symbol
        def fetch_ohlcv(self, s, i, a, b): raise NotImplementedError
        def fetch_orderbook_top(self, s): raise NotImplementedError
        def fetch_current_funding_rate(self, s): raise NotImplementedError
        def fetch_funding_rate_history(self, s, a, b): raise NotImplementedError
        def fetch_open_interest_history(self, s, a, b): raise NotImplementedError
    a = _MinimalAdapter()
    # Non-ccxt-shaped input returned unchanged
    assert a.from_native_symbol("BTC-USDT-SWAP") == "BTC-USDT-SWAP"


def test_unauthenticated_adapter_refuses_write_methods():
    adapter = get_exchange("okx")   # no credentials
    with pytest.raises(NotImplementedError):
        adapter.create_market_order("BTC-USDT-SWAP", "buy", 0.01)
    with pytest.raises(NotImplementedError):
        adapter.fetch_positions()
