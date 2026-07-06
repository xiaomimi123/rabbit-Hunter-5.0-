"""ExchangeAdapter — the single seam every exchange-specific detail
must go through.

Everything the rest of the codebase needs from an exchange lives on this
class. Adding a new exchange = one new implementation file + one line in
the factory. It does NOT mean grepping the codebase for `ccxt.okx` calls.

Two categories of methods:
  1. Read-only market data (available even without API credentials):
     ohlcv, orderbook, funding, open interest.
  2. Authenticated actions (only usable when credentials are configured):
     create_market_order, fetch_positions.

An adapter that hasn't been authenticated raises NotImplementedError from
the write methods rather than silently no-op-ing — a caller that needs to
place orders must have configured the adapter for it.

Symbol convention:
  - The rabbit_hunter internal form is "BASE-QUOTE-SWAP" (e.g. BTC-USDT-SWAP).
  - Each adapter converts to and from its exchange-native form via
    `to_native_symbol(sym)` and `from_native_symbol(native)`.

This keeps the router, feature engine, ledger, and strategy code
exchange-agnostic — they only ever see the internal form.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import pandas as pd


class ExchangeAdapter(ABC):
    """Interface every exchange must implement.

    Names track what the rest of the codebase already calls today (e.g.
    `fetch_ohlcv`) so the migration to this abstraction is mechanical.
    """

    name: str = "base"

    # ------------------------------------------------------------------
    # Symbol translation
    # ------------------------------------------------------------------

    @abstractmethod
    def to_native_symbol(self, symbol: str) -> str:
        """Convert internal form (BTC-USDT-SWAP) to exchange-native form."""

    def from_native_symbol(self, native: str) -> str:
        """Inverse of to_native_symbol. Default assumes ccxt-style
        BASE/QUOTE:QUOTE — overridable when the exchange uses another form."""
        try:
            base_quote, tail = native.split(":")
            base, quote = base_quote.split("/")
            return f"{base}-{quote}-SWAP"
        except ValueError:
            return native

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_ohlcv(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        """Return columns: timestamp, open, high, low, close, volume."""

    @abstractmethod
    def fetch_orderbook_top(self, symbol: str) -> dict[str, float] | None:
        """Return {bid, ask, mid, spread} or None on error."""

    @abstractmethod
    def fetch_current_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        """Return {rate, next_funding_time_ms} or None on error."""

    @abstractmethod
    def fetch_funding_rate_history(
        self, symbol: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        """Return columns: timestamp, funding_rate."""

    @abstractmethod
    def fetch_open_interest_history(
        self, symbol: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        """Return columns: timestamp, oi."""

    # ------------------------------------------------------------------
    # Authenticated / private (only useful when credentials configured)
    # ------------------------------------------------------------------

    def create_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        params: dict | None = None,
    ) -> dict:
        raise NotImplementedError(
            f"{self.name}.create_market_order requires authenticated adapter"
        )

    def fetch_positions(self) -> list[dict]:
        raise NotImplementedError(
            f"{self.name}.fetch_positions requires authenticated adapter"
        )


# ============================================================
# Factory — the single lookup point that maps a config string to an
# ExchangeAdapter class. New exchanges register here.
# ============================================================

def get_exchange(name: str, **kwargs: Any) -> ExchangeAdapter:
    """Construct an adapter by name. Kwargs are passed through to the
    adapter constructor (typically credentials for authenticated mode).

    Raises ValueError if `name` is not registered — a hard failure at
    startup is preferred over silently returning None and crashing later
    with 'NoneType has no attribute fetch_ohlcv'."""
    from .okx import OKXAdapter
    from .binance import BinanceAdapter
    registry: dict[str, type[ExchangeAdapter]] = {
        "okx": OKXAdapter,
        "binance": BinanceAdapter,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(
            f"unknown exchange={name!r}. Registered: {sorted(registry)}. "
            "Add a new exchange by implementing ExchangeAdapter and "
            "registering it in exchanges/base.py::get_exchange."
        )
    return registry[key](**kwargs)
