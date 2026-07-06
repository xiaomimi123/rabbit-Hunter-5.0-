"""BinanceAdapter — ExchangeAdapter implementation for Binance USD-M perp.

Two primary use cases:
  1. Extended funding-rate history (Binance keeps multi-year, OKX ~90 days).
     This is what data_engine.binance_funding was doing standalone; now
     it's just adapter.fetch_funding_rate_history().
  2. Future primary trading venue — if we ever want to run live orders
     on Binance instead of OKX, the same LiveExecutor and reconciliation
     code works with zero changes; only the config's live_execution.exchange
     value changes.

Symbol convention matches OKXAdapter: internal form "BTC-USDT-SWAP",
native form "BTC/USDT:USDT". This makes the codebase's internal-form
symbols venue-agnostic — the router, feature engine, strategy code
never sees "which exchange".
"""
from __future__ import annotations

import time
from typing import Any, Literal

import ccxt
import pandas as pd

from .base import ExchangeAdapter


_BINANCE_INTERVAL_MAP = {"1H": "1h", "1h": "1h", "15m": "15m",
                         "5m": "5m", "1D": "1d"}
_LIMIT = 1000  # Binance funding endpoint accepts up to 1000
_OHLCV_LIMIT = 1500  # Binance klines endpoint cap
_SLEEP_MS = 200


class BinanceAdapter(ExchangeAdapter):
    """Binance USD-M perpetual adapter."""

    name = "binance"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        # Binance has no third credential (compared to OKX's passphrase);
        # accept and ignore for signature parity with OKXAdapter.
        api_passphrase: str = "",
        testnet: bool = False,
        client: Any = None,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        self._client_override = client
        # Silence "unused arg" — kept in the signature so get_exchange()
        # can pass the same kwargs regardless of exchange.
        _ = api_passphrase

    def _client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        params: dict[str, Any] = {
            "enableRateLimit": True,
            "options": {"defaultType": "future"},   # USD-M perpetual
        }
        if self._api_key:
            params["apiKey"] = self._api_key
            params["secret"] = self._api_secret
        client = ccxt.binance(params)
        if self._testnet:
            client.set_sandbox_mode(True)
        return client

    @property
    def authenticated(self) -> bool:
        # Binance requires only key + secret.
        return bool(self._api_key and self._api_secret)

    # ------------------------------------------------------------------
    # Symbol translation — matches OKXAdapter's shape on purpose
    # ------------------------------------------------------------------

    def to_native_symbol(self, symbol: str) -> str:
        base, quote, _tail = symbol.split("-")
        return f"{base}/{quote}:{quote}"

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self, symbol: str, interval: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        from rabbit_hunter.data_engine.retry import call_with_retry
        client = self._client()
        native = self.to_native_symbol(symbol)
        tf = _BINANCE_INTERVAL_MAP[interval]
        all_rows: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = call_with_retry(
                lambda c=cursor: client.fetch_ohlcv(
                    native, timeframe=tf, since=c, limit=_OHLCV_LIMIT,
                ),
                max_attempts=3,
            )
            if not batch:
                break
            all_rows.extend(batch)
            last_ts = batch[-1][0]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            time.sleep(_SLEEP_MS / 1000)
        df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high",
                                              "low", "close", "volume"])
        df = df[df["timestamp"] < end_ms].reset_index(drop=True)
        df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
        return df

    def fetch_orderbook_top(self, symbol: str) -> dict[str, float] | None:
        try:
            client = self._client()
            ob = client.fetch_order_book(self.to_native_symbol(symbol), limit=1)
            bid = float(ob["bids"][0][0]) if ob["bids"] else None
            ask = float(ob["asks"][0][0]) if ob["asks"] else None
            if bid is None or ask is None:
                return None
            return {"bid": bid, "ask": ask,
                    "mid": (bid + ask) / 2, "spread": ask - bid}
        except Exception:
            return None

    def fetch_current_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        try:
            fr = self._client().fetch_funding_rate(self.to_native_symbol(symbol))
            return {
                "rate": float(fr["fundingRate"]),
                "next_funding_time_ms": int(fr["fundingTimestamp"]),
            }
        except Exception:
            return None

    def fetch_funding_rate_history(
        self, symbol: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        client = self._client()
        native = self.to_native_symbol(symbol)
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = client.fetch_funding_rate_history(
                native, since=cursor, limit=_LIMIT,
            )
            if not batch:
                break
            for r in batch:
                rows.append({"timestamp": r["timestamp"],
                             "funding_rate": float(r["fundingRate"])})
            last_ts = batch[-1]["timestamp"]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            time.sleep(_SLEEP_MS / 1000)
        if rows:
            df = pd.DataFrame(rows).astype(
                {"timestamp": "int64", "funding_rate": "float64"}
            )
        else:
            df = pd.DataFrame({
                "timestamp": pd.Series(dtype="int64"),
                "funding_rate": pd.Series(dtype="float64"),
            })
        df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp") \
                                          .reset_index(drop=True)
        return df

    def fetch_open_interest_history(
        self, symbol: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        """Binance OI history — accepts long ranges but the payload
        shape differs slightly from OKX. Errors mid-fetch are swallowed
        rather than aborting: a partial series is preferable."""
        client = self._client()
        native = self.to_native_symbol(symbol)
        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            try:
                batch = client.fetch_open_interest_history(
                    native, timeframe="1h", since=cursor, limit=500,
                )
            except ccxt.ExchangeError:
                break
            if not batch:
                break
            for r in batch:
                rows.append({"timestamp": r["timestamp"],
                             "oi": float(r["openInterestAmount"] or 0.0)})
            last_ts = batch[-1]["timestamp"]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            time.sleep(_SLEEP_MS / 1000)
        df = pd.DataFrame(rows, columns=["timestamp", "oi"])
        if df.empty:
            df = pd.DataFrame({
                "timestamp": pd.Series(dtype="int64"),
                "oi": pd.Series(dtype="float64"),
            })
        else:
            df = df.astype({"timestamp": "int64", "oi": "float64"}, copy=False)
        df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp") \
                                          .reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Authenticated / private
    # ------------------------------------------------------------------

    def create_market_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        params: dict | None = None,
    ) -> dict:
        if not self.authenticated:
            raise NotImplementedError(
                "BinanceAdapter.create_market_order needs api credentials"
            )
        # Binance USD-M params default (hedge/one-way mode is account-level).
        return self._client().create_market_order(
            symbol=self.to_native_symbol(symbol),
            side=side, amount=amount, params=(params or {}),
        )

    def fetch_positions(self) -> list[dict]:
        if not self.authenticated:
            raise NotImplementedError(
                "BinanceAdapter.fetch_positions needs api credentials"
            )
        return self._client().fetch_positions() or []
