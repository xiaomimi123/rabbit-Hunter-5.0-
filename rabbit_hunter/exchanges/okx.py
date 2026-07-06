"""OKXAdapter — the ExchangeAdapter implementation for OKX perpetual.

Wraps the existing okx_fetcher functions so:
  - New code (LiveExecutor, future data pipeline) uses the adapter
  - Existing callers of okx_fetcher.* continue to work unchanged
  - Adding Binance means writing a peer file, not editing this one
"""
from __future__ import annotations

import os
import time
from typing import Any, Literal

import ccxt
import pandas as pd

from .base import ExchangeAdapter


_OKX_INTERVAL_MAP = {"1H": "1h", "1h": "1h", "15m": "15m", "5m": "5m", "1D": "1d"}
_LIMIT = 200
_SLEEP_MS = 200
# OKX Open Interest history public endpoint retains ~90 days.
_OI_MAX_LOOKBACK_MS = 90 * 24 * 3_600_000


class OKXAdapter(ExchangeAdapter):
    """OKX perpetual adapter. Authenticated mode is entered when the
    three api_* kwargs are all non-empty; otherwise the client stays
    read-only and every write method raises NotImplementedError."""

    name = "okx"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        testnet: bool = False,
        client: Any = None,
    ):
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._testnet = testnet
        # Tests can inject a client (typically a MagicMock) to bypass
        # the ccxt constructor entirely.
        self._client_override = client

    # ------------------------------------------------------------------
    # Client construction — lazy so read-only tests don't need ccxt
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        if self._client_override is not None:
            return self._client_override
        params: dict[str, Any] = {"enableRateLimit": True}
        if self._api_key:
            params["apiKey"] = self._api_key
            params["secret"] = self._api_secret
            params["password"] = self._api_passphrase
        client = ccxt.okx(params)
        if self._testnet:
            client.set_sandbox_mode(True)
        return client

    @property
    def authenticated(self) -> bool:
        return bool(self._api_key and self._api_secret and self._api_passphrase)

    # ------------------------------------------------------------------
    # Symbol translation
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
        tf = _OKX_INTERVAL_MAP[interval]
        all_rows: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = call_with_retry(
                lambda c=cursor: client.fetch_ohlcv(
                    native, timeframe=tf, since=c, limit=_LIMIT,
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
                             "funding_rate": r["fundingRate"]})
            last_ts = batch[-1]["timestamp"]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1
            time.sleep(_SLEEP_MS / 1000)
        df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
        if df.empty:
            df = pd.DataFrame({
                "timestamp": pd.Series(dtype="int64"),
                "funding_rate": pd.Series(dtype="float64"),
            })
        else:
            df = df.astype({"timestamp": "int64", "funding_rate": "float64"}, copy=False)
        df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp") \
                                          .reset_index(drop=True)
        return df

    def fetch_open_interest_history(
        self, symbol: str, start_ms: int, end_ms: int,
    ) -> pd.DataFrame:
        client = self._client()
        native = self.to_native_symbol(symbol)
        effective_start = max(start_ms, end_ms - _OI_MAX_LOOKBACK_MS)
        rows: list[dict] = []
        cursor = effective_start
        while cursor < end_ms:
            try:
                batch = client.fetch_open_interest_history(
                    native, timeframe="1h", since=cursor, limit=_LIMIT,
                )
            except ccxt.ExchangeError:
                break
            if not batch:
                break
            for r in batch:
                rows.append({"timestamp": r["timestamp"],
                             "oi": r["openInterestAmount"]})
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
                "OKXAdapter.create_market_order needs api credentials — "
                "construct with api_key/api_secret/api_passphrase."
            )
        client = self._client()
        return client.create_market_order(
            symbol=self.to_native_symbol(symbol),
            side=side, amount=amount,
            params=(params or {"tdMode": "cross"}),
        )

    def fetch_positions(self) -> list[dict]:
        if not self.authenticated:
            raise NotImplementedError(
                "OKXAdapter.fetch_positions needs api credentials"
            )
        return self._client().fetch_positions() or []
