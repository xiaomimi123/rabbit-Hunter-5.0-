"""Binance USD-M perpetual funding-rate history fetcher.

Binance retains multi-year funding history via public REST — much deeper
than OKX (~90 days). We keep OHLCV/OI on OKX (matches trading target),
but source funding from Binance so the strategy's funding-rate factor
actually has data across the full backtest window.

Rate scale: Binance's `fundingRate` is per 8-hour settlement, expressed
as decimal (e.g. 0.0001 = 0.01%). Same format as OKX.
"""
from __future__ import annotations
import time
from typing import Any
import ccxt
import pandas as pd

_LIMIT = 1000  # Binance funding history endpoint accepts up to 1000
_SLEEP_MS = 200


def _build_binance() -> Any:
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},  # USD-M perpetual
    })


def _to_binance_symbol(okx_symbol: str) -> str:
    """Map OKX-style `BTC-USDT-SWAP` → Binance USD-M ccxt symbol `BTC/USDT:USDT`."""
    base, quote, _swap = okx_symbol.split("-")
    return f"{base}/{quote}:{quote}"


def fetch_funding_rate_history_binance(
    okx_symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """Fetch funding history for an OKX-notated symbol from Binance.

    Timestamps and rate scale match OKX conventions so downstream code
    doesn't care which venue supplied the data.
    """
    ex = _build_binance()
    binance_symbol = _to_binance_symbol(okx_symbol)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_funding_rate_history(binance_symbol, since=cursor, limit=_LIMIT)
        if not batch:
            break
        for r in batch:
            rows.append({"timestamp": r["timestamp"], "funding_rate": float(r["fundingRate"])})
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    if rows:
        df = pd.DataFrame(rows).astype({"timestamp": "int64", "funding_rate": "float64"})
    else:
        df = pd.DataFrame({
            "timestamp": pd.Series(dtype="int64"),
            "funding_rate": pd.Series(dtype="float64"),
        })
    df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df
