from __future__ import annotations
import time
from typing import Any
import ccxt
import pandas as pd

_OKX_INTERVAL_MAP = {"1H": "1h", "15m": "15m", "1h": "1h", "5m": "5m", "1D": "1d"}
_LIMIT = 200
_SLEEP_MS = 200


def _build_exchange() -> Any:
    return ccxt.okx({"enableRateLimit": True})


def _to_ccxt_symbol(symbol: str) -> str:
    # OKX 永续："BTC-USDT-SWAP" -> ccxt: "BTC/USDT:USDT"
    base, quote, tail = symbol.split("-")
    return f"{base}/{quote}:{quote}"


def fetch_ohlcv(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    ex = _build_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol)
    tf = _OKX_INTERVAL_MAP[interval]
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe=tf, since=cursor, limit=_LIMIT)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] < end_ms].reset_index(drop=True)
    df = df.drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def fetch_funding_rate_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    ex = _build_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_funding_rate_history(ccxt_symbol, since=cursor, limit=_LIMIT)
        if not batch:
            break
        for r in batch:
            rows.append({"timestamp": r["timestamp"], "funding_rate": r["fundingRate"]})
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df


def fetch_open_interest_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """OKX Open Interest 历史。ccxt: fetch_open_interest_history。"""
    ex = _build_exchange()
    ccxt_symbol = _to_ccxt_symbol(symbol)
    rows: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = ex.fetch_open_interest_history(ccxt_symbol, timeframe="1h", since=cursor, limit=_LIMIT)
        if not batch:
            break
        for r in batch:
            rows.append({"timestamp": r["timestamp"], "oi": r["openInterestAmount"]})
        last_ts = batch[-1]["timestamp"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(_SLEEP_MS / 1000)
    df = pd.DataFrame(rows, columns=["timestamp", "oi"])
    df = df[df["timestamp"] < end_ms].drop_duplicates(subset="timestamp").reset_index(drop=True)
    return df
