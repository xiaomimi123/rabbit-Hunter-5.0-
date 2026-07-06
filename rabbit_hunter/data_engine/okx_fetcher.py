"""Thin wrapper around OKXAdapter — kept for backwards compatibility.

All ccxt/OKX-specific code has moved to rabbit_hunter/exchanges/okx.py.
This module now just exposes the same function surface downstream code
has always used (fetch_ohlcv, fetch_orderbook_top, etc.) so existing
imports keep working while the OKX hardcode has been removed.

New code should call `get_exchange("okx")` directly and use the adapter
methods. This wrapper stays until every call site is migrated.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from rabbit_hunter.exchanges.base import get_exchange


def _adapter():
    return get_exchange("okx")


def fetch_orderbook_top(symbol: str) -> dict[str, float] | None:
    return _adapter().fetch_orderbook_top(symbol)


def fetch_current_funding_rate(symbol: str) -> dict[str, Any] | None:
    return _adapter().fetch_current_funding_rate(symbol)


def fetch_ohlcv(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    return _adapter().fetch_ohlcv(symbol, interval, start_ms, end_ms)


def fetch_funding_rate_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    return _adapter().fetch_funding_rate_history(symbol, start_ms, end_ms)


def fetch_open_interest_history(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    return _adapter().fetch_open_interest_history(symbol, start_ms, end_ms)
