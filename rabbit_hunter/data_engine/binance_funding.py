"""Thin wrapper over BinanceAdapter — kept for backwards compatibility.

The full implementation moved to rabbit_hunter/exchanges/binance.py.
This module now just exposes the one function downstream code has
imported for a while (`fetch_funding_rate_history_binance`) so old
callers keep working. New code should call get_exchange("binance")
directly.
"""
from __future__ import annotations

import pandas as pd

from rabbit_hunter.exchanges.base import get_exchange


def _adapter():
    return get_exchange("binance")


def fetch_funding_rate_history_binance(
    okx_symbol: str, start_ms: int, end_ms: int,
) -> pd.DataFrame:
    """Fetch funding history for an internal-form symbol from Binance.

    The parameter name says `okx_symbol` for historical reasons — it's
    really the codebase's internal symbol form (BTC-USDT-SWAP), which
    happens to match what OKX-facing code passes around. Rate scale
    and timestamps match OKX so downstream code stays venue-agnostic.
    """
    return _adapter().fetch_funding_rate_history(okx_symbol, start_ms, end_ms)
