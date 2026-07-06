"""Tests for okx_fetcher — the backwards-compatibility shim over
OKXAdapter. The shim is one-line delegations, so the meaningful test
is that the batch/pagination behavior of OKXAdapter is unchanged.
"""
from unittest.mock import MagicMock, patch

import pandas as pd

from rabbit_hunter.data_engine.okx_fetcher import fetch_ohlcv


def _fake_ohlcv_batch(base_ms: int, n: int):
    return [
        [base_ms + i * 3_600_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
        for i in range(n)
    ]


def test_fetch_ohlcv_pages_and_stops_at_end():
    """OKXAdapter pages through fetch_ohlcv batches until an empty
    response, then dedupes and returns rows strictly before end_ms."""
    start_ms = 1_700_000_000_000
    end_ms = start_ms + 3 * 3_600_000  # 3-hour window
    mock_client = MagicMock()
    mock_client.fetch_ohlcv.side_effect = [
        _fake_ohlcv_batch(start_ms, 2),
        _fake_ohlcv_batch(start_ms + 2 * 3_600_000, 1),
        [],
    ]

    # Patch get_exchange to return an OKXAdapter with our mock client injected.
    from rabbit_hunter.exchanges.okx import OKXAdapter
    adapter = OKXAdapter(client=mock_client)
    with patch("rabbit_hunter.data_engine.okx_fetcher._adapter",
               return_value=adapter):
        df = fetch_ohlcv("BTC-USDT-SWAP", "1H", start_ms, end_ms)

    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] == start_ms
