from unittest.mock import MagicMock, patch
import pandas as pd
from rabbit_hunter.data_engine.okx_fetcher import fetch_ohlcv


def _fake_ohlcv_batch(base_ms: int, n: int):
    return [
        [base_ms + i * 3_600_000, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10.0 + i]
        for i in range(n)
    ]


def test_fetch_ohlcv_pages_and_stops_at_end():
    start_ms = 1_700_000_000_000
    end_ms = start_ms + 3 * 3_600_000  # 3 小时窗口
    mock_ex = MagicMock()
    # 第一次返回 2 根，第二次返回 1 根（末尾），第三次为空
    mock_ex.fetch_ohlcv.side_effect = [
        _fake_ohlcv_batch(start_ms, 2),
        _fake_ohlcv_batch(start_ms + 2 * 3_600_000, 1),
        [],
    ]
    with patch("rabbit_hunter.data_engine.okx_fetcher._build_exchange", return_value=mock_ex):
        df = fetch_ohlcv("BTC-USDT-SWAP", "1H", start_ms, end_ms)
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].iloc[0] == start_ms
