from datetime import datetime, timezone
import pandas as pd
from rabbit_hunter.data_engine.storage import write_ohlcv, read_ohlcv


def _mk_df(start_dt: datetime, n: int, step_h: int = 1):
    ts = [int((start_dt.timestamp() + i * step_h * 3600) * 1000) for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts,
        "open": [1.0 + i for i in range(n)],
        "high": [1.5 + i for i in range(n)],
        "low": [0.5 + i for i in range(n)],
        "close": [1.2 + i for i in range(n)],
        "volume": [10.0 + i for i in range(n)],
    })


def test_write_and_read_roundtrip(tmp_path):
    df = _mk_df(datetime(2025, 1, 1, tzinfo=timezone.utc), 48)  # 2 天
    paths = write_ohlcv(df, tmp_path, "BTC-USDT-SWAP", "1H")
    assert len(paths) >= 1
    for p in paths:
        assert p.exists()
    start_ms = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp() * 1000)
    df_back = read_ohlcv(tmp_path, "BTC-USDT-SWAP", "1H", start_ms, end_ms)
    assert len(df_back) == 48
    assert df_back["timestamp"].is_monotonic_increasing


def test_write_crossing_month_creates_two_partitions(tmp_path):
    df = _mk_df(datetime(2025, 1, 31, 20, tzinfo=timezone.utc), 12)
    paths = write_ohlcv(df, tmp_path, "BTC-USDT-SWAP", "1H")
    partition_paths = {p.parent.name for p in paths}
    # 应该同时落在 month=01 和 month=02
    assert any("month=01" in p.as_posix() for p in paths)
    assert any("month=02" in p.as_posix() for p in paths)
