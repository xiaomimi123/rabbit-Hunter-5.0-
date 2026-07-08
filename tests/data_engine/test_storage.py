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


def test_write_drops_implausible_timestamps(tmp_path):
    """Zero-epoch and other bogus timestamps must never reach disk —
    otherwise data/health computes a 54-year "gap" that hides real
    issues. Reproduces the operator-reported BTC anomaly."""
    good_start = datetime(2025, 6, 1, tzinfo=timezone.utc)
    df = _mk_df(good_start, 24)
    # Inject a batch of zero-epoch rows (as the stub fetcher once did)
    # and one far-future row that would come from a bad clock.
    bad_zero = pd.DataFrame({
        "timestamp": [0, 3_600_000, 7_200_000],
        "open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
        "close": [1.0] * 3, "volume": [0.0] * 3,
    })
    bad_future = pd.DataFrame({
        "timestamp": [int(datetime(2099, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)],
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0.0],
    })
    poisoned = pd.concat([bad_zero, df, bad_future], ignore_index=True)
    paths = write_ohlcv(poisoned, tmp_path, "BTC-USDT-SWAP", "1H")

    # Only the good rows land on disk — no year=1970 or year=2099
    # partition should exist anywhere under the archive.
    for p in tmp_path.rglob("year=1970/*"): raise AssertionError(f"year=1970 leaked: {p}")
    for p in tmp_path.rglob("year=2099/*"): raise AssertionError(f"year=2099 leaked: {p}")
    start_ms = int(good_start.timestamp() * 1000)
    end_ms = start_ms + 25 * 3_600_000
    df_back = read_ohlcv(tmp_path, "BTC-USDT-SWAP", "1H", start_ms, end_ms)
    assert len(df_back) == 24  # only the plausible ones


def test_write_rejecting_all_rows_is_a_noop(tmp_path):
    """A frame containing ONLY bogus timestamps must return an empty
    path list, not raise — write_ohlcv is called on best-effort data
    from live shadow ticks and must never crash."""
    bad = pd.DataFrame({
        "timestamp": [0, 100, 200],
        "open": [1.0] * 3, "high": [1.0] * 3, "low": [1.0] * 3,
        "close": [1.0] * 3, "volume": [0.0] * 3,
    })
    paths = write_ohlcv(bad, tmp_path, "BTC-USDT-SWAP", "1H")
    assert paths == []
    # No files or partitions were written
    assert not list(tmp_path.rglob("*.parquet"))
