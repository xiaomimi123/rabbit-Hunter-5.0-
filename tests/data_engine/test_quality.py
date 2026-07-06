import pandas as pd
import pytest
from rabbit_hunter.data_engine.quality import check_ohlcv


def _mk(ts_seq, close_val=100.0):
    return pd.DataFrame({
        "timestamp": ts_seq,
        "open": [close_val] * len(ts_seq),
        "high": [close_val + 1] * len(ts_seq),
        "low": [close_val - 1] * len(ts_seq),
        "close": [close_val] * len(ts_seq),
        "volume": [10.0] * len(ts_seq),
    })


def test_clean_bars_pass():
    df = _mk([0, 3_600_000, 7_200_000])
    r = check_ohlcv(df, "1H")
    assert r.is_ok
    assert len(r.clean_df) == 3
    assert r.issues == []


def test_gap_detected():
    df = _mk([0, 3_600_000, 10_800_000])  # 缺一根
    r = check_ohlcv(df, "1H")
    assert not r.is_ok
    assert any(i["type"] == "gap" for i in r.issues)


def test_bad_prices_dropped():
    df = _mk([0, 3_600_000], close_val=100.0)
    df.loc[1, "close"] = -1.0
    r = check_ohlcv(df, "1H")
    assert len(r.clean_df) == 1
    assert any(i["type"] == "invalid_price" for i in r.issues)


def test_duplicate_timestamps_dropped():
    df = _mk([0, 3_600_000, 3_600_000])
    r = check_ohlcv(df, "1H")
    assert len(r.clean_df) == 2
    assert any(i["type"] == "duplicate_timestamp" for i in r.issues)
