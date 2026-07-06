import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.indicators import compute_indicators


def _mk_trend_df(n: int = 300):
    close = np.linspace(100, 200, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close - 0.2
    vol = np.full(n, 100.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


def test_indicators_columns_present():
    df = _mk_trend_df()
    out = compute_indicators(df)
    for col in [
        "ema20", "ema60", "ema200", "ema20_slope",
        "adx", "di_plus", "di_minus",
        "rsi_14",
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct", "zscore_20",
        "atr_14", "atr_pct",
        "volume_ratio_20",
    ]:
        assert col in out.columns, f"missing {col}"


def test_ema_stack_in_uptrend():
    df = _mk_trend_df()
    out = compute_indicators(df).iloc[-1]
    # 稳定上涨中 EMA20 > EMA60 > EMA200
    assert out["ema20"] > out["ema60"] > out["ema200"]


def test_no_lookahead_last_row_stable():
    df = _mk_trend_df()
    full = compute_indicators(df).iloc[-1]
    partial = compute_indicators(df.iloc[:-1])
    partial_last_after_extend = compute_indicators(df).iloc[-2]
    # 去掉最后一行再算，再对同一位置的历史行取值，应与全量的对应行完全一致
    for col in ["ema20", "adx", "rsi_14", "atr_14"]:
        assert np.isclose(partial.iloc[-1][col], partial_last_after_extend[col], equal_nan=True), col
