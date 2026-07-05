from __future__ import annotations

import numpy as np
import pandas as pd

# NOTE: The upstream `pandas-ta` package (twopirllc/pandas-ta) no longer
# publishes releases compatible with Python 3.11 on PyPI (only 0.4.67b0/
# 0.4.71b0 remain, both requiring Python >=3.12); the historical
# 0.3.14b0 beta referenced by the spec has been removed from the index
# entirely. `pandas-ta-classic` is the actively maintained, API-compatible
# continuation of the same project (same function names and output column
# names, e.g. ADX_14/DMP_14/DMN_14, BBL_20_2.0/BBM_20_2.0/BBU_20_2.0) and
# supports our installed numpy 2.4 / pandas 3.0 stack, so it is used here
# as a drop-in replacement for `ta`.
import pandas_ta_classic as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # EMA
    out["ema20"] = ta.ema(out["close"], length=20)
    out["ema60"] = ta.ema(out["close"], length=60)
    out["ema200"] = ta.ema(out["close"], length=200)
    out["ema20_slope"] = out["ema20"].diff()

    # ADX / DI
    adx_df = ta.adx(out["high"], out["low"], out["close"], length=14)
    if adx_df is not None:
        out["adx"] = adx_df["ADX_14"]
        out["di_plus"] = adx_df["DMP_14"]
        out["di_minus"] = adx_df["DMN_14"]
    else:
        out["adx"] = np.nan
        out["di_plus"] = np.nan
        out["di_minus"] = np.nan

    # RSI
    out["rsi_14"] = ta.rsi(out["close"], length=14)

    # Bollinger Bands
    bb = ta.bbands(out["close"], length=20, std=2.0)
    if bb is not None:
        out["bb_lower"] = bb["BBL_20_2.0"]
        out["bb_middle"] = bb["BBM_20_2.0"]
        out["bb_upper"] = bb["BBU_20_2.0"]
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]
        out["bb_pct"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    else:
        for c in ["bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_pct"]:
            out[c] = np.nan

    # Z-Score(20)
    rolling_mean = out["close"].rolling(20).mean()
    rolling_std = out["close"].rolling(20).std()
    out["zscore_20"] = (out["close"] - rolling_mean) / rolling_std

    # ATR
    out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)
    out["atr_pct"] = out["atr_14"] / out["close"]

    # Volume ratio
    out["volume_ratio_20"] = out["volume"] / out["volume"].rolling(20).mean()

    return out
