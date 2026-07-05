from __future__ import annotations
import numpy as np
import pandas as pd


def _session_of(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 16:
        return "europe"
    return "us"


def compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    q_hi = out["atr_pct"].rolling(500, min_periods=50).quantile(0.9)
    q_lo = out["atr_pct"].rolling(500, min_periods=50).quantile(0.1)

    def label(row_atr, q_high, q_low, adx) -> str:
        # Guard against degenerate rolling windows (e.g. constant atr_pct)
        # where q_high == q_low == row_atr: in that case the bar sits at
        # both boundaries simultaneously and is not a genuine outlier, so
        # it must not be mistaken for high_vol/low_vol.
        if pd.notna(q_high) and pd.notna(q_low) and row_atr >= q_high and row_atr > q_low:
            return "high_vol"
        if pd.notna(q_high) and pd.notna(q_low) and row_atr <= q_low and row_atr < q_high:
            return "low_vol"
        if pd.notna(adx) and adx > 25:
            return "trending"
        return "ranging"

    regimes = [label(a, qh, ql, adx) for a, qh, ql, adx in zip(out["atr_pct"], q_hi, q_lo, out["adx"])]
    out["regime"] = regimes

    ts = pd.to_datetime(out["timestamp"], unit="ms", utc=True)
    out["session"] = [_session_of(t.hour) for t in ts]
    out["day_of_week"] = ts.dt.dayofweek.astype(int).to_numpy()
    return out
