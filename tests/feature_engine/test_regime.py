import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.regime import compute_regime


def _mk(n=600, adx_val=30.0, atr_pct_val=0.02):
    ts_ms = [i * 3_600_000 for i in range(n)]
    return pd.DataFrame({
        "timestamp": ts_ms,
        "adx": [adx_val] * n,
        "atr_pct": [atr_pct_val] * n,
    })


def test_trending_when_adx_high():
    df = _mk()
    out = compute_regime(df)
    assert out["regime"].iloc[-1] == "trending"


def test_ranging_when_adx_low():
    df = _mk(adx_val=10.0)
    out = compute_regime(df)
    assert out["regime"].iloc[-1] == "ranging"


def test_high_vol_wins_over_trend():
    df = _mk(adx_val=30.0)
    # 尾巴放极大 atr_pct
    df.loc[df.index[-1], "atr_pct"] = 1.0
    out = compute_regime(df)
    assert out["regime"].iloc[-1] == "high_vol"


def test_session_and_dow():
    df = _mk(n=48)
    out = compute_regime(df)
    for s in out["session"]:
        assert s in {"asia", "europe", "us"}
    for d in out["day_of_week"]:
        assert 0 <= d <= 6
