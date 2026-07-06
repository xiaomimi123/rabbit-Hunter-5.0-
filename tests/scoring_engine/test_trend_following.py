import pandas as pd
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams


def _params():
    return TFParams(
        ema_fast=20, ema_slow=60, ema_trend=200,
        adx_threshold=25, volume_ratio_threshold=1.2,
        confirm_ema_fast=20, confirm_adx_threshold=20,
    )


def _row(**kw):
    base = {
        "close": 100.0,
        "ema20": 105.0, "ema60": 100.0, "ema200": 90.0,
        "adx": 30.0, "di_plus": 25.0, "di_minus": 15.0,
        "volume_ratio_20": 1.5,
        "ema20_1h_on_15m": 105.0, "adx_1h_on_15m": 22.0,
    }
    base.update(kw)
    return base


def test_long_score_high_in_uptrend():
    tf = TrendFollowing(_params())
    out = tf.score(_row(), pd.DataFrame())
    assert out.long > 0.6
    assert out.short < 0.2


def test_short_score_high_in_downtrend():
    tf = TrendFollowing(_params())
    out = tf.score(
        _row(ema20=90, ema60=100, ema200=110, di_plus=15, di_minus=25,
             ema20_1h_on_15m=90),
        pd.DataFrame(),
    )
    assert out.short > 0.6
    assert out.long < 0.2


def test_low_score_when_adx_below_threshold():
    tf = TrendFollowing(_params())
    out = tf.score(_row(adx=15), pd.DataFrame())
    # ADX 弱 → 两边分都低
    assert out.long < 0.5
    assert out.short < 0.5


def test_components_present():
    tf = TrendFollowing(_params())
    out = tf.score(_row(), pd.DataFrame())
    for k in ("ema_stack", "adx", "volume", "confirm"):
        assert k in out.components
