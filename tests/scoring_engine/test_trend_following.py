import pandas as pd
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams


def _params():
    return TFParams(
        ema_fast=20, ema_slow=60, ema_trend=200,
        adx_threshold=25, volume_ratio_threshold=1.2,
        confirm_ema_fast=20, confirm_adx_threshold=20,
        funding_weight=0.20, funding_threshold=0.0003,
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
    # Favorable funding for long (negative funding = shorts crowded)
    out = tf.score(_row(funding_rate=-0.0003), pd.DataFrame())
    assert out.long > 0.6
    assert out.short < 0.2


def test_short_score_high_in_downtrend():
    tf = TrendFollowing(_params())
    # Favorable funding for short (positive funding = longs crowded)
    out = tf.score(
        _row(ema20=90, ema60=100, ema200=110, di_plus=15, di_minus=25,
             ema20_1h_on_15m=90, funding_rate=0.0003),
        pd.DataFrame(),
    )
    assert out.short > 0.6
    assert out.long < 0.2


def test_low_score_when_adx_below_threshold():
    tf = TrendFollowing(_params())
    out = tf.score(_row(adx=15), pd.DataFrame())
    # ADX weak + neutral funding → both sides low
    assert out.long < 0.5
    assert out.short < 0.5


def test_components_present():
    tf = TrendFollowing(_params())
    out = tf.score(_row(), pd.DataFrame())
    for k in ("ema_stack", "adx", "volume", "confirm", "funding"):
        assert k in out.components


def test_funding_boosts_contrarian_side():
    """High positive funding (longs crowded) must boost short score,
    high negative funding (shorts crowded) must boost long score."""
    tf = TrendFollowing(_params())
    # Neutral EMA/ADX so funding dominates
    neutral_row = _row(
        ema20=100.0, ema60=100.0, ema200=100.0,  # no stack direction
        adx=15.0,  # trend_gate off
    )
    high_pos_funding = tf.score({**neutral_row, "funding_rate": 0.001}, pd.DataFrame())
    high_neg_funding = tf.score({**neutral_row, "funding_rate": -0.001}, pd.DataFrame())
    # Positive funding → short side boosted
    assert high_pos_funding.short > high_pos_funding.long
    # Negative funding → long side boosted
    assert high_neg_funding.long > high_neg_funding.short
    # Missing funding → symmetric contribution → neither side favored
    missing = tf.score(neutral_row, pd.DataFrame())
    assert abs(missing.long - missing.short) < 1e-9
