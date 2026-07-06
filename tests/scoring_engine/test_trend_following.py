import pandas as pd
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams


def _params(**overrides):
    kw = dict(
        ema_fast=20, ema_slow=60, ema_trend=200,
        adx_threshold=25, volume_ratio_threshold=1.2,
        confirm_ema_fast=20, confirm_adx_threshold=20,
        funding_weight=0.20, funding_threshold=0.0003,
    )
    kw.update(overrides)
    return TFParams(**kw)


def _row(**kw):
    base = {
        "close": 100.0,
        "ema20": 105.0, "ema60": 100.0, "ema200": 90.0,
        "adx": 30.0, "di_plus": 25.0, "di_minus": 15.0,
        "volume_ratio_20": 1.5,
        "ema20_1h_on_15m": 105.0, "adx_1h_on_15m": 22.0,
        # rsi_14 and zscore_20 default to neutral so the extreme-momentum
        # gate is a no-op unless a test explicitly sets them.
        "rsi_14": 50.0, "zscore_20": 0.0,
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


# ============================================================
# v0.1.3 · extreme-momentum gate
# ============================================================

def test_gate_off_by_default_matches_v012_behavior():
    """When require_extreme_momentum=False (default), v0.1.3 must produce
    the same scores as v0.1.2 — the gate is opt-in."""
    tf = TrendFollowing(_params())  # default: gate disabled
    out = tf.score(_row(funding_rate=-0.0003), pd.DataFrame())
    # Same assertion as test_long_score_high_in_uptrend
    assert out.long > 0.6
    assert out.short < 0.2
    # Metadata should still expose the gate multipliers, and they should
    # be 1.0 when the gate is off.
    assert out.metadata["momentum_gate_long"] == 1.0
    assert out.metadata["momentum_gate_short"] == 1.0


def test_gate_kills_long_without_extreme_momentum():
    """With gate ON, an aligned uptrend but neutral RSI (50) / neutral
    z-score (0) must produce long=0. This is exactly the Cluster-4
    'trend continuation without momentum' case that had WR 0%."""
    tf = TrendFollowing(_params(require_extreme_momentum=True))
    out = tf.score(_row(funding_rate=-0.0003, rsi_14=50.0, zscore_20=0.0),
                   pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0
    assert out.metadata["momentum_gate_long"] == 0.0


def test_gate_allows_long_with_extreme_rsi():
    """RSI ≥ 70 = overbought → gate passes for the long side."""
    tf = TrendFollowing(_params(require_extreme_momentum=True))
    out = tf.score(_row(funding_rate=-0.0003, rsi_14=72.0), pd.DataFrame())
    assert out.long > 0.6  # scores back to normal
    assert out.metadata["momentum_gate_long"] == 1.0


def test_gate_allows_short_with_extreme_rsi():
    """RSI ≤ 30 = deep oversold → gate passes for the short side (this
    is the Cluster-1 'momentum breakdown' case that had 62% WR on
    mass-crash days)."""
    tf = TrendFollowing(_params(require_extreme_momentum=True))
    out = tf.score(
        _row(ema20=90, ema60=100, ema200=110, di_plus=15, di_minus=25,
             ema20_1h_on_15m=90, funding_rate=0.0003, rsi_14=25.0),
        pd.DataFrame(),
    )
    assert out.short > 0.6
    assert out.metadata["momentum_gate_short"] == 1.0


def test_gate_allows_via_extreme_zscore():
    """|z-score| ≥ 1.5 alone (even with neutral RSI) is enough to pass."""
    tf = TrendFollowing(_params(require_extreme_momentum=True))
    # long side with big positive z
    out_long = tf.score(_row(funding_rate=-0.0003, rsi_14=50.0, zscore_20=1.8),
                        pd.DataFrame())
    assert out_long.long > 0.6

    # short side with big negative z
    out_short = tf.score(
        _row(ema20=90, ema60=100, ema200=110, di_plus=15, di_minus=25,
             ema20_1h_on_15m=90, funding_rate=0.0003,
             rsi_14=50.0, zscore_20=-1.8),
        pd.DataFrame(),
    )
    assert out_short.short > 0.6


def test_gate_handles_nan_rsi_and_zscore():
    """NaN inputs must default to neutral values (50 / 0), which means
    the gate blocks (no extreme momentum evidence). This prevents a
    silent bypass on the first bars of a series."""
    tf = TrendFollowing(_params(require_extreme_momentum=True))
    out = tf.score(_row(funding_rate=-0.0003,
                        rsi_14=float("nan"), zscore_20=float("nan")),
                   pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0
