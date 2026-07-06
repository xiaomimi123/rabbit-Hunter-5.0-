import pandas as pd
import pytest
from rabbit_hunter.scoring_engine.strategies.mean_reversion import MeanReversion, MRParams


def _params(**overrides):
    base = {
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0,
        "bb_extreme_low": 0.15,
        "bb_extreme_high": 0.85,
        "zscore_threshold": 2.0,
        "w_rsi": 0.35,
        "w_bb": 0.35,
        "w_zscore": 0.30,
    }
    base.update(overrides)
    return MRParams(**base)


def _row(**overrides):
    base = {
        "rsi_14": 50.0,
        "bb_pct": 0.5,
        "zscore_20": 0.0,
        "regime": "ranging",
    }
    base.update(overrides)
    return base


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        MeanReversion(_params(w_rsi=0.4, w_bb=0.4, w_zscore=0.3))


def test_long_score_high_on_extreme_oversold():
    mr = MeanReversion(_params())
    # RSI 15 (below 30 oversold, halfway between 30 and 0)
    # bb_pct 0.05 (below 0.15 threshold, ~66% of the way to 0)
    # zscore -3 (below -2 threshold, 50% past)
    out = mr.score(_row(rsi_14=15.0, bb_pct=0.05, zscore_20=-3.0), pd.DataFrame())
    assert out.long > 0.5
    assert out.short < 0.1


def test_short_score_high_on_extreme_overbought():
    mr = MeanReversion(_params())
    out = mr.score(_row(rsi_14=85.0, bb_pct=0.95, zscore_20=3.0), pd.DataFrame())
    assert out.short > 0.5
    assert out.long < 0.1


def test_zero_scores_when_regime_is_trending():
    """Regime gate must silence the strategy in `trending` regime — even
    if RSI/BB/Z would fire strongly."""
    mr = MeanReversion(_params())
    row = _row(rsi_14=15.0, bb_pct=0.05, zscore_20=-3.0, regime="trending")
    out = mr.score(row, pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0


def test_zero_scores_when_regime_is_high_vol():
    """`high_vol` is NOT in default allowed_regimes — mean reversion in a
    volatility explosion is a fast way to lose money."""
    mr = MeanReversion(_params())
    row = _row(rsi_14=15.0, bb_pct=0.05, zscore_20=-3.0, regime="high_vol")
    out = mr.score(row, pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0


def test_components_and_metadata_present():
    mr = MeanReversion(_params())
    out = mr.score(_row(), pd.DataFrame())
    for k in ("rsi", "bb", "zscore", "regime_gate"):
        assert k in out.components
    for k in ("rsi_value", "bb_pct_value", "zscore_value", "regime"):
        assert k in out.metadata


def test_missing_features_do_not_crash():
    """Any missing input → factor contributes 0 to both sides."""
    mr = MeanReversion(_params())
    out = mr.score({"regime": "ranging"}, pd.DataFrame())  # no RSI/BB/Z
    assert out.long == 0.0
    assert out.short == 0.0
