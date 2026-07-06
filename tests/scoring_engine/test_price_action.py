import pandas as pd
import pytest
from rabbit_hunter.scoring_engine.strategies.price_action import PriceAction, PAParams


def _params(**overrides):
    base = {
        "swing_proximity_atr": 2.0,
        "w_confirmed_engulfing": 0.65,
        "w_swing_proximity": 0.35,
    }
    base.update(overrides)
    return PAParams(**base)


def _row(**overrides):
    """A neutral current row — no pattern firing, no signal."""
    base = {
        "close": 100.0,
        "atr_14": 5.0,
        "swing_high_last": 110.0,
        "swing_low_last": 90.0,
        "structure_regime": "range",
    }
    base.update(overrides)
    return base


def _history(*rows):
    """Build a features_history DataFrame from a sequence of row dicts.
    The LAST row is treated as 'current' by the strategy contract."""
    return pd.DataFrame(list(rows))


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        PriceAction(_params(w_confirmed_engulfing=0.5, w_swing_proximity=0.3))


def test_confirmed_bullish_engulfing_at_swing_low_in_uptrend_scores_max_long():
    """Prev bar had bull engulfing, current bar closes ABOVE prev.high →
    confirmed_bull=1. Also close is AT swing_low_last → prox_long=1.
    Structure = uptrend → gate=1. Total = 0.65 + 0.35 = 1.0."""
    pa = PriceAction(_params())
    prev = {"high": 95.0, "low": 88.0, "close": 94.0,
            "pattern_engulfing_bull": 1, "pattern_engulfing_bear": 0}
    curr = _row(
        close=90.0,
        swing_low_last=90.0,
        structure_regime="uptrend",
    )
    # curr.close (90) is NOT > prev.high (95), so this shouldn't fire.
    # Fix: make prev.high lower than curr.close.
    prev["high"] = 89.0  # curr close 90 > 89 → confirmed
    out = pa.score(curr, _history(prev, curr))
    assert out.long == pytest.approx(1.0)
    assert out.short == pytest.approx(0.0)


def test_confirmed_bearish_engulfing_at_swing_high_in_downtrend_scores_max_short():
    pa = PriceAction(_params())
    prev = {"high": 112.0, "low": 111.0, "close": 111.5,
            "pattern_engulfing_bull": 0, "pattern_engulfing_bear": 1}
    curr = _row(
        close=110.0,  # < prev.low 111 → confirms
        swing_high_last=110.0,
        structure_regime="downtrend",
    )
    out = pa.score(curr, _history(prev, curr))
    assert out.short == pytest.approx(1.0)
    assert out.long == pytest.approx(0.0)


def test_unconfirmed_engulfing_scores_low():
    """Prev had engulfing but current close did NOT break — no confirmation.
    Only proximity might contribute."""
    pa = PriceAction(_params())
    prev = {"high": 95.0, "low": 88.0, "close": 94.0,
            "pattern_engulfing_bull": 1, "pattern_engulfing_bear": 0}
    curr = _row(
        close=92.0,  # < prev.high 95 → NOT confirmed
        swing_low_last=90.0,  # close 92 near 90 → some proximity
        structure_regime="uptrend",
    )
    out = pa.score(curr, _history(prev, curr))
    # Only proximity contributes. dist=2, max_dist=10 → prox=0.8. Score = 0.35*0.8 = 0.28
    assert 0.20 < out.long < 0.35
    assert out.short == 0.0


def test_range_structure_zeros_all_scores():
    pa = PriceAction(_params())
    prev = {"high": 89.0, "low": 88.0, "close": 88.5,
            "pattern_engulfing_bull": 1, "pattern_engulfing_bear": 0}
    curr = _row(
        close=90.0,
        swing_low_last=90.0,
        structure_regime="range",
    )
    out = pa.score(curr, _history(prev, curr))
    assert out.long == 0.0
    assert out.short == 0.0


def test_no_prev_bar_zeros_confirmed_engulfing():
    """Not enough history for prev-bar lookup → confirmed factor is 0.
    Only proximity can still fire."""
    pa = PriceAction(_params())
    curr = _row(
        close=90.0, swing_low_last=90.0, structure_regime="uptrend",
    )
    out = pa.score(curr, _history(curr))  # only 1 bar of history
    # Only proximity: prox_long = 1.0 → 0.35 * 1.0 = 0.35
    assert out.long == pytest.approx(0.35)


def test_missing_features_dont_crash():
    pa = PriceAction(_params())
    out = pa.score({"structure_regime": "uptrend"}, pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0


def test_components_and_metadata_present():
    pa = PriceAction(_params())
    curr = _row(structure_regime="uptrend")
    out = pa.score(curr, _history(curr))
    for k in ("confirmed_engulfing", "swing_proximity", "structure_gate"):
        assert k in out.components
    for k in ("structure_regime", "swing_high_last", "swing_low_last",
              "confirmed_bull", "confirmed_bear"):
        assert k in out.metadata
