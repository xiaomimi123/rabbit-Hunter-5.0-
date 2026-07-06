import pandas as pd
import pytest
from rabbit_hunter.scoring_engine.strategies.price_action import PriceAction, PAParams


def _params(**overrides):
    base = {
        "swing_proximity_atr": 2.0,
        "doji_proximity_atr": 1.0,
        "pin_bar_shadow_body_multiplier": 2.0,
        "pin_bar_shadow_range_fraction": 0.6,
        "w_confirmed_engulfing": 0.45,
        "w_swing_proximity": 0.25,
        "w_directional_pin_bar": 0.15,
        "w_doji_at_swing": 0.15,
    }
    base.update(overrides)
    return PAParams(**base)


def _row(**overrides):
    """A neutral current row — no pattern firing."""
    base = {
        "open": 100.0,
        "high": 100.5,
        "low": 99.5,
        "close": 100.0,
        "atr_14": 5.0,
        "swing_high_last": 110.0,
        "swing_low_last": 90.0,
        "structure_regime": "range",
        "pattern_pinbar": 0,
        "pattern_doji": 0,
    }
    base.update(overrides)
    return base


def _history(*rows):
    return pd.DataFrame(list(rows))


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        PriceAction(_params(w_confirmed_engulfing=0.5))  # sum = 1.05


def test_confirmed_engulfing_at_swing_low_in_uptrend_scores_max_engulfing_plus_prox():
    """Confirmed bull engulfing + at swing_low = 0.45 + 0.25 = 0.70
    (no pin bar, no doji)."""
    pa = PriceAction(_params())
    prev = {"high": 89.0, "low": 88.0, "close": 88.5,
            "pattern_engulfing_bull": 1, "pattern_engulfing_bear": 0}
    curr = _row(
        close=90.0, swing_low_last=90.0, structure_regime="uptrend",
    )
    out = pa.score(curr, _history(prev, curr))
    assert out.long == pytest.approx(0.70)
    assert out.short == pytest.approx(0.0)


def test_directional_pin_bar_bullish_scores_pin_weight():
    """Bull pin bar (long lower shadow) alone → 0.15 weight."""
    pa = PriceAction(_params())
    # o=100, c=100.2, h=100.4, l=99.0 → body=0.2, lower=1.0, upper=0.2, rng=1.4
    # lower/body = 5 > 2, lower/rng = 0.71 > 0.6 → bullish pin
    curr = _row(
        open=100.0, close=100.2, high=100.4, low=99.0,
        pattern_pinbar=1,
        structure_regime="uptrend",
        swing_low_last=50.0, swing_high_last=150.0,  # far from swings, no prox contribution
    )
    out = pa.score(curr, _history(curr))
    assert out.long == pytest.approx(0.15)
    assert out.short == pytest.approx(0.0)


def test_directional_pin_bar_bearish_scores_pin_weight():
    """Bear pin bar (long upper shadow) alone → 0.15 weight for short."""
    pa = PriceAction(_params())
    # o=100, c=99.8, h=101.0, l=99.6 → body=0.2, upper=1.0, lower=0.2, rng=1.4
    # upper/body = 5 > 2, upper/rng = 0.71 > 0.6 → bearish pin
    curr = _row(
        open=100.0, close=99.8, high=101.0, low=99.6,
        pattern_pinbar=1,
        structure_regime="downtrend",
        swing_low_last=50.0, swing_high_last=150.0,
    )
    out = pa.score(curr, _history(curr))
    assert out.short == pytest.approx(0.15)
    assert out.long == pytest.approx(0.0)


def test_pinbar_flag_but_shallow_shadows_no_direction():
    """pattern_pinbar=1 but shadows aren't long enough → no signal."""
    pa = PriceAction(_params())
    # o=100, c=100.4, h=100.5, l=99.9 → body=0.4, upper=0.1, lower=0.1
    # Neither shadow qualifies (< 2×body)
    curr = _row(
        open=100.0, close=100.4, high=100.5, low=99.9,
        pattern_pinbar=1,
        structure_regime="uptrend",
        swing_low_last=50.0, swing_high_last=150.0,
    )
    out = pa.score(curr, _history(curr))
    assert out.long == 0.0
    assert out.short == 0.0


def test_doji_at_swing_low_scores_long():
    """Doji within 1×ATR of swing_low → long boost 0.15."""
    pa = PriceAction(_params())
    # atr=5, so doji_max_distance = 1 * 5 = 5.0. close=91 is 1 above swing_low 90 → in range.
    curr = _row(
        close=91.0, atr_14=5.0, swing_low_last=90.0,
        pattern_doji=1,
        structure_regime="uptrend",
    )
    out = pa.score(curr, _history(curr))
    # Only doji fires (0.15) + swing_proximity: close=91, dist=1, max_dist=10 → prox=0.9 → 0.25*0.9=0.225
    # Total: 0.15 + 0.225 = 0.375
    assert out.long == pytest.approx(0.375)
    assert out.short == 0.0


def test_doji_far_from_swing_no_boost():
    """Doji but close is far from swing → doji factor 0, only proximity partial."""
    pa = PriceAction(_params())
    # atr=5, doji_max = 5. close=100 is 10 above swing_low 90 → OUT of doji range.
    curr = _row(
        close=100.0, atr_14=5.0, swing_low_last=90.0,
        pattern_doji=1,
        structure_regime="uptrend",
    )
    out = pa.score(curr, _history(curr))
    # swing_prox: dist=10, max=10, prox=0 → nothing scores
    # doji factor: dist=10 > doji_max=5 → doji_long=0
    assert out.long == 0.0


def test_all_four_signals_stack_to_max():
    """Confirmed engulfing + pin bar + doji at swing_low + swing prox all fire → 1.0."""
    pa = PriceAction(_params())
    prev = {"high": 89.0, "low": 88.0, "close": 88.5,
            "pattern_engulfing_bull": 1, "pattern_engulfing_bear": 0}
    # o=90, c=90.2, h=90.4, l=89.0 → body=0.2, lower=1.0 → bull pin
    # close=90.2 near swing_low=90 within 1×ATR (5) → doji at swing (if pattern_doji=1)
    # close > prev.high (89) → engulfing confirmed
    curr = _row(
        open=90.0, close=90.2, high=90.4, low=89.0,
        atr_14=5.0, swing_low_last=90.0,
        pattern_pinbar=1, pattern_doji=1,
        structure_regime="uptrend",
    )
    out = pa.score(curr, _history(prev, curr))
    # engulfing 0.45 + prox (dist=0.2, max=10, prox=0.98) 0.245 + pin 0.15 + doji 0.15 = ~0.995
    assert out.long > 0.95
    assert out.short == 0.0


def test_range_structure_zeros_all_scores():
    pa = PriceAction(_params())
    prev = {"high": 89.0, "low": 88.0, "close": 88.5,
            "pattern_engulfing_bull": 1, "pattern_engulfing_bear": 0}
    curr = _row(
        open=90.0, close=90.2, high=90.4, low=89.0,
        pattern_pinbar=1, pattern_doji=1,
        swing_low_last=90.0,
        structure_regime="range",
    )
    out = pa.score(curr, _history(prev, curr))
    assert out.long == 0.0
    assert out.short == 0.0


def test_missing_features_dont_crash():
    pa = PriceAction(_params())
    out = pa.score({"structure_regime": "uptrend"}, pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0


def test_components_present():
    pa = PriceAction(_params())
    curr = _row(structure_regime="uptrend")
    out = pa.score(curr, _history(curr))
    for k in ("confirmed_engulfing", "swing_proximity", "directional_pin_bar",
              "doji_at_swing", "structure_gate"):
        assert k in out.components
