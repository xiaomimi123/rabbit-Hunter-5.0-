"""Price-action strategy plugin (v0.3.0).

v0.2 achieved Sharpe 1.28 by removing BOS + adding next-bar confirmation.
v0.3 layers in two more secondary confirmation signals:

  - directional pin bar (long lower shadow = bullish rejection,
    long upper shadow = bearish rejection) — the Feature Engine's
    `pattern_pinbar` column is direction-agnostic, so we re-derive
    direction from OHLC in the strategy itself
  - doji at swing (doji within 1×ATR of swing_low → long, swing_high → short)

Weights (must sum to 1.0):
  w_confirmed_engulfing:   0.45  (down from 0.65)
  w_swing_proximity:       0.25  (down from 0.35)
  w_directional_pin_bar:   0.15  (new)
  w_doji_at_swing:         0.15  (new)

Primary signal is still the confirmed engulfing at S/R; pin bar and doji
serve as additional confirmation — they can lift a mid-conviction setup
past the open-action threshold without themselves being enough to trigger
an entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..base import BaseStrategy, ScoreOutput


@dataclass(frozen=True)
class PAParams:
    swing_proximity_atr: float = 2.0
    # doji at swing uses a stricter proximity than engulfing — doji is a
    # weaker signal so we require tighter alignment with S/R.
    doji_proximity_atr: float = 1.0
    # pin bar direction detection: shadow must be at least this multiple
    # of body AND this fraction of the bar's range.
    pin_bar_shadow_body_multiplier: float = 2.0
    pin_bar_shadow_range_fraction: float = 0.6
    allowed_structures: tuple[str, ...] = field(
        default_factory=lambda: ("uptrend", "downtrend")
    )
    w_confirmed_engulfing: float = 0.45
    w_swing_proximity: float = 0.25
    w_directional_pin_bar: float = 0.15
    w_doji_at_swing: float = 0.15


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _pin_bar_direction(
    o: float,
    h: float,
    l: float,
    c: float,
    body_mult: float,
    range_frac: float,
) -> int:
    """+1 = bullish pin (long lower shadow), -1 = bearish pin (long upper),
    0 = neither. Assumes pattern_pinbar has already flagged this bar; we
    just resolve WHICH side the shadow is on."""
    body = abs(c - o)
    rng = h - l
    if rng <= 0 or body <= 0:
        return 0
    upper = h - max(o, c)
    lower = min(o, c) - l
    if lower > body_mult * body and lower / rng > range_frac:
        return 1
    if upper > body_mult * body and upper / rng > range_frac:
        return -1
    return 0


class PriceAction(BaseStrategy):
    """Confirmed engulfing + swing proximity + directional pin bar + doji at swing."""

    name = "price_action"
    version = "0.3.0"

    def __init__(self, params: PAParams):
        self.params = params
        w_sum = (
            params.w_confirmed_engulfing
            + params.w_swing_proximity
            + params.w_directional_pin_bar
            + params.w_doji_at_swing
        )
        if not math.isclose(w_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"PriceAction weights must sum to 1.0, got {w_sum:.6f} "
                f"(w_confirmed_engulfing={params.w_confirmed_engulfing}, "
                f"w_swing_proximity={params.w_swing_proximity}, "
                f"w_directional_pin_bar={params.w_directional_pin_bar}, "
                f"w_doji_at_swing={params.w_doji_at_swing})"
            )

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        p = self.params

        structure = features_row.get("structure_regime")
        struct_gate = 1.0 if structure in p.allowed_structures else 0.0

        close = features_row.get("close")
        atr = features_row.get("atr_14")
        swing_low = features_row.get("swing_low_last")
        swing_high = features_row.get("swing_high_last")

        # --- Factor 1: Confirmed engulfing (from v0.2, unchanged) ---
        confirmed_bull = 0.0
        confirmed_bear = 0.0
        if len(features_history) >= 2:
            prev = features_history.iloc[-2]
            prev_bull_raw = prev.get("pattern_engulfing_bull", 0)
            prev_bear_raw = prev.get("pattern_engulfing_bear", 0)
            prev_high = prev.get("high")
            prev_low = prev.get("low")

            prev_bull = int(prev_bull_raw) if _finite(prev_bull_raw) else 0
            prev_bear = int(prev_bear_raw) if _finite(prev_bear_raw) else 0

            if prev_bull and _finite(close) and _finite(prev_high):
                if float(close) > float(prev_high):
                    confirmed_bull = 1.0
            if prev_bear and _finite(close) and _finite(prev_low):
                if float(close) < float(prev_low):
                    confirmed_bear = 1.0

        # --- Factor 2: Proximity to recent swing (from v0.2) ---
        prox_long = 0.0
        prox_short = 0.0
        if _finite(atr) and float(atr) > 0 and _finite(close):
            atr_val = float(atr)
            close_val = float(close)
            engulf_max_distance = p.swing_proximity_atr * atr_val
            if _finite(swing_low):
                dist = close_val - float(swing_low)
                if dist >= 0:
                    prox_long = _clip01(1.0 - dist / engulf_max_distance)
            if _finite(swing_high):
                dist = float(swing_high) - close_val
                if dist >= 0:
                    prox_short = _clip01(1.0 - dist / engulf_max_distance)

        # --- Factor 3: Directional pin bar (NEW in v0.3) ---
        pin_long = 0.0
        pin_short = 0.0
        pin_raw = features_row.get("pattern_pinbar", 0)
        pin = int(pin_raw) if _finite(pin_raw) else 0
        if pin:
            o = features_row.get("open")
            h = features_row.get("high")
            l = features_row.get("low")
            if _finite(o) and _finite(h) and _finite(l) and _finite(close):
                direction = _pin_bar_direction(
                    float(o), float(h), float(l), float(close),
                    p.pin_bar_shadow_body_multiplier,
                    p.pin_bar_shadow_range_fraction,
                )
                if direction > 0:
                    pin_long = 1.0
                elif direction < 0:
                    pin_short = 1.0

        # --- Factor 4: Doji at swing level (NEW in v0.3) ---
        doji_long = 0.0
        doji_short = 0.0
        doji_raw = features_row.get("pattern_doji", 0)
        doji = int(doji_raw) if _finite(doji_raw) else 0
        if doji and _finite(atr) and float(atr) > 0 and _finite(close):
            doji_max_distance = p.doji_proximity_atr * float(atr)
            close_val = float(close)
            if _finite(swing_low):
                dist = close_val - float(swing_low)
                if 0 <= dist <= doji_max_distance:
                    doji_long = 1.0
            if _finite(swing_high):
                dist = float(swing_high) - close_val
                if 0 <= dist <= doji_max_distance:
                    doji_short = 1.0

        long_score = struct_gate * (
            p.w_confirmed_engulfing * confirmed_bull
            + p.w_swing_proximity * prox_long
            + p.w_directional_pin_bar * pin_long
            + p.w_doji_at_swing * doji_long
        )
        short_score = struct_gate * (
            p.w_confirmed_engulfing * confirmed_bear
            + p.w_swing_proximity * prox_short
            + p.w_directional_pin_bar * pin_short
            + p.w_doji_at_swing * doji_short
        )

        return ScoreOutput(
            long=_clip01(long_score),
            short=_clip01(short_score),
            components={
                "confirmed_engulfing": confirmed_bull - confirmed_bear,
                "swing_proximity": prox_long - prox_short,
                "directional_pin_bar": pin_long - pin_short,
                "doji_at_swing": doji_long - doji_short,
                "structure_gate": struct_gate,
            },
            metadata={
                "structure_regime": structure,
                "swing_high_last": swing_high,
                "swing_low_last": swing_low,
                "confirmed_bull": confirmed_bull,
                "confirmed_bear": confirmed_bear,
                "pin_direction": pin_long - pin_short,
                "doji_direction": doji_long - doji_short,
            },
        )
