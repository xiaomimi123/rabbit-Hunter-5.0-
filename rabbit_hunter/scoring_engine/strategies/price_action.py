"""Price-action strategy plugin (v0.2.0).

v0.1 was a failure — high scores correlated with LOWER win rate (backtest
showed 0% winrate at scores > 0.7). Analysis: BOS factor in a directional
structure often marked "trend exhaustion" rather than "continuation";
unconfirmed engulfing patterns had high false-positive rate.

v0.2 changes:
  1. REMOVED: BOS factor (was the worst signal)
  2. ADDED:   next-bar confirmation requirement — an engulfing pattern
              only scores if the FOLLOWING bar closes past the pattern's
              extreme (breaks prev.high for bull, prev.low for bear).
              This filters out "false engulfings" that reverse next bar.
  3. Retained: structure regime gate, swing proximity factor.

Weights (must sum to 1.0):
  w_confirmed_engulfing: 0.65
  w_swing_proximity:     0.35
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..base import BaseStrategy, ScoreOutput


@dataclass(frozen=True)
class PAParams:
    swing_proximity_atr: float = 2.0
    allowed_structures: tuple[str, ...] = field(
        default_factory=lambda: ("uptrend", "downtrend")
    )
    w_confirmed_engulfing: float = 0.65
    w_swing_proximity: float = 0.35


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


class PriceAction(BaseStrategy):
    """Confirmed engulfing (prev bar pattern + current bar breakout) at swing S/R."""

    name = "price_action"
    version = "0.2.0"

    def __init__(self, params: PAParams):
        self.params = params
        w_sum = params.w_confirmed_engulfing + params.w_swing_proximity
        if not math.isclose(w_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"PriceAction weights must sum to 1.0, got {w_sum:.6f} "
                f"(w_confirmed_engulfing={params.w_confirmed_engulfing}, "
                f"w_swing_proximity={params.w_swing_proximity})"
            )

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        p = self.params

        structure = features_row.get("structure_regime")
        struct_gate = 1.0 if structure in p.allowed_structures else 0.0

        # --- Factor 1: Confirmed engulfing ---
        # Need at least 2 bars of history to look back one. features_history
        # is expected to include the current bar as the last row.
        confirmed_bull = 0.0
        confirmed_bear = 0.0
        if len(features_history) >= 2:
            prev = features_history.iloc[-2]
            curr_close = features_row.get("close")
            prev_bull_raw = prev.get("pattern_engulfing_bull", 0)
            prev_bear_raw = prev.get("pattern_engulfing_bear", 0)
            prev_high = prev.get("high")
            prev_low = prev.get("low")

            prev_bull = int(prev_bull_raw) if _finite(prev_bull_raw) else 0
            prev_bear = int(prev_bear_raw) if _finite(prev_bear_raw) else 0

            if prev_bull and _finite(curr_close) and _finite(prev_high):
                if float(curr_close) > float(prev_high):
                    confirmed_bull = 1.0
            if prev_bear and _finite(curr_close) and _finite(prev_low):
                if float(curr_close) < float(prev_low):
                    confirmed_bear = 1.0

        # --- Factor 2: Proximity to recent swing ---
        close = features_row.get("close")
        atr = features_row.get("atr_14")
        swing_low = features_row.get("swing_low_last")
        swing_high = features_row.get("swing_high_last")

        prox_long = 0.0
        prox_short = 0.0
        if _finite(atr) and float(atr) > 0 and _finite(close):
            atr_val = float(atr)
            close_val = float(close)
            max_distance = p.swing_proximity_atr * atr_val
            if _finite(swing_low):
                dist = close_val - float(swing_low)
                if dist >= 0:
                    prox_long = _clip01(1.0 - dist / max_distance)
            if _finite(swing_high):
                dist = float(swing_high) - close_val
                if dist >= 0:
                    prox_short = _clip01(1.0 - dist / max_distance)

        long_score = struct_gate * (
            p.w_confirmed_engulfing * confirmed_bull
            + p.w_swing_proximity * prox_long
        )
        short_score = struct_gate * (
            p.w_confirmed_engulfing * confirmed_bear
            + p.w_swing_proximity * prox_short
        )

        return ScoreOutput(
            long=_clip01(long_score),
            short=_clip01(short_score),
            components={
                "confirmed_engulfing": confirmed_bull - confirmed_bear,
                "swing_proximity": prox_long - prox_short,
                "structure_gate": struct_gate,
            },
            metadata={
                "structure_regime": structure,
                "swing_high_last": swing_high,
                "swing_low_last": swing_low,
                "confirmed_bull": confirmed_bull,
                "confirmed_bear": confirmed_bear,
            },
        )
