"""Mean-reversion strategy plugin (Phase 1b, v0.1.0).

Contrarian signal: when price is extremely stretched (RSI oversold/overbought,
BB pct near band edge, z-score >|2|), bet on reversion. Gated hard to
`regime ∈ {ranging, low_vol}` — mean reversion is the wrong bet in a
trending regime, so we simply refuse to score there and let other
strategies (e.g. trend_following) take those bars.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from ..base import BaseStrategy, ScoreOutput


@dataclass(frozen=True)
class MRParams:
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    bb_extreme_low: float = 0.15
    bb_extreme_high: float = 0.85
    zscore_threshold: float = 2.0
    # regime hard gate — mean reversion ONLY plays when the market is
    # in one of these regimes. Everything else scores 0.
    allowed_regimes: tuple[str, ...] = field(
        default_factory=lambda: ("ranging", "low_vol")
    )
    w_rsi: float = 0.35
    w_bb: float = 0.35
    w_zscore: float = 0.30


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class MeanReversion(BaseStrategy):
    """RSI + BB + Z-Score contrarian, gated to ranging/low_vol regimes."""

    name = "mean_reversion"
    version = "0.1.0"

    def __init__(self, params: MRParams):
        self.params = params
        # Sanity check: weights must sum to 1.0. Fail loud rather than
        # silently normalizing — this is a config error the user should fix.
        w_sum = params.w_rsi + params.w_bb + params.w_zscore
        if not math.isclose(w_sum, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"MeanReversion weights must sum to 1.0, got {w_sum:.6f} "
                f"(w_rsi={params.w_rsi}, w_bb={params.w_bb}, w_zscore={params.w_zscore})"
            )

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        p = self.params

        regime = features_row.get("regime")
        # regime gate — refuse to score outside allowed regimes
        regime_gate = 1.0 if regime in p.allowed_regimes else 0.0

        rsi = features_row.get("rsi_14")
        bb_pct = features_row.get("bb_pct")
        zscore = features_row.get("zscore_20")

        # --- RSI factor ---
        if rsi is None or (isinstance(rsi, float) and math.isnan(rsi)):
            rsi_long = 0.0
            rsi_short = 0.0
        else:
            # Long score: how far BELOW oversold threshold (linear ramp
            # from 0 at threshold to 1 at rsi=0). Symmetric for short.
            rsi_long = _clip01((p.rsi_oversold - rsi) / max(p.rsi_oversold, 1e-9))
            rsi_short = _clip01(
                (rsi - p.rsi_overbought) / max(100.0 - p.rsi_overbought, 1e-9)
            )

        # --- Bollinger Band position factor ---
        if bb_pct is None or (isinstance(bb_pct, float) and math.isnan(bb_pct)):
            bb_long = 0.0
            bb_short = 0.0
        else:
            # Long: how far bb_pct is below the extreme_low threshold.
            # Ramps from 0 at threshold to 1 at bb_pct <= 0.
            bb_long = _clip01((p.bb_extreme_low - bb_pct) / max(p.bb_extreme_low, 1e-9))
            bb_short = _clip01(
                (bb_pct - p.bb_extreme_high)
                / max(1.0 - p.bb_extreme_high, 1e-9)
            )

        # --- Z-Score factor ---
        if zscore is None or (isinstance(zscore, float) and math.isnan(zscore)):
            z_long = 0.0
            z_short = 0.0
        else:
            # Long: negative zscore stronger the more negative it is.
            # Ramps from 0 at -zscore_threshold to 1 at -2 × zscore_threshold.
            z_long = _clip01(
                (-zscore - p.zscore_threshold) / max(p.zscore_threshold, 1e-9)
            )
            z_short = _clip01(
                (zscore - p.zscore_threshold) / max(p.zscore_threshold, 1e-9)
            )

        # Combine — regime_gate multiplies the whole thing.
        long_score = regime_gate * (
            p.w_rsi * rsi_long + p.w_bb * bb_long + p.w_zscore * z_long
        )
        short_score = regime_gate * (
            p.w_rsi * rsi_short + p.w_bb * bb_short + p.w_zscore * z_short
        )

        return ScoreOutput(
            long=_clip01(long_score),
            short=_clip01(short_score),
            components={
                "rsi": rsi_long - rsi_short,
                "bb": bb_long - bb_short,
                "zscore": z_long - z_short,
                "regime_gate": regime_gate,
            },
            metadata={
                "rsi_value": rsi,
                "bb_pct_value": bb_pct,
                "zscore_value": zscore,
                "regime": regime,
            },
        )
