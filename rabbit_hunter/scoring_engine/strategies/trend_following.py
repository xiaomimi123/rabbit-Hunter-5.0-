from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from ..base import BaseStrategy, ScoreOutput


@dataclass(frozen=True)
class TFParams:
    ema_fast: int
    ema_slow: int
    ema_trend: int
    adx_threshold: float
    volume_ratio_threshold: float
    confirm_ema_fast: int
    confirm_adx_threshold: float
    # Funding-rate factor (added in v0.1.2). Contrarian: high positive funding
    # means longs are crowded → short edge; high negative funding means shorts
    # are crowded → long edge. `funding_threshold` sets the funding value that
    # maps to a full 1.0 contribution before clipping.
    funding_weight: float = 0.20
    funding_threshold: float = 0.0003  # 0.03% per 8-hour interval
    # Extreme-momentum gate (v0.1.3): if enabled, zero out the score unless the
    # bar shows extreme momentum in the trade direction. Cluster analysis of
    # 273 trades showed Cluster 4 (trend continuation without extreme momentum,
    # WR 0%) is a pure loser while Cluster 1 (deep-oversold / -overbought
    # momentum breakdown, WR 62% on crash days) carries most of the alpha.
    require_extreme_momentum: bool = False
    rsi_short_threshold: float = 30.0   # RSI at or below → short-side OK
    rsi_long_threshold: float = 70.0    # RSI at or above → long-side OK
    zscore_extreme_threshold: float = 1.5  # |z-score| at or above → OK either way


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class TrendFollowing(BaseStrategy):
    """Trend-following strategy: EMA stack + ADX/DI + volume + 15m confirm."""

    name = "trend_following"
    version = "0.1.3"

    # Weights rebalanced when funding factor was added (v0.1.2). Sum of
    # (W_EMA + W_ADX + W_VOL + W_CONF + funding_weight) must equal 1.0.
    W_EMA = 0.30
    W_ADX = 0.20
    W_VOL = 0.15
    W_CONF = 0.15

    def __init__(self, params: TFParams):
        self.params = params

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        p = self.params

        ema_f = features_row.get("ema20")
        ema_s = features_row.get("ema60")
        ema_t = features_row.get("ema200")
        adx = features_row.get("adx")
        di_plus = features_row.get("di_plus")
        di_minus = features_row.get("di_minus")
        vol_ratio = features_row.get("volume_ratio_20")
        confirm_ema = features_row.get("ema20_1h_on_15m")
        confirm_adx = features_row.get("adx_1h_on_15m")

        # --- 1. EMA stack direction ---
        if ema_f is None or ema_s is None or ema_t is None:
            long_stack = 0.0
            short_stack = 0.0
        elif ema_f > ema_s > ema_t:
            long_stack, short_stack = 1.0, 0.0
        elif ema_f < ema_s < ema_t:
            long_stack, short_stack = 0.0, 1.0
        else:
            long_stack, short_stack = 0.0, 0.0

        # --- 2. ADX strength (magnitude) + regime gate ---
        # adx_score: how far above threshold (used as the "adx" component magnitude).
        # trend_gate: binary regime filter - below threshold, the market is not
        # trending, so ADX/volume/confirm evidence should not count even if the
        # EMA stack happens to be aligned (this is what keeps both long and short
        # low when ADX is weak).
        if adx is None:
            adx_score = 0.0
            trend_gate = 0.0
        else:
            adx_score = _clip01((adx - p.adx_threshold) / max(p.adx_threshold, 1e-9))
            trend_gate = 1.0 if adx >= p.adx_threshold else 0.0

        # --- 3. DI direction split ---
        if di_plus is None or di_minus is None:
            di_long = 0.5
            di_short = 0.5
        else:
            total = di_plus + di_minus
            if total <= 0:
                di_long = 0.5
                di_short = 0.5
            else:
                di_long = di_plus / total
                di_short = di_minus / total

        # --- 4. Volume ---
        if vol_ratio is None:
            vol_score = 0.0
        else:
            vol_score = _clip01(
                (vol_ratio - p.volume_ratio_threshold) / max(p.volume_ratio_threshold, 1e-9)
            )

        # --- 5. 15m/1h confirm, gated by the direction of the EMA stack ---
        confirm_long = 0.0
        confirm_short = 0.0
        if ema_f is not None and confirm_ema is not None:
            if long_stack:
                confirm_long = 1.0 if ema_f >= confirm_ema else 0.0
            if short_stack:
                confirm_short = 1.0 if ema_f <= confirm_ema else 0.0
        if confirm_adx is not None and confirm_adx < p.confirm_adx_threshold:
            confirm_long *= 0.5
            confirm_short *= 0.5

        # --- 6. Funding-rate contrarian factor (v0.1.2) ---
        # Independent signal — NOT gated by trend_gate — because funding
        # positioning gives an edge even when trend strength is modest.
        # Positive funding → longs pay shorts → likely short-side edge.
        # Negative funding → shorts pay longs → likely long-side edge.
        funding_rate = features_row.get("funding_rate")
        if funding_rate is None or (isinstance(funding_rate, float) and math.isnan(funding_rate)):
            funding_long = 0.5
            funding_short = 0.5
        else:
            funding_norm = float(funding_rate) / max(p.funding_threshold, 1e-12)
            funding_long = _clip01(-funding_norm)
            funding_short = _clip01(funding_norm)

        long_score = (
            self.W_EMA * long_stack
            + self.W_ADX * adx_score * di_long * trend_gate
            + self.W_VOL * vol_score * trend_gate
            + self.W_CONF * confirm_long * trend_gate
            + p.funding_weight * funding_long
        )
        short_score = (
            self.W_EMA * short_stack
            + self.W_ADX * adx_score * di_short * trend_gate
            + self.W_VOL * vol_score * trend_gate
            + self.W_CONF * confirm_short * trend_gate
            + p.funding_weight * funding_short
        )

        # v0.1.3 extreme-momentum gate: kills Cluster-4 (trend continuation
        # without extreme momentum, WR 0%) while preserving Cluster-1
        # (deep-oversold / -overbought momentum breakdown).
        momentum_gate_long = 1.0
        momentum_gate_short = 1.0
        if p.require_extreme_momentum:
            rsi = features_row.get("rsi_14")
            zscore = features_row.get("zscore_20")
            rsi_val = float(rsi) if rsi is not None and not (isinstance(rsi, float) and math.isnan(rsi)) else 50.0
            z_val = float(zscore) if zscore is not None and not (isinstance(zscore, float) and math.isnan(zscore)) else 0.0
            extreme_up = (rsi_val >= p.rsi_long_threshold) or (z_val >= p.zscore_extreme_threshold)
            extreme_dn = (rsi_val <= p.rsi_short_threshold) or (z_val <= -p.zscore_extreme_threshold)
            momentum_gate_long = 1.0 if extreme_up else 0.0
            momentum_gate_short = 1.0 if extreme_dn else 0.0

        long_score *= momentum_gate_long
        short_score *= momentum_gate_short

        return ScoreOutput(
            long=_clip01(long_score),
            short=_clip01(short_score),
            components={
                "ema_stack": long_stack - short_stack,
                "adx": adx_score,
                "volume": vol_score,
                "confirm": confirm_long - confirm_short,
                "funding": funding_long - funding_short,
            },
            metadata={
                "adx_value": adx,
                "vol_ratio": vol_ratio,
                "trend_gate": trend_gate,
                "di_long": di_long,
                "di_short": di_short,
                "funding_rate": funding_rate,
                "momentum_gate_long": momentum_gate_long,
                "momentum_gate_short": momentum_gate_short,
            },
        )
