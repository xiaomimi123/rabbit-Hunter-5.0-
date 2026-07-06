from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class HardRulesParams:
    min_quote_volume_24h: float
    atr_pct_max_multiplier: float
    atr_pct_baseline_window: int


def pass_hard_rules(features_row: dict, params: HardRulesParams) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    qv = features_row.get("quote_volume_24h")
    if qv is None or qv < params.min_quote_volume_24h:
        reasons.append("low_liquidity")

    atr_pct = features_row.get("atr_pct")
    baseline = features_row.get("atr_pct_baseline")
    if atr_pct is not None and baseline is not None and baseline > 0:
        if atr_pct > params.atr_pct_max_multiplier * baseline:
            reasons.append("extreme_volatility")

    return (len(reasons) == 0, reasons)
