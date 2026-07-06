"""Phase 3 · § 3.3 — Extreme-volatility circuit breaker.

Runs as the FIRST check on every bar, before any strategy scoring or
funding settlement. Reads the row's `atr_14` and `atr_pct_baseline`
(both already produced by Feature Engine) and trips when the ratio
exceeds the configured multiplier.

For a Phase 3 BACKTEST implementation, this lives in-process — same
loop as the strategies. The architecture spec's "独立熔断进程" concept
(a separate OS process that stays alive even if the main strategy loop
hangs) is a PRODUCTION concern that only matters when there's a real
strategy loop that can hang, i.e. Phase 4+. For backtest, the loop can't
"hang" so the in-process gate is sufficient to model the SAME behavior.

When tripped:
  1. All positions on the offending symbol are closed at the bar's close
     price (via BacktestExecutor.close_at with reason "circuit_breaker").
  2. New entries on that symbol are refused for this bar.
  3. A snapshot with action="circuit_breaker" is appended for reporting.

Recovery: the trip is per-bar. If the next bar's ratio is back below
threshold, normal flow resumes automatically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rabbit_hunter.config.schema import CircuitBreakerConfig


@dataclass(frozen=True)
class CircuitBreakerResult:
    tripped: bool
    reason: str | None
    atr_ratio: float


class CircuitBreaker:
    def __init__(self, cfg: CircuitBreakerConfig):
        self.cfg = cfg

    def check(self, features_row: dict) -> CircuitBreakerResult:
        if not self.cfg.enabled:
            return CircuitBreakerResult(False, None, 1.0)

        atr = features_row.get("atr_14")
        baseline = features_row.get("atr_pct_baseline")
        # Note: atr_pct_baseline is on the atr_pct scale (atr/price), so to
        # compare we need to normalize atr → atr_pct. Alternatively we can
        # use atr_pct directly if present.
        atr_pct = features_row.get("atr_pct")

        # Prefer atr_pct if available (already normalized); fall back to
        # computing on the fly if only atr_14 + close are present.
        if atr_pct is None or (isinstance(atr_pct, float) and math.isnan(atr_pct)):
            close = features_row.get("close")
            if atr is None or close is None or close == 0:
                return CircuitBreakerResult(False, None, 1.0)
            if isinstance(atr, float) and math.isnan(atr):
                return CircuitBreakerResult(False, None, 1.0)
            atr_pct_val = float(atr) / float(close)
        else:
            atr_pct_val = float(atr_pct)

        if baseline is None or (isinstance(baseline, float) and math.isnan(baseline)):
            # Baseline not yet warm (early bars) → no trip
            return CircuitBreakerResult(False, None, 1.0)

        baseline_val = float(baseline)
        if baseline_val <= 0:
            return CircuitBreakerResult(False, None, 1.0)

        ratio = atr_pct_val / baseline_val
        if ratio >= self.cfg.atr_shock_multiplier:
            return CircuitBreakerResult(
                tripped=True,
                reason=f"atr_ratio={ratio:.2f}>={self.cfg.atr_shock_multiplier:.1f}",
                atr_ratio=ratio,
            )
        return CircuitBreakerResult(False, None, ratio)
