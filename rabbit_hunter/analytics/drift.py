"""Concept drift detector — compare live shadow performance against a
frozen backtest baseline, one cluster at a time.

Two failure modes it catches:
  1. A specific cluster's winrate collapsed in the live sample vs the
     baseline (regime change, competitor slippage, etc.).
  2. The mix of clusters shifted — e.g. the shadow suddenly produces
     mostly Cluster-4 trades that the baseline said are unprofitable.

Each cluster in the baseline is scored two ways:
  - Δ winrate — absolute pp change from baseline
  - Δ avg PnL — relative change; uses baseline sharpe/std to normalize
    into a z-score so "how many σ off is this" is directly comparable
    across clusters with very different variance.

A drift alert fires when EITHER measure exceeds its threshold AND the
live sample has ≥ min_n trades (guards against 1-trade false positives).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from .baseline import BaselineSnapshot, ClusterBaseline
from .cluster_performance import ClusterPerformanceReport, ClusterStats


@dataclass(frozen=True)
class DriftThresholds:
    # Fire if |Δ winrate| ≥ this fraction (pp).
    winrate_delta_alert: float = 0.15   # 15 percentage points
    # Fire if |Δ avg PnL| / baseline pnl_std × sqrt(N_live) ≥ this z.
    avg_pnl_zscore_alert: float = 2.5   # ~99% CI two-tailed
    # Minimum trades in the live sample before a cluster can trigger.
    min_live_trades: int = 20


@dataclass
class ClusterDriftFinding:
    cluster: str
    baseline_n: int
    live_n: int
    baseline_winrate: float
    live_winrate: float
    winrate_delta: float
    baseline_avg_pnl: float
    live_avg_pnl: float
    avg_pnl_zscore: float
    triggered: bool
    reason: str


@dataclass
class DriftReport:
    ok: bool
    findings: list[ClusterDriftFinding] = field(default_factory=list)
    # Clusters present in baseline but not in live (informational, not an alert)
    baseline_only: list[str] = field(default_factory=list)
    live_only: list[str] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        head = "drift: OK" if self.ok else "drift: ALERT"
        lines = [head]
        for f in self.findings:
            marker = "!!" if f.triggered else "  "
            lines.append(
                f"  {marker} {f.cluster}: baseline WR {f.baseline_winrate:.1%} "
                f"→ live WR {f.live_winrate:.1%} "
                f"(Δ{f.winrate_delta*100:+.1f}pp, z={f.avg_pnl_zscore:+.2f}, "
                f"n={f.live_n}) — {f.reason}"
            )
        for c in self.baseline_only:
            lines.append(f"  -- baseline had {c} but shadow has 0 trades")
        for c in self.live_only:
            lines.append(f"  ++ shadow has {c} but baseline had 0 trades "
                         "(unexpected cluster in live)")
        return lines


# ============================================================
# Core comparison
# ============================================================

def _score_cluster(
    baseline: ClusterBaseline,
    live: ClusterStats,
    thresholds: DriftThresholds,
) -> ClusterDriftFinding:
    live_wr = live.winrate
    wr_delta = live_wr - baseline.winrate

    # z-score of live avg PnL vs baseline distribution (mean=baseline avg,
    # std=baseline per-trade std). Standard error of the mean = std / sqrt(n).
    # So z = (live_avg - baseline_avg) * sqrt(n) / baseline_std.
    if baseline.pnl_std > 0 and live.n >= 1:
        z = ((live.avg_pnl - baseline.avg_pnl) *
             math.sqrt(live.n) / baseline.pnl_std)
    else:
        z = 0.0

    triggered = False
    reasons: list[str] = []
    if live.n < thresholds.min_live_trades:
        reasons.append(f"below min_n={thresholds.min_live_trades}")
    else:
        if abs(wr_delta) >= thresholds.winrate_delta_alert:
            triggered = True
            reasons.append(
                f"winrate|Δ|={abs(wr_delta)*100:.1f}pp≥"
                f"{thresholds.winrate_delta_alert*100:.0f}pp"
            )
        if abs(z) >= thresholds.avg_pnl_zscore_alert:
            triggered = True
            reasons.append(
                f"|z|={abs(z):.2f}≥{thresholds.avg_pnl_zscore_alert}"
            )
        if not triggered:
            reasons.append("within tolerance")

    return ClusterDriftFinding(
        cluster=live.cluster,
        baseline_n=baseline.n, live_n=live.n,
        baseline_winrate=baseline.winrate, live_winrate=live_wr,
        winrate_delta=wr_delta,
        baseline_avg_pnl=baseline.avg_pnl,
        live_avg_pnl=live.avg_pnl,
        avg_pnl_zscore=z,
        triggered=triggered,
        reason=";".join(reasons),
    )


def compare(
    baseline: BaselineSnapshot,
    live: ClusterPerformanceReport,
    thresholds: DriftThresholds | None = None,
) -> DriftReport:
    thresholds = thresholds or DriftThresholds()
    baseline_by = baseline.by_cluster()
    live_by = {s.cluster: s for s in live.stats}

    findings: list[ClusterDriftFinding] = []
    baseline_only: list[str] = []
    live_only: list[str] = []

    for cluster, bl in baseline_by.items():
        live_stats = live_by.get(cluster)
        if live_stats is None or live_stats.n == 0:
            baseline_only.append(cluster)
            continue
        findings.append(_score_cluster(bl, live_stats, thresholds))

    for cluster, live_stats in live_by.items():
        if live_stats.n == 0:
            continue
        if cluster not in baseline_by:
            live_only.append(cluster)

    any_triggered = any(f.triggered for f in findings) \
        or len(live_only) > 0
    return DriftReport(
        ok=not any_triggered,
        findings=findings,
        baseline_only=baseline_only,
        live_only=live_only,
    )
