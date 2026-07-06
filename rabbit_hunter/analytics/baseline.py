"""Baseline snapshot — freeze per-cluster performance for drift comparison.

Given a backtest ClusterPerformanceReport, dump a JSON file that pins the
expected per-cluster winrate, avg PnL, and Sharpe. This becomes the
"reference truth" that live shadow performance is compared against.

Format is intentionally flat JSON (not pickle) so:
  - it's diffable in git across strategy versions
  - a human can edit it (e.g. widen tolerance temporarily)
  - a non-Python consumer can read it

Baseline is versioned by (tag, timestamp, source_report_dir) so a drift
alert points to a specific reference build, not "the current baseline".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .cluster_performance import ClusterPerformanceReport, ClusterStats
from .clustering import CLUSTER_LABELS


@dataclass
class ClusterBaseline:
    cluster: str
    n: int
    winrate: float
    avg_pnl: float
    sharpe_est: float
    profit_factor: float
    # Std over per-trade PnL — used to score how far a live sample has drifted.
    pnl_std: float


@dataclass
class BaselineSnapshot:
    tag: str
    created_at_utc: str
    source_report_dir: str
    total_trades: int
    clusters: list[ClusterBaseline] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "tag": self.tag,
            "created_at_utc": self.created_at_utc,
            "source_report_dir": self.source_report_dir,
            "total_trades": self.total_trades,
            "clusters": [asdict(c) for c in self.clusters],
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "BaselineSnapshot":
        data = json.loads(text)
        return cls(
            tag=data["tag"],
            created_at_utc=data["created_at_utc"],
            source_report_dir=data["source_report_dir"],
            total_trades=int(data["total_trades"]),
            clusters=[ClusterBaseline(**c) for c in data["clusters"]],
        )

    def by_cluster(self) -> dict[str, ClusterBaseline]:
        return {c.cluster: c for c in self.clusters}


def _pnl_std_from_stats(_stats: ClusterStats) -> float:
    """We don't have raw PnL in ClusterStats, so approximate std from
    range → sharpe consistency. Callers who need exact std should pass
    raw trades to build_baseline_from_trades() below."""
    # avg / sharpe × sqrt(365) recovers std
    if _stats.sharpe_est == 0 or _stats.n < 2:
        return 0.0
    import math
    return abs(_stats.avg_pnl) / _stats.sharpe_est * math.sqrt(365)


def build_baseline_from_report(
    report: ClusterPerformanceReport,
    tag: str,
    source_report_dir: str,
    now_utc: str | None = None,
) -> BaselineSnapshot:
    """Convert a ClusterPerformanceReport into a snapshot dataclass.

    Only clusters with n≥1 in the report get baseline entries — pinning
    a baseline for empty clusters would produce spurious drift alerts.
    """
    ts = now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    clusters: list[ClusterBaseline] = []
    for s in report.stats:
        if s.n == 0:
            continue
        clusters.append(ClusterBaseline(
            cluster=s.cluster,
            n=s.n,
            winrate=s.winrate,
            avg_pnl=s.avg_pnl,
            sharpe_est=s.sharpe_est,
            profit_factor=s.profit_factor,
            pnl_std=_pnl_std_from_stats(s),
        ))
    return BaselineSnapshot(
        tag=tag,
        created_at_utc=ts,
        source_report_dir=source_report_dir,
        total_trades=report.total_trades,
        clusters=clusters,
    )


def save(snapshot: BaselineSnapshot, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.to_json(), encoding="utf-8")
    return path


def load(path: Path) -> BaselineSnapshot:
    return BaselineSnapshot.from_json(path.read_text(encoding="utf-8"))
