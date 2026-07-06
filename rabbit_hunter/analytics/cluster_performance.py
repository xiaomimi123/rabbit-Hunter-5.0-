"""Per-cluster performance analyzer.

Given a set of closed trades (backtest trades.parquet OR shadow
ledger.closed_trades), classify each and aggregate performance per
cluster. Emits both a structured object (for programmatic consumers like
the dashboard) and a markdown table (for CLI + LLM digestion).

The output shape matches what the manual A/B analysis produced —
n / winrate / total_pnl / avg / best / worst / sharpe / PF / avg_hold /
symbols_touched — so switching from ad-hoc scripts to this module is
a drop-in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .clustering import (
    CLUSTER_LABELS, CLUSTER_DESCRIPTIONS,
    classify_trades_df,
)


@dataclass
class ClusterStats:
    cluster: str
    n: int
    winners: int
    losers: int
    winrate: float
    total_pnl: float
    avg_pnl: float
    best_pnl: float
    worst_pnl: float
    profit_factor: float
    sharpe_est: float
    avg_bars_held: float
    symbols_touched: int

    def to_row(self) -> dict:
        return {
            "cluster": self.cluster,
            "n": self.n,
            "winners": self.winners,
            "losers": self.losers,
            "winrate": self.winrate,
            "total_pnl": self.total_pnl,
            "avg_pnl": self.avg_pnl,
            "best_pnl": self.best_pnl,
            "worst_pnl": self.worst_pnl,
            "profit_factor": self.profit_factor,
            "sharpe_est": self.sharpe_est,
            "avg_bars_held": self.avg_bars_held,
            "symbols_touched": self.symbols_touched,
        }


@dataclass
class ClusterPerformanceReport:
    total_trades: int
    stats: list[ClusterStats]

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_row() for s in self.stats])


# ============================================================
# Aggregation
# ============================================================

def _sharpe(pnl: pd.Series) -> float:
    std = float(pnl.std())
    if std <= 0:
        return 0.0
    return float(pnl.mean()) / std * float(np.sqrt(365))


def _profit_factor(pnl: pd.Series) -> float:
    winners = pnl[pnl > 0].sum()
    losers = -pnl[pnl < 0].sum()
    if losers > 0:
        return float(winners / losers)
    if winners > 0:
        return float("inf")
    return 0.0


def _stats_for_group(cluster: str, grp: pd.DataFrame,
                     pnl_col: str, bars_col: str | None) -> ClusterStats:
    pnl = grp[pnl_col]
    n = len(grp)
    winners = int((pnl > 0).sum())
    losers = int((pnl < 0).sum())
    return ClusterStats(
        cluster=cluster,
        n=n, winners=winners, losers=losers,
        winrate=winners / n if n else 0.0,
        total_pnl=float(pnl.sum()),
        avg_pnl=float(pnl.mean()) if n else 0.0,
        best_pnl=float(pnl.max()) if n else 0.0,
        worst_pnl=float(pnl.min()) if n else 0.0,
        profit_factor=_profit_factor(pnl),
        sharpe_est=_sharpe(pnl),
        avg_bars_held=float(grp[bars_col].mean()) if bars_col and n else 0.0,
        symbols_touched=int(grp["symbol"].nunique()) if "symbol" in grp.columns else 0,
    )


def analyze(
    df: pd.DataFrame,
    schema: str = "backtest",
    pnl_col: str | None = None,
    bars_col: str | None = None,
) -> ClusterPerformanceReport:
    """Compute per-cluster stats. Returns clusters in the canonical order
    with zero-row placeholders for absent clusters so consumers always
    see a full 6-row table."""
    if df.empty:
        return ClusterPerformanceReport(
            total_trades=0,
            stats=[
                ClusterStats(
                    cluster=c, n=0, winners=0, losers=0,
                    winrate=0.0, total_pnl=0.0, avg_pnl=0.0,
                    best_pnl=0.0, worst_pnl=0.0, profit_factor=0.0,
                    sharpe_est=0.0, avg_bars_held=0.0, symbols_touched=0,
                )
                for c in CLUSTER_LABELS
            ],
        )
    if pnl_col is None:
        pnl_col = ("pnl_after_fees"
                   if "pnl_after_fees" in df.columns else "pnl")
    if bars_col is None and "bars_held" in df.columns:
        bars_col = "bars_held"

    labeled = df.copy()
    labeled["_cluster"] = classify_trades_df(labeled, schema=schema)

    stats: list[ClusterStats] = []
    for cluster in CLUSTER_LABELS:
        grp = labeled[labeled["_cluster"] == cluster]
        if grp.empty:
            stats.append(ClusterStats(
                cluster=cluster, n=0, winners=0, losers=0,
                winrate=0.0, total_pnl=0.0, avg_pnl=0.0,
                best_pnl=0.0, worst_pnl=0.0, profit_factor=0.0,
                sharpe_est=0.0, avg_bars_held=0.0, symbols_touched=0,
            ))
        else:
            stats.append(_stats_for_group(cluster, grp, pnl_col, bars_col))
    return ClusterPerformanceReport(total_trades=len(df), stats=stats)


# ============================================================
# Rendering
# ============================================================

def _fmt_pf(x: float) -> str:
    return "∞" if x == float("inf") else f"{x:.2f}"


def render_markdown(report: ClusterPerformanceReport) -> str:
    if report.total_trades == 0:
        return "No trades to analyze."
    header = ("| Cluster | N | WR | Total PnL | Avg | PF | Sharpe | "
              "Best | Worst | Symbols |")
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for s in report.stats:
        if s.n == 0:
            # Show a dashed row for absent clusters so operators know
            # the cluster exists but has no trades yet
            lines.append(
                f"| {s.cluster} | 0 | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {s.cluster} | {s.n} | {s.winrate:.1%} | "
            f"{s.total_pnl:+,.2f} | {s.avg_pnl:+,.2f} | "
            f"{_fmt_pf(s.profit_factor)} | {s.sharpe_est:.2f} | "
            f"{s.best_pnl:+,.2f} | {s.worst_pnl:+,.2f} | "
            f"{s.symbols_touched} |"
        )
    lines.append("")
    lines.append(f"_Total classified trades: {report.total_trades}_")
    return "\n".join(lines)
