"""Tests for the concept drift detector."""
from __future__ import annotations

import pytest

from rabbit_hunter.analytics.baseline import (
    BaselineSnapshot, ClusterBaseline,
)
from rabbit_hunter.analytics.cluster_performance import (
    ClusterPerformanceReport, ClusterStats,
)
from rabbit_hunter.analytics.drift import (
    DriftThresholds, compare,
)


def _baseline_one_cluster(
    n: int = 100, wr: float = 0.6, avg: float = 20.0,
    sharpe: float = 3.0, std: float = 40.0,
) -> BaselineSnapshot:
    return BaselineSnapshot(
        tag="test", created_at_utc="2026-01-01T00:00:00Z",
        source_report_dir="reports/x", total_trades=n,
        clusters=[
            ClusterBaseline(
                cluster="1_momentum_breakdown",
                n=n, winrate=wr, avg_pnl=avg,
                sharpe_est=sharpe, profit_factor=1.5,
                pnl_std=std,
            ),
        ],
    )


def _live_one_cluster(
    cluster: str = "1_momentum_breakdown",
    n: int = 30, wr: float = 0.6, avg: float = 20.0,
    total: float | None = None,
) -> ClusterPerformanceReport:
    if total is None:
        total = avg * n
    winners = int(round(wr * n))
    stats = [
        ClusterStats(
            cluster=cluster, n=n, winners=winners, losers=n - winners,
            winrate=wr, total_pnl=total, avg_pnl=avg,
            best_pnl=avg * 2, worst_pnl=avg * -1,
            profit_factor=1.5, sharpe_est=3.0,
            avg_bars_held=5.0, symbols_touched=1,
        ),
    ]
    return ClusterPerformanceReport(total_trades=n, stats=stats)


# ============================================================
# OK path
# ============================================================

def test_ok_when_live_matches_baseline():
    baseline = _baseline_one_cluster(wr=0.6, avg=20.0, std=40.0)
    live = _live_one_cluster(n=30, wr=0.6, avg=20.0)
    r = compare(baseline, live)
    assert r.ok is True
    assert len(r.findings) == 1
    assert r.findings[0].triggered is False


def test_ok_below_min_live_trades_never_fires():
    """A wildly-off tiny sample must not trigger — sample size matters."""
    baseline = _baseline_one_cluster(wr=0.6, avg=20.0, std=40.0)
    live = _live_one_cluster(n=5, wr=0.0, avg=-100.0)   # only 5 trades
    r = compare(baseline, live, DriftThresholds(min_live_trades=20))
    assert r.ok is True
    assert r.findings[0].triggered is False
    assert "below min_n" in r.findings[0].reason


# ============================================================
# Winrate drift
# ============================================================

def test_alert_on_large_winrate_drop():
    baseline = _baseline_one_cluster(wr=0.6, avg=20.0, std=40.0)
    live = _live_one_cluster(n=30, wr=0.30, avg=20.0)  # -30pp
    r = compare(baseline, live,
                 DriftThresholds(winrate_delta_alert=0.15,
                                 avg_pnl_zscore_alert=100.0,  # disable
                                 min_live_trades=20))
    assert r.ok is False
    assert r.findings[0].triggered is True
    assert "winrate" in r.findings[0].reason


def test_alert_on_large_winrate_gain_also_fires():
    """|Δ WR| is bidirectional — a spike is also drift."""
    baseline = _baseline_one_cluster(wr=0.5, avg=20.0, std=40.0)
    live = _live_one_cluster(n=30, wr=0.95, avg=20.0)  # +45pp
    r = compare(baseline, live,
                 DriftThresholds(winrate_delta_alert=0.15,
                                 avg_pnl_zscore_alert=100.0,
                                 min_live_trades=20))
    assert r.findings[0].triggered is True


def test_boundary_winrate_delta_does_not_trigger():
    baseline = _baseline_one_cluster(wr=0.6, avg=20.0, std=40.0)
    live = _live_one_cluster(n=30, wr=0.7499, avg=20.0)   # just under +15pp
    r = compare(baseline, live,
                 DriftThresholds(winrate_delta_alert=0.15,
                                 avg_pnl_zscore_alert=100.0,
                                 min_live_trades=20))
    assert not r.findings[0].triggered


# ============================================================
# Avg PnL z-score drift
# ============================================================

def test_alert_on_avg_pnl_zscore():
    """Live avg PnL = -20 vs baseline avg = 20, std=40, n=25.
    z = (-40) * sqrt(25) / 40 = -5.0 → alerts."""
    baseline = _baseline_one_cluster(wr=0.6, avg=20.0, std=40.0)
    live = _live_one_cluster(n=25, wr=0.6, avg=-20.0)
    r = compare(baseline, live,
                 DriftThresholds(winrate_delta_alert=100.0,
                                 avg_pnl_zscore_alert=2.5,
                                 min_live_trades=20))
    assert r.findings[0].triggered
    assert abs(r.findings[0].avg_pnl_zscore) >= 2.5


# ============================================================
# Mix drift — new cluster in live not seen in baseline
# ============================================================

def test_unexpected_live_cluster_flags():
    baseline = _baseline_one_cluster()
    # Live has a cluster (2_momentum_breakout) that baseline never covered
    live_stats = [
        ClusterStats(
            cluster="2_momentum_breakout", n=25, winners=15, losers=10,
            winrate=0.6, total_pnl=100.0, avg_pnl=4.0,
            best_pnl=50, worst_pnl=-20, profit_factor=1.4,
            sharpe_est=1.0, avg_bars_held=6.0, symbols_touched=1,
        ),
    ]
    live = ClusterPerformanceReport(total_trades=25, stats=live_stats)
    r = compare(baseline, live)
    assert not r.ok
    assert r.live_only == ["2_momentum_breakout"]
    assert r.baseline_only == ["1_momentum_breakdown"]


def test_baseline_only_is_informational_not_alert():
    """When baseline saw a cluster and live hasn't yet (n=0), that's a
    "still waiting" info, not an alert on its own."""
    baseline = _baseline_one_cluster()
    live = ClusterPerformanceReport(
        total_trades=0,
        stats=[
            # Zero live stats — no trades for cluster 1 yet
            ClusterStats(
                cluster="1_momentum_breakdown", n=0, winners=0, losers=0,
                winrate=0.0, total_pnl=0.0, avg_pnl=0.0,
                best_pnl=0.0, worst_pnl=0.0, profit_factor=0.0,
                sharpe_est=0.0, avg_bars_held=0.0, symbols_touched=0,
            ),
        ],
    )
    r = compare(baseline, live)
    # baseline_only is populated, no findings triggered → OK
    assert r.baseline_only == ["1_momentum_breakdown"]
    assert r.ok is True


# ============================================================
# as_lines() logging
# ============================================================

def test_as_lines_ok_starts_with_ok():
    baseline = _baseline_one_cluster()
    live = _live_one_cluster(n=30, wr=0.6, avg=20.0)
    lines = compare(baseline, live).as_lines()
    assert lines[0].startswith("drift: OK")


def test_as_lines_alert_starts_with_alert():
    baseline = _baseline_one_cluster()
    live = _live_one_cluster(n=30, wr=0.3, avg=20.0)
    r = compare(baseline, live,
                 DriftThresholds(winrate_delta_alert=0.15,
                                 avg_pnl_zscore_alert=100.0,
                                 min_live_trades=20))
    lines = r.as_lines()
    assert lines[0].startswith("drift: ALERT")
    assert any("!!" in ln for ln in lines[1:])
