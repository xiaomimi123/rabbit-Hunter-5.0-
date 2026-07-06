"""Tests for baseline snapshot serialization."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rabbit_hunter.analytics.baseline import (
    BaselineSnapshot, ClusterBaseline,
    build_baseline_from_report, save, load,
)
from rabbit_hunter.analytics.cluster_performance import analyze


def _mk_trades(n_short_win: int = 10, n_short_lose: int = 5,
               n_long_win: int = 5) -> pd.DataFrame:
    rows = []
    for _ in range(n_short_win):
        rows.append({"side": "short", "pnl_after_fees": +30.0,
                     "rsi_14_t0": 25.0, "zscore_20_t0": 0.0,
                     "bb_pct_t0": 0.5, "structure_regime_t0": "range",
                     "bos_flag_t0": 0, "symbol": "BTC-USDT-SWAP",
                     "bars_held": 5})
    for _ in range(n_short_lose):
        rows.append({"side": "short", "pnl_after_fees": -20.0,
                     "rsi_14_t0": 25.0, "zscore_20_t0": 0.0,
                     "bb_pct_t0": 0.5, "structure_regime_t0": "range",
                     "bos_flag_t0": 0, "symbol": "BTC-USDT-SWAP",
                     "bars_held": 5})
    for _ in range(n_long_win):
        rows.append({"side": "long", "pnl_after_fees": +40.0,
                     "rsi_14_t0": 75.0, "zscore_20_t0": 0.0,
                     "bb_pct_t0": 0.5, "structure_regime_t0": "range",
                     "bos_flag_t0": 0, "symbol": "ETH-USDT-SWAP",
                     "bars_held": 5})
    return pd.DataFrame(rows)


def test_build_baseline_from_report_skips_empty_clusters():
    df = _mk_trades()
    report = analyze(df, schema="backtest")
    snap = build_baseline_from_report(report, tag="v0.1.3",
                                       source_report_dir="reports/x")
    labels = {c.cluster for c in snap.clusters}
    # Only cluster 1 and 2 have trades in this fixture
    assert labels == {"1_momentum_breakdown", "2_momentum_breakout"}
    assert snap.tag == "v0.1.3"
    assert snap.total_trades == len(df)


def test_baseline_json_roundtrip(tmp_path: Path):
    df = _mk_trades()
    report = analyze(df, schema="backtest")
    snap = build_baseline_from_report(report, tag="v0.1.3",
                                       source_report_dir="reports/x")
    p = save(snap, tmp_path / "baseline.json")
    assert p.exists()
    loaded = load(p)
    assert loaded.tag == snap.tag
    assert loaded.total_trades == snap.total_trades
    assert len(loaded.clusters) == len(snap.clusters)
    assert loaded.clusters[0].cluster == snap.clusters[0].cluster


def test_baseline_json_is_valid_json(tmp_path: Path):
    import json
    df = _mk_trades()
    snap = build_baseline_from_report(
        analyze(df, schema="backtest"),
        tag="v0.1.3", source_report_dir="reports/x",
    )
    text = snap.to_json()
    # Should parse and contain expected top-level keys
    data = json.loads(text)
    assert set(data.keys()) >= {
        "tag", "created_at_utc", "source_report_dir",
        "total_trades", "clusters",
    }


def test_by_cluster_index():
    snap = BaselineSnapshot(
        tag="t", created_at_utc="2026-01-01", source_report_dir="",
        total_trades=1,
        clusters=[
            ClusterBaseline("1_momentum_breakdown", 100, 0.6, 20.0, 3.0, 1.5, 40.0),
        ],
    )
    idx = snap.by_cluster()
    assert "1_momentum_breakdown" in idx
    assert idx["1_momentum_breakdown"].n == 100
