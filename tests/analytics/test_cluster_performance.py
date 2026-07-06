"""Tests for the per-cluster performance analyzer."""
from __future__ import annotations

import pandas as pd
import pytest

from rabbit_hunter.analytics.cluster_performance import (
    analyze, render_markdown,
)
from rabbit_hunter.analytics.clustering import CLUSTER_LABELS


def _row(side: str, pnl: float, rsi: float = 50.0, z: float = 0.0,
         bb: float = 0.5, structure: str = "range", bos: int = 0,
         symbol: str = "BTC-USDT-SWAP", bars: int = 5) -> dict:
    return {
        "side": side, "pnl_after_fees": pnl,
        "rsi_14_t0": rsi, "zscore_20_t0": z, "bb_pct_t0": bb,
        "structure_regime_t0": structure, "bos_flag_t0": bos,
        "symbol": symbol, "bars_held": bars,
    }


# ============================================================
# Empty inputs
# ============================================================

def test_analyze_empty_df_returns_full_zero_table():
    r = analyze(pd.DataFrame(), schema="backtest")
    assert r.total_trades == 0
    # Every canonical cluster is present as a placeholder row
    assert [s.cluster for s in r.stats] == CLUSTER_LABELS
    assert all(s.n == 0 for s in r.stats)


# ============================================================
# Correctness — a known-shape input yields known stats
# ============================================================

def test_analyze_groups_by_cluster_correctly():
    df = pd.DataFrame([
        # Two shorts into deep oversold → cluster 1
        _row("short", +100.0, rsi=25.0),
        _row("short", -40.0,  rsi=25.0, symbol="ETH-USDT-SWAP"),
        # One long into deep overbought → cluster 2
        _row("long",  +80.0,  rsi=75.0),
        # One trend continuation → cluster 4
        _row("long",  -30.0,  structure="uptrend"),
    ])
    r = analyze(df, schema="backtest")
    assert r.total_trades == 4
    stats_by_cluster = {s.cluster: s for s in r.stats}

    c1 = stats_by_cluster["1_momentum_breakdown"]
    assert c1.n == 2
    assert c1.winners == 1
    assert c1.losers == 1
    assert c1.winrate == 0.5
    assert c1.total_pnl == pytest.approx(60.0)
    assert c1.symbols_touched == 2   # BTC + ETH

    c2 = stats_by_cluster["2_momentum_breakout"]
    assert c2.n == 1
    assert c2.winrate == 1.0
    assert c2.profit_factor == float("inf")

    c4 = stats_by_cluster["4_trend_continuation"]
    assert c4.n == 1
    assert c4.winrate == 0.0
    assert c4.total_pnl == pytest.approx(-30.0)

    # Empty clusters still appear
    c3 = stats_by_cluster["3_range_breakout"]
    assert c3.n == 0


def test_analyze_uses_pnl_col_override():
    """When trades.parquet uses `pnl` instead of `pnl_after_fees`,
    the analyzer must still work."""
    df = pd.DataFrame([
        {"side": "short", "pnl": +50.0, "rsi_14_t0": 25.0,
         "zscore_20_t0": 0.0, "bb_pct_t0": 0.5,
         "structure_regime_t0": "range", "bos_flag_t0": 0,
         "symbol": "BTC-USDT-SWAP", "bars_held": 5},
    ])
    r = analyze(df, schema="backtest")
    assert r.stats[0].total_pnl == 50.0


# ============================================================
# Rendering
# ============================================================

def test_render_markdown_shows_all_clusters():
    df = pd.DataFrame([_row("short", +100.0, rsi=25.0)])
    md = render_markdown(analyze(df, schema="backtest"))
    for cluster in CLUSTER_LABELS:
        assert cluster in md


def test_render_markdown_dashes_empty_clusters():
    """Empty clusters render as `— — —` placeholders, NOT as zeros
    (would look like the classifier failed)."""
    df = pd.DataFrame([_row("short", +100.0, rsi=25.0)])
    md = render_markdown(analyze(df, schema="backtest"))
    # cluster 2 (breakout) has 0 trades → dashes on its line
    cluster_2_line = [
        line for line in md.splitlines()
        if line.startswith("| 2_momentum_breakout")
    ][0]
    assert "— |" in cluster_2_line


def test_render_markdown_empty_report():
    md = render_markdown(analyze(pd.DataFrame(), schema="backtest"))
    assert md == "No trades to analyze."


# ============================================================
# Shadow schema
# ============================================================

def test_analyze_works_with_shadow_schema():
    df = pd.DataFrame([{
        "side": "short",
        "pnl_after_fees": +100.0,
        "entry_snapshot": {
            "rsi_14": 25.0, "zscore_20": 0.0, "bb_pct": 0.5,
            "structure_regime": "range", "bos_flag": 0,
        },
        "symbol": "BTC-USDT-SWAP", "bars_held": 5,
    }])
    r = analyze(df, schema="shadow")
    stats_by_cluster = {s.cluster: s for s in r.stats}
    assert stats_by_cluster["1_momentum_breakdown"].n == 1
