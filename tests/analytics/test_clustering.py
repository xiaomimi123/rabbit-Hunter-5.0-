"""Tests for the trade clustering classifier.

Pins every predicate boundary + priority-ordering + both extractors
(backtest schema vs shadow schema). If a future refactor reorders the
predicates or renames a feature column, these tests break loudly.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rabbit_hunter.analytics.clustering import (
    CLUSTER_LABELS, TradeFeatures,
    classify, classify_backtest_row, classify_shadow_trade,
    classify_trades_df,
    features_from_backtest_row, features_from_shadow_trade,
)


def _f(side: str = "long", rsi: float = 50.0, z: float = 0.0,
       bb: float = 0.5, structure: str = "range", bos: int = 0) -> TradeFeatures:
    return TradeFeatures(
        side=side, rsi_14=rsi, zscore_20=z, bb_pct=bb,
        structure_regime=structure, bos_flag=bos,
    )


# ============================================================
# Individual cluster predicates
# ============================================================

def test_short_deep_oversold_rsi_is_cluster_1():
    assert classify(_f(side="short", rsi=25.0)) == "1_momentum_breakdown"


def test_short_deep_oversold_zscore_is_cluster_1():
    assert classify(_f(side="short", rsi=50.0, z=-2.0)) == "1_momentum_breakdown"


def test_short_deep_oversold_bb_pct_is_cluster_1():
    assert classify(_f(side="short", rsi=50.0, bb=0.05)) == "1_momentum_breakdown"


def test_long_deep_overbought_is_cluster_2():
    assert classify(_f(side="long", rsi=75.0)) == "2_momentum_breakout"


def test_range_with_bos_is_cluster_3():
    assert classify(_f(structure="range", bos=1)) == "3_range_breakout"


def test_range_with_high_z_is_cluster_3():
    assert classify(_f(structure="range", z=1.7)) == "3_range_breakout"


def test_long_in_uptrend_is_cluster_4():
    assert classify(_f(side="long", structure="uptrend")) == "4_trend_continuation"


def test_short_in_downtrend_is_cluster_4():
    assert classify(_f(side="short", structure="downtrend")) == "4_trend_continuation"


def test_long_in_downtrend_is_cluster_5():
    assert classify(_f(side="long", structure="downtrend")) == "5_trend_reversal"


def test_short_in_uptrend_is_cluster_5():
    assert classify(_f(side="short", structure="uptrend")) == "5_trend_reversal"


def test_no_match_is_cluster_6():
    """Long side, neutral RSI, range structure with no BOS and moderate z."""
    assert classify(_f(side="long", structure="range")) == "6_other"


# ============================================================
# Priority ordering — the earlier predicate wins when multiple match
# ============================================================

def test_momentum_beats_trend_context():
    """A short in a downtrend AT deep-oversold levels is Cluster 1
    (momentum breakdown), not Cluster 4 (trend continuation)."""
    assert classify(_f(side="short", rsi=25.0, structure="downtrend")) \
        == "1_momentum_breakdown"


def test_momentum_beats_range_breakout():
    """A short at deep-oversold IN a range with BOS still resolves to
    momentum breakdown."""
    assert classify(_f(side="short", rsi=25.0, structure="range", bos=1)) \
        == "1_momentum_breakdown"


# ============================================================
# Boundary values — strict less-than semantics on the momentum predicates
# ============================================================

def test_rsi_exactly_30_short_is_not_breakdown():
    """Predicate is rsi < 30 (strict). At 30 the trade drops to next
    tier (range/trend/other)."""
    # short, structure=range, no BOS, moderate z → should fall through
    r = classify(_f(side="short", rsi=30.0, structure="range", z=0.0))
    assert r != "1_momentum_breakdown"


def test_zscore_exactly_neg_1_8_short_is_not_breakdown():
    r = classify(_f(side="short", rsi=50.0, z=-1.8, structure="range"))
    assert r != "1_momentum_breakdown"


# ============================================================
# Extractors — backtest schema vs shadow schema
# ============================================================

def test_backtest_row_uses_t0_suffix():
    row = {"side": "short", "rsi_14_t0": 25.0,
           "zscore_20_t0": 0.0, "bb_pct_t0": 0.5,
           "structure_regime_t0": "range", "bos_flag_t0": 0}
    feats = features_from_backtest_row(row)
    assert feats.side == "short"
    assert feats.rsi_14 == 25.0
    assert classify_backtest_row(row) == "1_momentum_breakdown"


def test_shadow_trade_reads_entry_snapshot():
    trade = {
        "side": "short",
        "entry_snapshot": {
            "rsi_14": 25.0, "zscore_20": 0.0, "bb_pct": 0.5,
            "structure_regime": "range", "bos_flag": 0,
        },
    }
    feats = features_from_shadow_trade(trade)
    assert feats.rsi_14 == 25.0
    assert classify_shadow_trade(trade) == "1_momentum_breakdown"


def test_shadow_trade_missing_entry_snapshot_defaults_to_neutral():
    """A malformed shadow trade with no entry_snapshot should default
    to neutral features (RSI=50, z=0), landing in Cluster 6, NOT
    crashing."""
    assert classify_shadow_trade({"side": "long"}) == "6_other"


def test_nan_features_treated_as_neutral():
    """NaN inputs must map to safe defaults, not silently fire a
    momentum cluster."""
    row = {"side": "short", "rsi_14_t0": float("nan"),
           "zscore_20_t0": float("nan"), "bb_pct_t0": float("nan"),
           "structure_regime_t0": "range", "bos_flag_t0": 0}
    # Neutral RSI/z/bb → does NOT hit cluster 1
    assert classify_backtest_row(row) == "6_other"


# ============================================================
# Bulk API
# ============================================================

def test_classify_trades_df_backtest_schema():
    df = pd.DataFrame([
        {"side": "short", "rsi_14_t0": 25.0, "zscore_20_t0": 0.0,
         "bb_pct_t0": 0.5, "structure_regime_t0": "range", "bos_flag_t0": 0},
        {"side": "long", "rsi_14_t0": 75.0, "zscore_20_t0": 0.0,
         "bb_pct_t0": 0.5, "structure_regime_t0": "range", "bos_flag_t0": 0},
    ])
    out = classify_trades_df(df, schema="backtest")
    assert list(out) == ["1_momentum_breakdown", "2_momentum_breakout"]


def test_classify_trades_df_empty_returns_empty():
    out = classify_trades_df(pd.DataFrame(), schema="backtest")
    assert len(out) == 0


def test_classify_trades_df_rejects_unknown_schema():
    with pytest.raises(ValueError):
        classify_trades_df(pd.DataFrame({"a": [1]}), schema="martian")


def test_all_cluster_labels_reachable_via_classify():
    """Every label in CLUSTER_LABELS must be produced by at least one
    input — prevents adding a label that's never triggered."""
    seen: set[str] = set()
    seen.add(classify(_f(side="short", rsi=25.0)))
    seen.add(classify(_f(side="long", rsi=75.0)))
    seen.add(classify(_f(structure="range", bos=1)))
    seen.add(classify(_f(side="long", structure="uptrend")))
    seen.add(classify(_f(side="long", structure="downtrend")))
    seen.add(classify(_f(side="long", structure="range")))
    assert seen == set(CLUSTER_LABELS)
