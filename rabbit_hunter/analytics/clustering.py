"""Trade clustering — categorize every trade by "why did we open it".

This is the classifier that emerged from the manual A/B analysis of the
273-trade backtest. It maps each trade to one of 6 mutually-exclusive
clusters using features captured at entry time:

  1. momentum_breakdown  — short into deep oversold (RSI<30 / z<-1.8 / bb_pct<0.1)
  2. momentum_breakout   — long into deep overbought
  3. range_breakout      — trade in a range-bound structure with BOS or |z|>1.5
  4. trend_continuation  — long in uptrend / short in downtrend (WEAK — WR 0%)
  5. trend_reversal      — long in downtrend / short in uptrend
  6. other               — didn't fit any of the above

The classification is priority-ordered: momentum first (strongest signal),
then range breakouts, then trend context. A trade that matches multiple
clusters gets assigned to the FIRST one that fires — matches how a human
trader labels the dominant reason.

The feature columns needed live on both:
  - backtest trades.parquet (with `_t0` suffix from the entry snapshot)
  - shadow ledger closed_trades (nested inside `entry_snapshot`)

Two entry points — `classify_row` (raw dict) and `classify_trades_df`
(bulk pandas) — so both consumers use the same code path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


# ============================================================
# Cluster definitions — the ordered list is the priority for the
# classify_* functions; the first matching predicate wins.
# ============================================================

CLUSTER_LABELS = [
    "1_momentum_breakdown",
    "2_momentum_breakout",
    "3_range_breakout",
    "4_trend_continuation",
    "5_trend_reversal",
    "6_other",
]


CLUSTER_DESCRIPTIONS: dict[str, str] = {
    "1_momentum_breakdown":
        "Short into deep oversold — RSI<30 or z<-1.8 or bb_pct<0.1",
    "2_momentum_breakout":
        "Long into deep overbought — RSI>70 or z>1.8 or bb_pct>0.9",
    "3_range_breakout":
        "Range structure + BOS or |z|>1.5",
    "4_trend_continuation":
        "Long uptrend or short downtrend (weak alpha in prior study)",
    "5_trend_reversal":
        "Long downtrend or short uptrend (counter-trend entry)",
    "6_other":
        "None of the above patterns matched",
}


@dataclass(frozen=True)
class TradeFeatures:
    """The features the classifier needs. Kept in a dataclass so both
    entry points normalize to the same shape."""
    side: str                     # "long" | "short"
    rsi_14: float
    zscore_20: float
    bb_pct: float
    structure_regime: str         # "uptrend" | "downtrend" | "range"
    bos_flag: int


# ============================================================
# Extraction — normalize the different-shape inputs to TradeFeatures
# ============================================================

def _f(v: Any, default: float) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        if f != f:   # NaN check
            return default
        return f
    except (TypeError, ValueError):
        return default


def features_from_backtest_row(row: dict[str, Any]) -> TradeFeatures:
    """Extract classifier inputs from a row of the backtest trades.parquet.
    Backtest snapshots suffix entry-time features with '_t0'."""
    return TradeFeatures(
        side=str(row.get("side", "long")),
        rsi_14=_f(row.get("rsi_14_t0"), 50.0),
        zscore_20=_f(row.get("zscore_20_t0"), 0.0),
        bb_pct=_f(row.get("bb_pct_t0"), 0.5),
        structure_regime=str(row.get("structure_regime_t0", "range")),
        bos_flag=int(_f(row.get("bos_flag_t0"), 0)),
    )


def features_from_shadow_trade(trade: dict[str, Any]) -> TradeFeatures:
    """Extract classifier inputs from a shadow-mode closed_trades entry.
    Shadow trades keep the entry snapshot nested under 'entry_snapshot'."""
    side = str(trade.get("side", "long"))
    snap = trade.get("entry_snapshot") or {}
    return TradeFeatures(
        side=side,
        rsi_14=_f(snap.get("rsi_14"), 50.0),
        zscore_20=_f(snap.get("zscore_20"), 0.0),
        bb_pct=_f(snap.get("bb_pct"), 0.5),
        structure_regime=str(snap.get("structure_regime", "range")),
        bos_flag=int(_f(snap.get("bos_flag"), 0)),
    )


# ============================================================
# Classifier — priority-ordered predicates
# ============================================================

def classify(features: TradeFeatures) -> str:
    side = features.side
    rsi = features.rsi_14
    z = features.zscore_20
    bb_pct = features.bb_pct
    structure = features.structure_regime
    bos = features.bos_flag

    if side == "short" and (rsi < 30 or z < -1.8 or bb_pct < 0.1):
        return "1_momentum_breakdown"
    if side == "long" and (rsi > 70 or z > 1.8 or bb_pct > 0.9):
        return "2_momentum_breakout"
    if structure == "range" and (bos == 1 or abs(z) > 1.5):
        return "3_range_breakout"
    if (side == "long" and structure == "uptrend") \
            or (side == "short" and structure == "downtrend"):
        return "4_trend_continuation"
    if (side == "long" and structure == "downtrend") \
            or (side == "short" and structure == "uptrend"):
        return "5_trend_reversal"
    return "6_other"


def classify_backtest_row(row: dict) -> str:
    return classify(features_from_backtest_row(row))


def classify_shadow_trade(trade: dict) -> str:
    return classify(features_from_shadow_trade(trade))


# ============================================================
# Bulk API — for reports over trades.parquet
# ============================================================

def classify_trades_df(
    df: pd.DataFrame,
    schema: str = "backtest",
) -> pd.Series:
    """Attach a cluster label per row. `schema` picks the extractor:

      - "backtest": columns like rsi_14_t0, zscore_20_t0, etc.
      - "shadow":   entry_snapshot column contains a dict per row.

    Returns a Series aligned with df.index.
    """
    if schema not in ("backtest", "shadow"):
        raise ValueError(f"unknown schema={schema}")
    if df.empty:
        return pd.Series([], dtype="object", index=df.index)
    if schema == "backtest":
        rows = df.to_dict(orient="records")
        return pd.Series([classify_backtest_row(r) for r in rows], index=df.index)
    # shadow — the entry_snapshot is a nested dict/JSON
    def _row_cluster(r: dict) -> str:
        return classify_shadow_trade(r)
    return pd.Series(
        [_row_cluster(r) for r in df.to_dict(orient="records")],
        index=df.index,
    )
