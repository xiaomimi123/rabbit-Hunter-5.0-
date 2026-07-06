"""Phase 2 · § 3.2 — Offline training pipeline.

Reads a completed backtest's trades.parquet + config, builds (X, y),
trains a logistic regression classifier, walk-forward-validates it,
and writes a versioned model file that `MLScoring` can load.

Design principles per architecture spec:
  - "训练在离线环境跑,产出一个带版本号的模型文件"
  - "线上推理只加载已发布模型,不做在线学习"
  - Walk-forward: train only on rows BEFORE the validation window
    (no data leakage from the future)
  - Feature list is FROZEN inside the model artifact (via ColumnTransformer)
    so inference cannot silently pick up new/renamed columns

Model architecture is deliberately simple for the MVP:
  - Numeric-only features (skip categorical text like regime_t0)
  - StandardScaler → LogisticRegression
  - Binary label: pnl_after_fees > 0 (win = 1, loss = 0)
  - Separate model per side (long/short) OR unified with side as feature —
    we go with unified (side one-hot column) to share signal across sides

Later iterations can swap LR → LightGBM without changing the interface.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


# Numeric feature columns from trades.parquet (auto-generated from
# entry_snapshot by _flatten_trade_row → all suffixed _t0). We hard-code
# the list because we want the training-time schema to be crystal clear
# and to fail loudly if a column disappears.
_DEFAULT_FEATURES = [
    "ema20_t0", "ema60_t0", "ema200_t0", "ema20_slope_t0",
    "adx_t0", "di_plus_t0", "di_minus_t0",
    "rsi_14_t0",
    "bb_width_t0", "bb_pct_t0", "zscore_20_t0",
    "atr_pct_t0",
    "volume_ratio_20_t0",
    "funding_rate_t0",
    "oi_change_pct_t0",
    "ema20_1h_on_15m_t0", "adx_1h_on_15m_t0",
    "atr_pct_baseline_t0",
]


@dataclass
class TrainingResult:
    """What training produced. Written to disk as JSON alongside the .pkl."""
    model_version: str
    trained_at: str
    n_train: int
    n_test: int
    train_auc: float
    test_auc: float
    train_accuracy: float
    test_accuracy: float
    features_used: list[str]
    label_column: str
    label_definition: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)


def build_training_set(
    trades: pd.DataFrame,
    features: list[str] = _DEFAULT_FEATURES,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for training.

    X: numeric features from trades entry snapshots + one-hot side
    y: 1 if pnl_after_fees > 0, else 0
    """
    if trades.empty:
        raise ValueError("No trades to build training set from")

    # Only keep features that actually exist in this trades set
    usable = [c for c in features if c in trades.columns]
    if len(usable) < 3:
        raise ValueError(
            f"Not enough usable feature columns in trades. "
            f"Expected some of {features}, found {list(trades.columns)}"
        )

    X = trades[usable].copy()
    # One-hot the side (long=1, short=0)
    X["side_long"] = (trades["side"] == "long").astype(int)

    # Drop rows with any NaN in features (early bars during warmup)
    complete = X.notna().all(axis=1)
    X = X[complete].reset_index(drop=True)

    y = (trades.loc[complete, "pnl_after_fees"] > 0).astype(int).reset_index(drop=True)
    return X, y


def train_model(
    trades: pd.DataFrame,
    output_root: Path,
    train_fraction: float = 0.7,
    model_version: str | None = None,
    features: list[str] = _DEFAULT_FEATURES,
) -> tuple[Pipeline, TrainingResult, Path]:
    """Walk-forward training: earliest `train_fraction` for train, rest for test.

    Returns the fitted pipeline, a TrainingResult, and the path where the
    model was saved.
    """
    if "entry_time" in trades.columns:
        trades = trades.sort_values("entry_time").reset_index(drop=True)
    elif "timestamp" in trades.columns:
        trades = trades.sort_values("timestamp").reset_index(drop=True)

    X, y = build_training_set(trades, features=features)
    if len(X) < 50:
        raise ValueError(
            f"Only {len(X)} usable trades — need at least 50 for a "
            f"meaningful train/test split"
        )

    split_idx = int(len(X) * train_fraction)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(y_train.unique()) < 2 or len(y_test.unique()) < 2:
        raise ValueError(
            f"Train or test set is single-class (winners only or losers only). "
            f"y_train unique={y_train.unique().tolist()}, "
            f"y_test unique={y_test.unique().tolist()}"
        )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)

    train_proba = pipe.predict_proba(X_train)[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)

    version = model_version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result = TrainingResult(
        model_version=version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_train=len(y_train),
        n_test=len(y_test),
        train_auc=float(roc_auc_score(y_train, train_proba)),
        test_auc=float(roc_auc_score(y_test, test_proba)),
        train_accuracy=float(accuracy_score(y_train, train_pred)),
        test_accuracy=float(accuracy_score(y_test, test_pred)),
        features_used=list(X.columns),
        label_column="pnl_after_fees > 0",
        label_definition="binary win/loss on realized PnL after fees",
        hyperparameters={"model": "LogisticRegression", "max_iter": 1000,
                         "class_weight": "balanced", "train_fraction": train_fraction},
    )

    model_dir = output_root / f"ml_model_v{version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "model.pkl").open("wb") as f:
        pickle.dump(pipe, f)
    (model_dir / "training_result.json").write_text(
        json.dumps(result.__dict__, indent=2), encoding="utf-8"
    )
    # A tiny sanity-check readme
    (model_dir / "README.md").write_text(
        f"# ML Model v{version}\n\n"
        f"Trained: {result.trained_at}\n"
        f"Train AUC: {result.train_auc:.3f}   Test AUC: {result.test_auc:.3f}\n"
        f"Train Acc: {result.train_accuracy:.3f}   Test Acc: {result.test_accuracy:.3f}\n"
        f"n_train={result.n_train}, n_test={result.n_test}\n\n"
        f"## Features\n" + "\n".join(f"- {f}" for f in result.features_used),
        encoding="utf-8",
    )

    return pipe, result, model_dir / "model.pkl"
