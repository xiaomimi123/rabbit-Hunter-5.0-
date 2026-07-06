"""Tests for the v0.1.9 MLScoring fast path.

The optimization replaces per-call pandas DataFrame construction +
sklearn Pipeline.predict_proba with direct numpy scaling + booster
prediction. Both must produce identical probabilities — otherwise a
speedup would silently change strategy behavior.

The measured speedup on the real 3-symbol backtest was 6.8× (242s →
36s) with identical 88 trades / same PnL.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rabbit_hunter.ml.ml_scoring import MLScoring, MLScoringParams


FEATURES = ["ema20_t0", "rsi_14_t0", "adx_t0", "atr_pct_t0", "side_long"]


def _make_model(tmp_path: Path) -> Path:
    """Train a tiny LogisticRegression on synthetic data + pickle it in
    the shape MLScoring expects (Pipeline with 'scaler' + 'clf')."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "ema20_t0": rng.normal(100, 5, n),
        "rsi_14_t0": rng.normal(50, 10, n),
        "adx_t0": rng.normal(25, 5, n),
        "atr_pct_t0": rng.normal(0.01, 0.002, n),
        "side_long": rng.integers(0, 2, n),
    })
    y = (X["rsi_14_t0"] < 30).astype(int)   # signal
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=500))])
    pipe.fit(X, y)
    import pickle
    p = tmp_path / "model.pkl"
    with p.open("wb") as f:
        pickle.dump(pipe, f)
    return p


def test_predict_batch_matches_pipeline_predict_proba(tmp_path):
    """The numpy fast path MUST return the same probs as the slow
    (pandas + sklearn Pipeline) path — otherwise the speedup silently
    changes strategy behavior."""
    model_path = _make_model(tmp_path)
    ml = MLScoring(MLScoringParams(model_path=str(model_path)))
    # Real feature dict (unsuffixed keys, matching what the Feature Engine
    # produces at inference)
    features_row = {
        "ema20": 105.0, "rsi_14": 28.0, "adx": 30.0, "atr_pct": 0.012,
    }
    # Fast path
    long_prob, short_prob = ml._predict_batch(features_row)

    # Slow reference: build the same X the fast path builds, but through
    # the whole sklearn pipeline
    vec = ml._extract_features(features_row)
    assert vec is not None
    X_long = vec.copy()
    X_short = vec.copy()
    if ml._side_long_idx >= 0:
        X_long[ml._side_long_idx] = 1.0
        X_short[ml._side_long_idx] = 0.0
    X_ref = pd.DataFrame([X_long, X_short], columns=ml._feature_names)
    ref = ml._pipe.predict_proba(X_ref)

    assert long_prob == pytest.approx(ref[0, 1], rel=1e-9, abs=1e-12)
    assert short_prob == pytest.approx(ref[1, 1], rel=1e-9, abs=1e-12)


def test_predict_batch_returns_nan_when_feature_missing(tmp_path):
    model_path = _make_model(tmp_path)
    ml = MLScoring(MLScoringParams(model_path=str(model_path)))
    features_row = {"ema20": 105.0}  # missing rsi_14, adx, atr_pct
    long_p, short_p = ml._predict_batch(features_row)
    assert long_p != long_p   # NaN
    assert short_p != short_p


def test_predict_batch_returns_nan_when_feature_is_nan(tmp_path):
    model_path = _make_model(tmp_path)
    ml = MLScoring(MLScoringParams(model_path=str(model_path)))
    features_row = {
        "ema20": 105.0, "rsi_14": float("nan"),
        "adx": 30.0, "atr_pct": 0.012,
    }
    long_p, short_p = ml._predict_batch(features_row)
    assert long_p != long_p


def test_score_output_uses_batch(tmp_path):
    """The public score() API returns something reasonable — not testing
    exact values, but that it doesn't fall over on the fast path."""
    model_path = _make_model(tmp_path)
    ml = MLScoring(MLScoringParams(model_path=str(model_path)))
    out = ml.score(
        features_row={
            "ema20": 105.0, "rsi_14": 28.0, "adx": 30.0, "atr_pct": 0.012,
        },
        features_history=pd.DataFrame(),
    )
    assert 0.0 <= out.long <= 1.0
    assert 0.0 <= out.short <= 1.0
    assert "prob_win_long" in out.components
    assert "prob_win_short" in out.components
