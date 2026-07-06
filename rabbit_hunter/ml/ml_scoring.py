"""Phase 2 · § 3.2 — ML-based BaseStrategy plugin.

Loads a pickled sklearn Pipeline (trained by `training.train_model`) and
scores every bar. Feature list is baked into the model via the trained
StandardScaler → no chance of column mismatch at inference time.

Same interface as TrendFollowing / PriceAction — plugs into the router
via the standard strategy registry. Composer combines its output with
rule-based strategies exactly as before.

Design constraints per architecture spec § 4.1:
  - "线上推理只加载已发布模型,不做在线学习"  → we never call .fit()
  - Deterministic: same features_row → same score, always
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..scoring_engine.base import BaseStrategy, ScoreOutput


@dataclass(frozen=True)
class MLScoringParams:
    model_path: str
    # Threshold for turning probability into a "confident enough" score.
    # A prob-of-win of 0.5 is coin-flip; we linearly rescale [threshold, 1.0]
    # into [0, 1] so the router sees "how far above chance is this."
    prob_threshold: float = 0.5
    # Which side does this model handle? Set to "both" (default) to have it
    # score both long/short (using side_long feature). A future refinement
    # could train separate long-only / short-only models.
    side_mode: str = "both"


class MLScoring(BaseStrategy):
    """Score-based strategy backed by a pickled sklearn model."""

    name = "ml_scoring"
    version = "0.1.0"

    def __init__(self, params: MLScoringParams):
        self.params = params
        path = Path(params.model_path)
        if not path.exists():
            raise FileNotFoundError(f"ML model not found at {path}")
        with path.open("rb") as f:
            self._pipe = pickle.load(f)
        # Recover the training feature list from the pickled StandardScaler
        try:
            scaler = self._pipe.named_steps["scaler"]
            self._feature_names = list(scaler.feature_names_in_)
        except Exception as e:
            raise RuntimeError(
                f"Could not recover feature list from pipeline: {e}. "
                f"Was the model trained with `train_model`?"
            )
        # v0.1.9 fast path — extract the scaler's mean/scale and the
        # classifier for direct numpy prediction, bypassing sklearn's
        # per-call feature-name validation (profiling showed 89% of
        # backtest CPU was pandas DataFrame construction + sklearn
        # validation, not the actual inference).
        self._scaler_mean = np.asarray(scaler.mean_, dtype=np.float64)
        self._scaler_scale = np.asarray(scaler.scale_, dtype=np.float64)
        self._clf = self._pipe.named_steps["clf"]
        # Index of the side_long column so we can flip it long vs short
        # without touching the rest of the feature vector.
        try:
            self._side_long_idx = self._feature_names.index("side_long")
        except ValueError:
            self._side_long_idx = -1

    def _extract_features(self, features_row: dict) -> np.ndarray | None:
        """Pull the training feature vector out of features_row (no
        DataFrame roundtrip). Returns None if any required feature is
        missing/NaN — caller treats that as a score of 0."""
        n = len(self._feature_names)
        vec = np.empty(n, dtype=np.float64)
        for i, feat in enumerate(self._feature_names):
            if i == self._side_long_idx:
                vec[i] = 0.0   # placeholder; caller sets per side
                continue
            stripped = feat[:-3] if feat.endswith("_t0") else feat
            v = features_row.get(stripped, features_row.get(feat))
            if v is None:
                return None
            fv = float(v)
            if fv != fv:   # NaN
                return None
            vec[i] = fv
        return vec

    def _predict_batch(self, features_row: dict) -> tuple[float, float]:
        """Predict long AND short in a single call. Returns
        (prob_long, prob_short). NaN when features are missing."""
        vec = self._extract_features(features_row)
        if vec is None:
            return float("nan"), float("nan")
        # Two rows: [long, short], differing only in side_long.
        X = np.stack([vec, vec.copy()])
        if self._side_long_idx >= 0:
            X[0, self._side_long_idx] = 1.0
            X[1, self._side_long_idx] = 0.0
        # Apply StandardScaler manually — pure numpy, no per-call sklearn
        # validation overhead. Equivalent to scaler.transform(X).
        Xs = (X - self._scaler_mean) / self._scaler_scale
        proba = self._clf.predict_proba(Xs)
        return float(proba[0, 1]), float(proba[1, 1])

    def _prob_to_score(self, prob: float) -> float:
        if np.isnan(prob):
            return 0.0
        if prob < self.params.prob_threshold:
            return 0.0
        # Rescale [threshold, 1.0] → [0.0, 1.0]
        thr = self.params.prob_threshold
        return (prob - thr) / max(1.0 - thr, 1e-9)

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        long_prob, short_prob = self._predict_batch(features_row)

        long_score = self._prob_to_score(long_prob)
        short_score = self._prob_to_score(short_prob)

        return ScoreOutput(
            long=long_score,
            short=short_score,
            components={
                "prob_win_long": long_prob if not np.isnan(long_prob) else 0.0,
                "prob_win_short": short_prob if not np.isnan(short_prob) else 0.0,
            },
            metadata={
                "model_path": self.params.model_path,
                "prob_threshold": self.params.prob_threshold,
            },
        )
