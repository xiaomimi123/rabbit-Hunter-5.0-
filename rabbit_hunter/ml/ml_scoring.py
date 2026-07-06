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

    def _predict_prob(self, features_row: dict, side: str) -> float:
        """Return prob(win) for the given side, or NaN if features missing."""
        row = {}
        for feat in self._feature_names:
            if feat == "side_long":
                row[feat] = 1 if side == "long" else 0
            else:
                # trades.parquet columns are _t0-suffixed. The feature engine
                # writes them without the suffix during live scoring. Try both.
                stripped = feat[:-3] if feat.endswith("_t0") else feat
                v = features_row.get(stripped, features_row.get(feat))
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return float("nan")
                row[feat] = float(v)
        X = pd.DataFrame([row])[self._feature_names]
        prob = self._pipe.predict_proba(X)[0, 1]
        return float(prob)

    def _prob_to_score(self, prob: float) -> float:
        if np.isnan(prob):
            return 0.0
        if prob < self.params.prob_threshold:
            return 0.0
        # Rescale [threshold, 1.0] → [0.0, 1.0]
        thr = self.params.prob_threshold
        return (prob - thr) / max(1.0 - thr, 1e-9)

    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput:
        long_prob = self._predict_prob(features_row, "long")
        short_prob = self._predict_prob(features_row, "short")

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
