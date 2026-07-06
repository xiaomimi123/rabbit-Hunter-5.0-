"""Phase 2 · § 3.2 — Trainable scoring model.

Strict separation between training (offline) and inference (production):
  - Training: `training.py` reads trades.parquet, builds (X, y), trains
    a versioned model, writes .pkl to models/
  - Inference: `MLScoring` strategy loads the pickle and scores like any
    other BaseStrategy — no online learning, no drift risk

Follows architecture spec § 4.1: model versioned, verifiable, replayable.
No AI in the decision path.
"""
from .training import build_training_set, train_model, TrainingResult
from .ml_scoring import MLScoring, MLScoringParams

__all__ = [
    "build_training_set", "train_model", "TrainingResult",
    "MLScoring", "MLScoringParams",
]
