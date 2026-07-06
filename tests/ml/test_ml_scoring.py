import pickle
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rabbit_hunter.ml.ml_scoring import MLScoring, MLScoringParams


def _mk_dummy_pipeline():
    """Fit a tiny sklearn pipeline on synthetic data with clear signal."""
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "adx_t0": rng.normal(30, 5, n),
        "rsi_14_t0": rng.normal(50, 10, n),
        "side_long": rng.integers(0, 2, n),
    })
    # Winners have adx > 32
    y = (X["adx_t0"] > 32).astype(int)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    return pipe


def test_ml_scoring_loads_pickled_model(tmp_path):
    pipe = _mk_dummy_pipeline()
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipe, f)

    strat = MLScoring(MLScoringParams(model_path=str(model_path)))
    assert strat.name == "ml_scoring"
    assert "side_long" in strat._feature_names


def test_ml_scoring_returns_zero_for_missing_features(tmp_path):
    pipe = _mk_dummy_pipeline()
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipe, f)

    strat = MLScoring(MLScoringParams(model_path=str(model_path)))
    # Empty features row → NaN prob → score 0
    out = strat.score({}, pd.DataFrame())
    assert out.long == 0.0
    assert out.short == 0.0


def test_ml_scoring_strong_signal_produces_positive_score(tmp_path):
    pipe = _mk_dummy_pipeline()
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipe, f)

    strat = MLScoring(MLScoringParams(model_path=str(model_path), prob_threshold=0.5))
    # Very high adx → model predicts win with high probability
    strong_row = {"adx": 50.0, "rsi_14": 60.0}
    out = strat.score(strong_row, pd.DataFrame())
    # At least one side should have non-zero score
    assert out.long > 0.0 or out.short > 0.0


def test_ml_scoring_reads_both_suffixed_and_unsuffixed_columns(tmp_path):
    """The model was trained with `_t0` suffixed columns (from trades.parquet),
    but live scoring feeds features WITHOUT the suffix (Feature Engine output).
    MLScoring must gracefully handle both."""
    pipe = _mk_dummy_pipeline()
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(pipe, f)

    strat = MLScoring(MLScoringParams(model_path=str(model_path)))
    # Feed without _t0 suffix — should still work
    row = {"adx": 40.0, "rsi_14": 55.0}
    out = strat.score(row, pd.DataFrame())
    # Should NOT crash; probs should be finite
    assert not np.isnan(out.components["prob_win_long"])
    assert not np.isnan(out.components["prob_win_short"])


def test_ml_scoring_missing_model_file_raises():
    with pytest.raises(FileNotFoundError):
        MLScoring(MLScoringParams(model_path="/nonexistent/model.pkl"))
