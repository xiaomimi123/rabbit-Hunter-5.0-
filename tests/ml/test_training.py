import json
import pandas as pd
import numpy as np
import pytest
from rabbit_hunter.ml.training import build_training_set, train_model, _DEFAULT_FEATURES


def _mk_trades(n=200, seed=42):
    """Synthesize trades with signal: winners have higher adx_t0 and rsi_14_t0.

    Trainer should be able to find the pattern → test AUC > 0.55."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        is_winner = rng.random() < 0.5
        rows.append({
            "entry_time": i * 3_600_000,
            "symbol": "BTC-USDT-SWAP" if i % 2 == 0 else "ETH-USDT-SWAP",
            "side": "long" if i % 3 == 0 else "short",
            # These are the signal columns — winners have higher values
            "ema20_t0":   rng.normal(100 + 5 * is_winner, 2),
            "ema60_t0":   rng.normal(100 + 3 * is_winner, 2),
            "ema200_t0":  rng.normal(100, 2),
            "ema20_slope_t0": rng.normal(0.5 * is_winner, 0.1),
            "adx_t0":     rng.normal(30 + 8 * is_winner, 3),
            "di_plus_t0": rng.normal(25, 3),
            "di_minus_t0": rng.normal(20, 3),
            "rsi_14_t0":  rng.normal(50 + 10 * is_winner, 5),
            "bb_width_t0": rng.normal(0.02, 0.005),
            "bb_pct_t0":  rng.normal(0.5, 0.15),
            "zscore_20_t0": rng.normal(0, 1),
            "atr_pct_t0": rng.normal(0.015, 0.003),
            "volume_ratio_20_t0": rng.normal(1.2, 0.2),
            "funding_rate_t0": rng.normal(0, 0.0001),
            "oi_change_pct_t0": rng.normal(0, 0.01),
            "ema20_1h_on_15m_t0": rng.normal(100 + 5 * is_winner, 2),
            "adx_1h_on_15m_t0": rng.normal(28 + 8 * is_winner, 3),
            "atr_pct_baseline_t0": rng.normal(0.014, 0.002),
            "pnl_after_fees": 10.0 if is_winner else -8.0,
        })
    return pd.DataFrame(rows)


def test_build_training_set_shape():
    trades = _mk_trades(50)
    X, y = build_training_set(trades)
    assert len(X) == len(y) == 50
    # Should include most default features + side_long
    assert "side_long" in X.columns
    assert "adx_t0" in X.columns
    assert y.isin([0, 1]).all()


def test_build_training_set_drops_nan_rows():
    trades = _mk_trades(20)
    trades.loc[5, "adx_t0"] = np.nan
    X, y = build_training_set(trades)
    # Row 5 should be dropped
    assert len(X) == 19


def test_build_training_set_requires_min_features():
    """Small trades DataFrame with only unrelated columns → error."""
    trades = pd.DataFrame([
        {"side": "long", "pnl_after_fees": 10.0, "weird_col": 1.0}
        for _ in range(20)
    ])
    with pytest.raises(ValueError, match="Not enough usable feature columns"):
        build_training_set(trades)


def test_train_model_produces_versioned_artifact(tmp_path):
    trades = _mk_trades(200)
    pipe, result, model_path = train_model(trades, output_root=tmp_path)

    assert model_path.exists()
    assert model_path.name == "model.pkl"
    assert (model_path.parent / "training_result.json").exists()
    assert (model_path.parent / "README.md").exists()

    # AUC should be well above chance because our synthetic data has signal
    assert result.test_auc > 0.55, f"Test AUC only {result.test_auc:.3f}"
    assert result.train_auc > 0.6

    # JSON should round-trip
    d = json.loads((model_path.parent / "training_result.json").read_text())
    assert d["model_version"] == result.model_version
    assert d["features_used"] == result.features_used


def test_train_model_rejects_tiny_dataset(tmp_path):
    trades = _mk_trades(10)
    with pytest.raises(ValueError, match="need at least 50"):
        train_model(trades, output_root=tmp_path)


def test_train_model_rejects_single_class(tmp_path):
    """All-winners or all-losers dataset → cannot train."""
    trades = _mk_trades(200)
    trades["pnl_after_fees"] = 10.0  # all winners
    with pytest.raises(ValueError, match="single-class"):
        train_model(trades, output_root=tmp_path)


def test_train_model_with_lightgbm(tmp_path):
    """LightGBM path handles non-linearities → test AUC should exceed
    logistic baseline on the same signal-heavy synthetic data."""
    pytest.importorskip("lightgbm")
    trades = _mk_trades(300)
    pipe, result, model_path = train_model(
        trades=trades, output_root=tmp_path, model_type="lightgbm"
    )
    assert model_path.exists()
    assert result.hyperparameters["model"] == "LGBMClassifier"
    # With clear synthetic signal, LightGBM should beat baseline 0.5
    assert result.test_auc > 0.55, f"LightGBM test AUC only {result.test_auc:.3f}"


def test_train_model_unknown_type_falls_back_to_logistic(tmp_path):
    """Unknown model_type argument silently falls back to the logistic
    baseline (safe default). If we want strict rejection later, adjust
    training.py to raise on unknown types."""
    trades = _mk_trades(100)
    _, result, _ = train_model(trades, output_root=tmp_path, model_type="unknown")
    assert result.hyperparameters["model"] == "LogisticRegression"
