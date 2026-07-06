"""Tests for the auto-retraining pipeline decision policy + logging.

Each test uses a synthetic training set (large enough to satisfy
train_model's 50-trade minimum). We stub train_model where the decision
policy is the point being tested and use real train_model where the
end-to-end wiring is the point.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from rabbit_hunter.ml.registry import get_active_model
from rabbit_hunter.ml.retraining import (
    RetrainConfig, _decide, load_trades_from_backtest,
    load_trades_from_shadow, retrain,
)


def _mk_trades(n: int = 200, seed: int = 0,
               win_rate: float = 0.55) -> pd.DataFrame:
    """Synthetic training set — a mix of features that has some signal
    (feature_a positively correlates with winning), so train_model
    produces a non-trivial model."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    # Base rate from win_rate + signal-driven perturbation
    base = np.where(rng.random(n) < win_rate, 1, 0)
    pnl = np.where(base == 1, 20 + signal * 5, -10 - noise * 3)
    return pd.DataFrame({
        "side": np.where(rng.random(n) < 0.5, "long", "short"),
        "pnl_after_fees": pnl,
        "entry_time": np.arange(n) * 3_600_000,
        # Feature columns — all 18 default features
        "ema20_t0":       100 + signal,
        "ema60_t0":       100 + signal * 0.8,
        "ema200_t0":      100 + signal * 0.5,
        "ema20_slope_t0": signal * 0.1,
        "adx_t0":         25 + signal * 5,
        "di_plus_t0":     20 + noise,
        "di_minus_t0":    20 - noise,
        "rsi_14_t0":      50 + signal * 10,
        "bb_width_t0":    0.04 + noise * 0.001,
        "bb_pct_t0":      0.5 + signal * 0.1,
        "zscore_20_t0":   signal,
        "atr_pct_t0":     0.01 + noise * 0.001,
        "volume_ratio_20_t0": 1.0 + noise * 0.1,
        "funding_rate_t0":   noise * 0.0001,
        "oi_change_pct_t0":  noise * 0.01,
        "ema20_1h_on_15m_t0": 100 + signal * 0.9,
        "adx_1h_on_15m_t0":   25 + signal * 4,
        "atr_pct_baseline_t0": 0.01 + noise * 0.0005,
    })


def _mk_ml_config(root: Path, model_path: Path | None = None) -> Path:
    d = root / "configs" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "ml_scoring.yaml"
    data: dict = {"name": "ml_scoring", "version": "0.1.0", "params": {}}
    if model_path is not None:
        data["params"]["model_path"] = str(model_path)
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


# ============================================================
# Decision policy
# ============================================================

def test_decide_promote_when_beats_margin():
    ok, action, _ = _decide(candidate_test_auc=0.60, prior_test_auc=0.55,
                              cfg=RetrainConfig(promote_margin_auc=0.02))
    assert ok is True
    assert action == "promote"


def test_decide_reject_when_delta_below_margin():
    ok, action, _ = _decide(candidate_test_auc=0.56, prior_test_auc=0.55,
                              cfg=RetrainConfig(promote_margin_auc=0.02))
    assert ok is False
    assert action == "reject_no_improvement"


def test_decide_reject_when_below_absolute_min():
    ok, action, _ = _decide(candidate_test_auc=0.50, prior_test_auc=0.45,
                              cfg=RetrainConfig(min_test_auc=0.52))
    assert ok is False
    assert action == "reject_below_min"


def test_decide_promote_when_no_prior_and_above_min():
    ok, action, _ = _decide(candidate_test_auc=0.55, prior_test_auc=None,
                              cfg=RetrainConfig(min_test_auc=0.52))
    assert ok is True
    assert action == "no_prior_active"


def test_decide_reject_when_no_prior_but_below_min():
    """min_test_auc is an absolute floor — no prior doesn't waive it."""
    ok, action, _ = _decide(candidate_test_auc=0.50, prior_test_auc=None,
                              cfg=RetrainConfig(min_test_auc=0.52))
    assert ok is False
    assert action == "reject_below_min"


def test_decide_delta_just_over_margin_promotes():
    """A comfortably-over-margin delta must promote. Uses integer-decimal
    values to sidestep binary float precision (0.57 − 0.55 in IEEE 754
    is 0.01999... < 0.02, which is not what this test is about)."""
    ok, action, _ = _decide(candidate_test_auc=0.65, prior_test_auc=0.55,
                              cfg=RetrainConfig(promote_margin_auc=0.02))
    assert ok is True


# ============================================================
# End-to-end retrain
# ============================================================

def test_retrain_no_prior_promotes_new_model(tmp_path):
    cfg = _mk_ml_config(tmp_path, None)   # no prior active
    models_root = tmp_path / "models"
    trades = _mk_trades(n=200)
    outcome = retrain(
        trades=trades,
        models_root=models_root,
        ml_config_path=cfg,
        cfg=RetrainConfig(min_test_auc=0.0),   # accept any AUC for the test
        prev_marker_path=tmp_path / "marker",
    )
    assert outcome.decision.action == "no_prior_active"
    assert get_active_model(cfg) is not None
    # Retrain log has one entry
    log = (models_root / "retrain_log.jsonl").read_text().splitlines()
    assert len(log) == 1
    entry = json.loads(log[0])
    assert entry["action"] == "no_prior_active"


def test_retrain_rejects_when_no_improvement(tmp_path):
    """Prior test AUC set artificially high so the candidate can't beat it."""
    prior_dir = tmp_path / "models" / "ml_model_vPRIOR"
    prior_dir.mkdir(parents=True)
    (prior_dir / "model.pkl").write_bytes(b"fake")
    (prior_dir / "training_result.json").write_text(json.dumps({
        "model_version": "PRIOR",
        "test_auc": 0.90,   # unbeatably good
        "train_auc": 0.95, "train_accuracy": 0.9, "test_accuracy": 0.85,
        "n_train": 100, "n_test": 30, "features_used": [],
        "hyperparameters": {},
        "trained_at": "2026-01-01T00:00:00Z",
    }), encoding="utf-8")
    cfg = _mk_ml_config(tmp_path, prior_dir / "model.pkl")
    outcome = retrain(
        trades=_mk_trades(n=200),
        models_root=tmp_path / "models",
        ml_config_path=cfg,
        cfg=RetrainConfig(min_test_auc=0.0, promote_margin_auc=0.01),
        prev_marker_path=tmp_path / "marker",
    )
    assert outcome.decision.action == "reject_no_improvement"
    # Config still points at prior
    assert get_active_model(cfg) == prior_dir / "model.pkl"


def test_retrain_appends_to_log_even_on_rejection(tmp_path):
    prior_dir = tmp_path / "models" / "ml_model_vPRIOR"
    prior_dir.mkdir(parents=True)
    (prior_dir / "model.pkl").write_bytes(b"fake")
    (prior_dir / "training_result.json").write_text(json.dumps({
        "model_version": "PRIOR", "test_auc": 0.99,
        "train_auc": 0.99, "train_accuracy": 1.0, "test_accuracy": 0.9,
        "n_train": 100, "n_test": 30, "features_used": [],
        "hyperparameters": {},
        "trained_at": "2026-01-01T00:00:00Z",
    }), encoding="utf-8")
    cfg = _mk_ml_config(tmp_path, prior_dir / "model.pkl")
    retrain(
        trades=_mk_trades(n=200),
        models_root=tmp_path / "models",
        ml_config_path=cfg,
        cfg=RetrainConfig(min_test_auc=0.0),
        prev_marker_path=tmp_path / "marker",
    )
    log_path = tmp_path / "models" / "retrain_log.jsonl"
    assert log_path.exists()
    entries = [json.loads(l) for l in log_path.read_text().splitlines()]
    assert entries[0]["action"] == "reject_no_improvement"


def test_retrain_error_logged_but_raised(tmp_path):
    """A training crash writes the reject_error entry BEFORE re-raising
    so we don't lose the audit trail."""
    cfg = _mk_ml_config(tmp_path, None)
    with patch("rabbit_hunter.ml.retraining.train_model",
               side_effect=RuntimeError("bad data")):
        with pytest.raises(RuntimeError):
            retrain(
                trades=_mk_trades(n=200),
                models_root=tmp_path / "models",
                ml_config_path=cfg,
                cfg=RetrainConfig(),
                prev_marker_path=tmp_path / "marker",
            )
    log_path = tmp_path / "models" / "retrain_log.jsonl"
    assert log_path.exists()
    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["action"] == "reject_error"
    assert "bad data" in entry["reason"]


# ============================================================
# Input loaders
# ============================================================

def test_load_trades_from_backtest_reads_parquet(tmp_path):
    report = tmp_path / "reports" / "run1"
    report.mkdir(parents=True)
    df = _mk_trades(n=10)
    df.to_parquet(report / "trades.parquet")
    loaded = load_trades_from_backtest(report)
    assert len(loaded) == 10


def test_load_trades_from_backtest_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trades_from_backtest(tmp_path / "no")


def test_load_trades_from_shadow_flattens_entry_snapshot(tmp_path):
    """Shadow closed_trades wrap features in `entry_snapshot`; the
    loader must lift them to the *_t0 columns that train_model expects."""
    from rabbit_hunter.backtest.ledger import Ledger
    state = tmp_path / "state"
    state.mkdir()
    ledger = Ledger(initial_capital=10_000.0)
    ledger.closed_trades = [
        {"symbol": "BTC-USDT-SWAP", "side": "short",
         "pnl_after_fees": +100.0, "entry_time": 1_700_000_000_000,
         "exit_time": 1_700_003_600_000,
         "entry_snapshot": {
             "ema20": 100.0, "ema60": 99.0, "ema200": 98.0,
             "adx": 30.0, "rsi_14": 25.0, "bb_pct": 0.5,
             "zscore_20": -2.0,
         }},
    ]
    with (state / "ledger.pkl").open("wb") as f:
        pickle.dump(ledger, f)
    df = load_trades_from_shadow(tmp_path)
    assert "rsi_14_t0" in df.columns
    assert df["rsi_14_t0"].iloc[0] == 25.0
    assert "entry_snapshot" not in df.columns


def test_load_trades_from_shadow_returns_empty_when_ledger_empty(tmp_path):
    from rabbit_hunter.backtest.ledger import Ledger
    state = tmp_path / "state"
    state.mkdir()
    with (state / "ledger.pkl").open("wb") as f:
        pickle.dump(Ledger(initial_capital=10_000.0), f)
    df = load_trades_from_shadow(tmp_path)
    assert df.empty
