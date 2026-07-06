"""Auto ML retraining pipeline.

The current MLScoring model is a snapshot from one backtest — as the
market evolves (new coins, funding regime shifts, changed correlations)
the model's per-trade edge decays. This pipeline periodically retrains
a candidate from fresh data, decides whether to promote it, and records
the decision for audit.

Two flavors of retrain input:

  1. Backtest trades — a completed rabbit backtest run's trades.parquet.
     Deterministic, reproducible, but stops evolving after each run.
  2. Shadow trades — the shadow-mode ledger's closed_trades. Truly
     recent but small and noisy until the runner has accumulated
     hundreds of trades.

Both feed the same `train_model` in ml/training.py. Decision policy is
"promote only if new test AUC beats current + margin"; margin defaults
to 0.01 so a rounding-error win doesn't churn the active model.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from .registry import (
    ModelDescription, _load_training_result, get_active_model,
    list_models, promote,
)
from .training import train_model


@dataclass
class RetrainDecision:
    """What the pipeline decided and why. Written to models/retrain_log.jsonl."""
    timestamp: str
    action: Literal["promote", "reject_no_improvement", "reject_below_min",
                     "reject_error", "no_prior_active"]
    candidate_path: str
    candidate_test_auc: float
    prior_active_path: str | None
    prior_test_auc: float | None
    margin: float
    reason: str


@dataclass
class RetrainConfig:
    """Policy knobs, all with sane defaults."""
    # New AUC must beat prior by at least this to justify promote.
    promote_margin_auc: float = 0.01
    # A candidate below this test AUC is rejected on absolute grounds
    # even if there's no prior — coin-flip models don't get promoted.
    min_test_auc: float = 0.52
    # Fraction of trades used for the walk-forward training split.
    train_fraction: float = 0.7
    model_type: Literal["logistic", "lightgbm"] = "lightgbm"


# ============================================================
# Data ingest
# ============================================================

def load_trades_from_backtest(report_dir: Path) -> pd.DataFrame:
    """Read a backtest report's trades.parquet."""
    p = report_dir / "trades.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no trades.parquet at {p}")
    return pd.read_parquet(p)


def load_trades_from_shadow(state_dir: Path) -> pd.DataFrame:
    """Read shadow ledger.pkl and flatten closed_trades to the same shape
    a backtest trades.parquet has, so `train_model` treats both the same.
    """
    ledger_p = state_dir / "state" / "ledger.pkl"
    if not ledger_p.exists():
        raise FileNotFoundError(f"no ledger at {ledger_p}")
    with ledger_p.open("rb") as f:
        ledger = pickle.load(f)
    if not ledger.closed_trades:
        return pd.DataFrame()
    rows: list[dict] = []
    for t in ledger.closed_trades:
        snap = t.get("entry_snapshot") or {}
        # Rename entry-snapshot features to *_t0 so column names match
        # what training expects.
        row = dict(t)
        row.pop("entry_snapshot", None)
        for k, v in snap.items():
            row[f"{k}_t0"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Decision policy
# ============================================================

def _load_prior_metrics(prior_path: Path | None) -> tuple[Path | None, float | None]:
    if prior_path is None:
        return None, None
    result = _load_training_result(prior_path.parent) or {}
    prior_auc = result.get("test_auc")
    if prior_auc is None:
        return prior_path, None
    return prior_path, float(prior_auc)


def _decide(
    candidate_test_auc: float,
    prior_test_auc: float | None,
    cfg: RetrainConfig,
) -> tuple[bool, str, str]:
    """Return (should_promote, action_string, reason_string)."""
    if candidate_test_auc < cfg.min_test_auc:
        return False, "reject_below_min", (
            f"candidate test_auc={candidate_test_auc:.4f} "
            f"< min_test_auc={cfg.min_test_auc:.4f}"
        )
    if prior_test_auc is None:
        return True, "no_prior_active", (
            f"no prior active model — promoting "
            f"candidate test_auc={candidate_test_auc:.4f}"
        )
    delta = candidate_test_auc - prior_test_auc
    if delta >= cfg.promote_margin_auc:
        return True, "promote", (
            f"candidate {candidate_test_auc:.4f} beats prior "
            f"{prior_test_auc:.4f} by {delta:+.4f} ≥ margin "
            f"{cfg.promote_margin_auc:.4f}"
        )
    return False, "reject_no_improvement", (
        f"candidate {candidate_test_auc:.4f} vs prior "
        f"{prior_test_auc:.4f} → Δ={delta:+.4f} < margin "
        f"{cfg.promote_margin_auc:.4f}"
    )


# ============================================================
# Public entry
# ============================================================

@dataclass
class RetrainOutcome:
    decision: RetrainDecision
    candidate: ModelDescription
    prior: ModelDescription | None = None

    def as_dict(self) -> dict:
        return {
            "decision": asdict(self.decision),
            "candidate_version": self.candidate.version,
            "candidate_test_auc": self.candidate.test_auc,
            "prior_version": self.prior.version if self.prior else None,
            "prior_test_auc": self.prior.test_auc if self.prior else None,
        }


def _append_log(models_root: Path, decision: RetrainDecision) -> None:
    log = models_root / "retrain_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(decision)) + "\n")


def retrain(
    trades: pd.DataFrame,
    models_root: Path,
    ml_config_path: Path,
    cfg: RetrainConfig | None = None,
    prev_marker_path: Path | None = None,
    now_utc: str | None = None,
) -> RetrainOutcome:
    """Train a candidate from `trades`, compare to the active model, and
    promote or reject per policy. Writes an audit entry to
    `models/retrain_log.jsonl` regardless of the outcome — even a rejected
    candidate stays on disk for later inspection.
    """
    cfg = cfg or RetrainConfig()
    ts = now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Discover prior state before training so we have both metrics side-by-side.
    prior_path = get_active_model(ml_config_path)
    prior_path, prior_auc = _load_prior_metrics(prior_path)

    try:
        _, result, candidate_path = train_model(
            trades=trades, output_root=models_root,
            train_fraction=cfg.train_fraction,
            model_type=cfg.model_type,
        )
    except Exception as e:
        decision = RetrainDecision(
            timestamp=ts, action="reject_error",
            candidate_path="", candidate_test_auc=0.0,
            prior_active_path=str(prior_path) if prior_path else None,
            prior_test_auc=prior_auc,
            margin=cfg.promote_margin_auc,
            reason=f"training failed: {type(e).__name__}: {e}",
        )
        _append_log(models_root, decision)
        raise

    should_promote, action, reason = _decide(
        candidate_test_auc=result.test_auc,
        prior_test_auc=prior_auc,
        cfg=cfg,
    )

    if should_promote:
        promote(ml_config_path, candidate_path,
                prev_marker_path=prev_marker_path)

    decision = RetrainDecision(
        timestamp=ts, action=action,
        candidate_path=str(candidate_path),
        candidate_test_auc=result.test_auc,
        prior_active_path=str(prior_path) if prior_path else None,
        prior_test_auc=prior_auc,
        margin=cfg.promote_margin_auc,
        reason=reason,
    )
    _append_log(models_root, decision)

    all_models = {m.path.resolve(): m for m in list_models(models_root)}
    candidate_desc = all_models.get(candidate_path.resolve())
    prior_desc = (all_models.get(prior_path.resolve())
                  if prior_path is not None else None)
    return RetrainOutcome(
        decision=decision,
        candidate=candidate_desc,
        prior=prior_desc,
    )
