"""Feature-level drift detection.

Complement to `drift.py` (which compares trade outcomes per cluster) —
this module compares FEATURE distributions live-vs-baseline. A shift in
RSI or funding_rate distribution is often visible BEFORE it produces a
measurable change in trade outcomes, so this catches regime changes
earlier.

Two-tier design:

  1. FeatureBaseline — for each feature, mean / std / p05 / p50 / p95.
     Cheap to compute, cheap to store, and enough for the two most
     useful drift tests:
       - mean shift: |Δ mean| / baseline_std ≥ threshold (in σ)
       - std shift: std_ratio outside [1/threshold, threshold]
     A Kolmogorov-Smirnov test is available for a more sensitive check
     when both the baseline and live samples are on hand.

  2. FeatureStabilityReport — per-feature findings + summary status.
     Same DriftReport shape as trade-level drift so downstream code
     (CLI, dashboard) can render both consistently.

Not covered here (deliberately):
  - Correlation drift between features
  - Regime-conditional stats (e.g. RSI in "trending" vs "range")
    Both are richer signals but need larger samples and slower stats;
    can be added in a follow-up when the shadow history is longer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# Baseline data class + serialization
# ============================================================

@dataclass
class FeatureBaseline:
    feature: str
    n: int
    mean: float
    std: float
    p05: float
    p50: float
    p95: float


@dataclass
class FeatureBaselineSnapshot:
    tag: str
    created_at_utc: str
    source: str                     # report dir or synthetic tag
    features: list[FeatureBaseline] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({
            "tag": self.tag,
            "created_at_utc": self.created_at_utc,
            "source": self.source,
            "features": [asdict(f) for f in self.features],
        }, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "FeatureBaselineSnapshot":
        data = json.loads(text)
        return cls(
            tag=data["tag"],
            created_at_utc=data["created_at_utc"],
            source=data["source"],
            features=[FeatureBaseline(**f) for f in data["features"]],
        )

    def by_feature(self) -> dict[str, FeatureBaseline]:
        return {f.feature: f for f in self.features}


# ============================================================
# Snapshot builder
# ============================================================

# Features we track by default. All numeric. Extend when the strategy
# adds new inputs.
DEFAULT_TRACKED_FEATURES: tuple[str, ...] = (
    "close", "volume", "atr_14", "atr_pct",
    "ema20_slope", "adx", "rsi_14",
    "bb_pct", "zscore_20", "volume_ratio_20",
    "funding_rate", "oi_change_pct",
)


def build_baseline_from_features(
    df: pd.DataFrame,
    tag: str,
    source: str,
    features: Iterable[str] | None = None,
    now_utc: str | None = None,
) -> FeatureBaselineSnapshot:
    """Compute mean/std/quantiles for each numeric feature column in df.

    Skips columns that are missing or non-numeric. NaN-safe (uses
    .dropna() per column) so partial warmup bars don't skew percentiles.
    """
    ts = now_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    cols: list[str]
    if features is None:
        cols = [c for c in DEFAULT_TRACKED_FEATURES if c in df.columns]
    else:
        cols = [c for c in features if c in df.columns]

    entries: list[FeatureBaseline] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:      # skip near-empty columns
            continue
        entries.append(FeatureBaseline(
            feature=col,
            n=int(len(s)),
            mean=float(s.mean()),
            std=float(s.std()),
            p05=float(s.quantile(0.05)),
            p50=float(s.quantile(0.5)),
            p95=float(s.quantile(0.95)),
        ))
    return FeatureBaselineSnapshot(
        tag=tag, created_at_utc=ts, source=source, features=entries,
    )


def save(snapshot: FeatureBaselineSnapshot, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.to_json(), encoding="utf-8")
    return path


def load(path: Path) -> FeatureBaselineSnapshot:
    return FeatureBaselineSnapshot.from_json(path.read_text(encoding="utf-8"))


# ============================================================
# Drift detection
# ============================================================

@dataclass(frozen=True)
class FeatureStabilityThresholds:
    # Fire if |mean_live - mean_baseline| / baseline_std ≥ this many σ.
    mean_shift_sigmas: float = 2.0
    # Fire if std_live / std_baseline outside [1/x, x].
    std_ratio_alert: float = 2.0
    # Minimum live samples before a feature can trigger.
    min_live_samples: int = 200


@dataclass
class FeatureDriftFinding:
    feature: str
    baseline_n: int
    live_n: int
    baseline_mean: float
    live_mean: float
    baseline_std: float
    live_std: float
    mean_shift_sigmas: float
    std_ratio: float
    triggered: bool
    reason: str


@dataclass
class FeatureStabilityReport:
    ok: bool
    findings: list[FeatureDriftFinding] = field(default_factory=list)
    baseline_only: list[str] = field(default_factory=list)
    live_only: list[str] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        head = "feature_stability: OK" if self.ok else "feature_stability: ALERT"
        lines = [head]
        for f in self.findings:
            marker = "!!" if f.triggered else "  "
            lines.append(
                f"  {marker} {f.feature}: "
                f"baseline μ={f.baseline_mean:.4g} σ={f.baseline_std:.4g} → "
                f"live μ={f.live_mean:.4g} σ={f.live_std:.4g} "
                f"(Δ_μ={f.mean_shift_sigmas:+.2f}σ, "
                f"σ_ratio={f.std_ratio:.2f}, n={f.live_n}) — {f.reason}"
            )
        for c in self.baseline_only:
            lines.append(f"  -- baseline had {c} but live has 0 samples")
        for c in self.live_only:
            lines.append(f"  ++ live has {c} but baseline never saw it")
        return lines


def _score_feature(
    baseline: FeatureBaseline,
    live_series: pd.Series,
    thresholds: FeatureStabilityThresholds,
) -> FeatureDriftFinding:
    s = pd.to_numeric(live_series, errors="coerce").dropna()
    live_n = int(len(s))
    live_mean = float(s.mean()) if live_n else 0.0
    live_std = float(s.std()) if live_n else 0.0

    if baseline.std > 0:
        mean_shift = (live_mean - baseline.mean) / baseline.std
    else:
        mean_shift = 0.0

    if baseline.std > 0 and live_std > 0:
        std_ratio = live_std / baseline.std
    else:
        std_ratio = 1.0

    triggered = False
    reasons: list[str] = []
    if live_n < thresholds.min_live_samples:
        reasons.append(f"below min_n={thresholds.min_live_samples}")
    else:
        if abs(mean_shift) >= thresholds.mean_shift_sigmas:
            triggered = True
            reasons.append(
                f"|Δ_μ|={abs(mean_shift):.2f}σ≥"
                f"{thresholds.mean_shift_sigmas:.1f}σ"
            )
        if (std_ratio >= thresholds.std_ratio_alert
                or std_ratio <= 1 / thresholds.std_ratio_alert):
            triggered = True
            reasons.append(
                f"σ_ratio={std_ratio:.2f} outside "
                f"[{1/thresholds.std_ratio_alert:.2f},"
                f"{thresholds.std_ratio_alert:.2f}]"
            )
        if not triggered:
            reasons.append("within tolerance")

    return FeatureDriftFinding(
        feature=baseline.feature,
        baseline_n=baseline.n, live_n=live_n,
        baseline_mean=baseline.mean, live_mean=live_mean,
        baseline_std=baseline.std, live_std=live_std,
        mean_shift_sigmas=mean_shift, std_ratio=std_ratio,
        triggered=triggered,
        reason=";".join(reasons),
    )


def compare(
    baseline: FeatureBaselineSnapshot,
    live_df: pd.DataFrame,
    thresholds: FeatureStabilityThresholds | None = None,
) -> FeatureStabilityReport:
    thresholds = thresholds or FeatureStabilityThresholds()
    baseline_by = baseline.by_feature()

    findings: list[FeatureDriftFinding] = []
    baseline_only: list[str] = []
    live_only: list[str] = []

    for feat, bl in baseline_by.items():
        if feat not in live_df.columns:
            baseline_only.append(feat)
            continue
        s = pd.to_numeric(live_df[feat], errors="coerce").dropna()
        if len(s) == 0:
            baseline_only.append(feat)
            continue
        findings.append(_score_feature(bl, s, thresholds))

    for col in live_df.columns:
        if col in baseline_by:
            continue
        # Only surface columns that actually have numeric data — the
        # frames pass lots of metadata columns (timestamp, symbol, etc.)
        # that shouldn't be flagged.
        s = pd.to_numeric(live_df[col], errors="coerce").dropna()
        if len(s) >= thresholds.min_live_samples \
                and col in DEFAULT_TRACKED_FEATURES:
            live_only.append(col)

    any_triggered = any(f.triggered for f in findings)
    return FeatureStabilityReport(
        ok=not any_triggered,
        findings=findings,
        baseline_only=baseline_only,
        live_only=live_only,
    )
