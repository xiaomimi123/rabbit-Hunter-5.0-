"""Tests for the feature-stability analyzer.

Uses synthetic frames sampled from known distributions so drift alerts
are deterministic — no hidden reliance on real market data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rabbit_hunter.analytics.feature_stability import (
    DEFAULT_TRACKED_FEATURES,
    FeatureBaseline, FeatureBaselineSnapshot,
    FeatureStabilityThresholds,
    build_baseline_from_features, compare, load, save,
)


def _feat_df(n: int, **overrides) -> pd.DataFrame:
    """Build a synthetic feature frame of length n with tracked feature
    columns filled from an N(0, 1) sample. Overrides let a test control
    a specific column."""
    rng = np.random.default_rng(seed=42)
    cols = {
        "close": rng.normal(100, 5, n),
        "volume": rng.normal(50, 5, n),
        "atr_14": rng.normal(1.0, 0.1, n),
        "atr_pct": rng.normal(0.01, 0.002, n),
        "ema20_slope": rng.normal(0, 0.1, n),
        "adx": rng.normal(25, 5, n),
        "rsi_14": rng.normal(50, 10, n),
        "bb_pct": rng.normal(0.5, 0.1, n),
        "zscore_20": rng.normal(0, 1, n),
        "volume_ratio_20": rng.normal(1.0, 0.2, n),
        "funding_rate": rng.normal(0.0001, 0.00005, n),
        "oi_change_pct": rng.normal(0, 0.01, n),
    }
    cols.update(overrides)
    return pd.DataFrame(cols)


# ============================================================
# Baseline building
# ============================================================

def test_baseline_builds_from_tracked_features_only():
    df = _feat_df(500)
    df["extra_col"] = 1.0    # not in DEFAULT_TRACKED_FEATURES
    snap = build_baseline_from_features(df, tag="v1", source="synthetic")
    names = {f.feature for f in snap.features}
    assert names <= set(DEFAULT_TRACKED_FEATURES)
    assert "extra_col" not in names


def test_baseline_skips_near_empty_columns():
    df = _feat_df(500)
    df["adx"] = float("nan")    # all NaN → dropped
    snap = build_baseline_from_features(df, tag="v1", source="synthetic")
    names = {f.feature for f in snap.features}
    assert "adx" not in names


def test_baseline_computes_mean_std_and_quantiles():
    """A known constant column produces mean=const, std≈0, and
    percentile bands = const."""
    df = _feat_df(500)
    df["rsi_14"] = 55.0
    snap = build_baseline_from_features(df, tag="v1", source="synthetic")
    fb = snap.by_feature()["rsi_14"]
    assert fb.mean == pytest.approx(55.0)
    assert fb.std == pytest.approx(0.0, abs=1e-9)
    assert fb.p05 == pytest.approx(55.0)


def test_baseline_json_roundtrip(tmp_path: Path):
    snap = build_baseline_from_features(_feat_df(500), tag="v1",
                                          source="synthetic")
    p = save(snap, tmp_path / "features.json")
    loaded = load(p)
    assert loaded.tag == "v1"
    assert len(loaded.features) == len(snap.features)


# ============================================================
# Compare — happy path
# ============================================================

def test_ok_when_live_matches_baseline():
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    live = _feat_df(500)   # same synthetic seed produces similar distribution
    # Use loose thresholds since we're re-sampling
    r = compare(baseline, live,
                 FeatureStabilityThresholds(mean_shift_sigmas=3.0,
                                              std_ratio_alert=3.0,
                                              min_live_samples=200))
    assert r.ok is True


def test_no_alert_below_min_samples():
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    # 100 rows < min_live_samples=200 → no alert even if far off
    live = _feat_df(100)
    live["rsi_14"] = 90.0
    r = compare(baseline, live,
                 FeatureStabilityThresholds(mean_shift_sigmas=2.0,
                                              min_live_samples=200))
    assert r.ok is True
    assert all(not f.triggered for f in r.findings)


# ============================================================
# Alerts — mean shift
# ============================================================

def test_alert_on_mean_shift():
    """rsi_14 baseline is N(50, 10). Setting live rsi_14 all to 80
    is (80-50)/10 = 3σ off — should alert at 2σ threshold."""
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    live = _feat_df(500)
    live["rsi_14"] = 80.0
    r = compare(baseline, live,
                 FeatureStabilityThresholds(mean_shift_sigmas=2.0,
                                              std_ratio_alert=100.0,
                                              min_live_samples=200))
    finding = next(f for f in r.findings if f.feature == "rsi_14")
    assert finding.triggered
    assert finding.mean_shift_sigmas > 2.5
    assert not r.ok


def test_no_alert_when_mean_shift_below_threshold():
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    live = _feat_df(500)
    live["rsi_14"] = 55.0    # ~0.5σ off
    r = compare(baseline, live,
                 FeatureStabilityThresholds(mean_shift_sigmas=2.0,
                                              std_ratio_alert=100.0,
                                              min_live_samples=200))
    finding = next(f for f in r.findings if f.feature == "rsi_14")
    assert not finding.triggered


# ============================================================
# Alerts — std ratio
# ============================================================

def test_alert_on_std_ratio_out_of_band():
    """Baseline rsi_14 std ~10. Live rsi_14 with std=30 → σ_ratio~3.0
    → alerts at 2.0 threshold."""
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    rng = np.random.default_rng(0)
    live = _feat_df(500)
    # Force wider spread but same mean
    live["rsi_14"] = rng.normal(50, 30, 500)
    r = compare(baseline, live,
                 FeatureStabilityThresholds(mean_shift_sigmas=10.0,  # disable
                                              std_ratio_alert=2.0,
                                              min_live_samples=200))
    finding = next(f for f in r.findings if f.feature == "rsi_14")
    assert finding.triggered
    assert finding.std_ratio >= 2.0


# ============================================================
# Missing features
# ============================================================

def test_baseline_only_flagged_when_feature_missing_from_live():
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    live = _feat_df(500).drop(columns=["rsi_14"])
    r = compare(baseline, live)
    assert "rsi_14" in r.baseline_only


def test_live_only_flagged_for_new_tracked_feature():
    """A feature the baseline never saw but that's in
    DEFAULT_TRACKED_FEATURES and has enough live data → live_only."""
    # Build baseline WITHOUT rsi_14
    df = _feat_df(2000).drop(columns=["rsi_14"])
    baseline = build_baseline_from_features(df, tag="bl", source="s")
    live = _feat_df(500)
    r = compare(baseline, live,
                 FeatureStabilityThresholds(min_live_samples=200))
    assert "rsi_14" in r.live_only


# ============================================================
# as_lines() logging
# ============================================================

def test_as_lines_ok_first_line():
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    live = _feat_df(500)
    lines = compare(baseline, live,
                     FeatureStabilityThresholds(mean_shift_sigmas=3.0,
                                                  std_ratio_alert=3.0,
                                                  min_live_samples=200)) \
        .as_lines()
    assert lines[0].startswith("feature_stability: OK")


def test_as_lines_alert_when_triggered():
    baseline = build_baseline_from_features(_feat_df(2000), tag="bl",
                                              source="s")
    live = _feat_df(500)
    live["rsi_14"] = 80.0
    lines = compare(baseline, live,
                     FeatureStabilityThresholds(mean_shift_sigmas=2.0,
                                                  std_ratio_alert=100.0,
                                                  min_live_samples=200)) \
        .as_lines()
    assert lines[0].startswith("feature_stability: ALERT")
    assert any("!!" in ln for ln in lines[1:])
