"""Tests for the shadow watchdog.

Watchdogs must be right at every branch — a false HEALTHY hides a dead
runner, a false DOWN pages someone unnecessarily. Every status/threshold
boundary is pinned.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rabbit_hunter.shadow.watchdog import check, WatchdogResult


NOW_MS = 1_700_000_000_000  # fixed reference for determinism


def _seed_metrics(state_dir: Path, last_tick_ts_ms: int) -> None:
    (state_dir / "state").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp_ms": [
        last_tick_ts_ms - 60_000,
        last_tick_ts_ms - 30_000,
        last_tick_ts_ms,
    ]}).to_parquet(state_dir / "state" / "metrics_history.parquet")


# ============================================================
# DOWN
# ============================================================

def test_down_when_state_dir_missing(tmp_path):
    r = check(tmp_path / "no-such", max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "down"
    assert not r.ok
    assert "no metrics_history" in r.reason


def test_down_when_metrics_history_missing_but_state_exists(tmp_path):
    (tmp_path / "state").mkdir()
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "down"


def test_down_when_metrics_history_corrupt(tmp_path):
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "metrics_history.parquet").write_bytes(b"not parquet")
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "down"
    assert "unreadable" in r.reason


def test_down_when_metrics_history_empty(tmp_path):
    (tmp_path / "state").mkdir()
    pd.DataFrame({"timestamp_ms": []}).to_parquet(
        tmp_path / "state" / "metrics_history.parquet"
    )
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "down"
    assert "empty" in r.reason


# ============================================================
# HEALTHY / STALE — threshold boundary
# ============================================================

def test_healthy_when_last_tick_just_now(tmp_path):
    _seed_metrics(tmp_path, last_tick_ts_ms=NOW_MS)
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "healthy"
    assert r.ok
    assert r.seconds_since_last_tick == 0.0


def test_healthy_exactly_at_threshold(tmp_path):
    _seed_metrics(tmp_path, last_tick_ts_ms=NOW_MS - 300_000)
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    # inclusive boundary
    assert r.status == "healthy"
    assert r.seconds_since_last_tick == 300.0


def test_stale_when_last_tick_beyond_threshold(tmp_path):
    _seed_metrics(tmp_path, last_tick_ts_ms=NOW_MS - 600_000)  # 10 min
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "stale"
    assert not r.ok
    assert r.seconds_since_last_tick == 600.0
    assert "600s ago" in r.reason


# ============================================================
# Uses the LATEST timestamp, not the first
# ============================================================

def test_uses_max_timestamp_not_first(tmp_path):
    # Write out-of-order rows; watchdog must find the latest one
    (tmp_path / "state").mkdir(parents=True)
    pd.DataFrame({"timestamp_ms": [
        NOW_MS - 3600_000,
        NOW_MS - 30_000,       # latest
        NOW_MS - 1800_000,
    ]}).to_parquet(tmp_path / "state" / "metrics_history.parquet")
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    assert r.status == "healthy"
    assert r.seconds_since_last_tick == 30.0


# ============================================================
# as_line() strings — used by CLI
# ============================================================

def test_as_line_healthy_message(tmp_path):
    _seed_metrics(tmp_path, last_tick_ts_ms=NOW_MS - 60_000)
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    line = r.as_line()
    assert "OK" in line
    assert "60s ago" in line


def test_as_line_stale_message(tmp_path):
    _seed_metrics(tmp_path, last_tick_ts_ms=NOW_MS - 900_000)
    r = check(tmp_path, max_silence_seconds=300, now_ms=NOW_MS)
    line = r.as_line()
    assert "STALE" in line


def test_as_line_down_message(tmp_path):
    r = check(tmp_path / "missing", max_silence_seconds=300, now_ms=NOW_MS)
    line = r.as_line()
    assert "DOWN" in line
