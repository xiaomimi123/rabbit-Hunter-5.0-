"""Tests for the data-pipeline health checker.

Fabricates partitioned parquet layouts under a tmp_path to exercise every
status branch (healthy / stale / gaps / missing / empty) and every
problem-detection path (empty partition, corrupt file, unknown interval).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rabbit_hunter.data_engine.health import (
    check_symbol, check_all, summarize, HealthReport, INTERVAL_MS,
)


def _write_bars(root: Path, symbol: str, interval: str,
                start_ms: int, n: int, gaps: set[int] | None = None) -> None:
    """Write `n` bars starting at `start_ms`, skipping any indices in `gaps`."""
    interval_ms = INTERVAL_MS[interval]
    ts = []
    for i in range(n):
        if gaps and i in gaps:
            continue
        ts.append(start_ms + i * interval_ms)
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [1.0] * len(ts),
        "high": [1.0] * len(ts),
        "low":  [1.0] * len(ts),
        "close": [1.0] * len(ts),
        "volume": [1.0] * len(ts),
    })
    # Partition by year/month like the storage module does
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    for (year, month), grp in df.groupby(["_year", "_month"]):
        p = root / "raw" / "okx" / symbol / interval / f"year={year}" / f"month={month:02d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        grp.drop(columns=["_year", "_month"]).to_parquet(p, index=False)


NOW_MS = 1_700_000_000_000  # fixed reference so tests are deterministic


# ============================================================
# Missing / empty
# ============================================================

def test_missing_when_no_partitions(tmp_path):
    r = check_symbol(tmp_path, "BTC-USDT-SWAP", "1H", now_ms=NOW_MS)
    assert r.status == "missing"
    assert r.rows == 0
    assert any("no partitions" in p for p in r.problems)


def test_missing_when_unknown_interval(tmp_path):
    r = check_symbol(tmp_path, "BTC-USDT-SWAP", "3H", now_ms=NOW_MS)
    assert r.status == "missing"
    assert any("unknown interval" in p for p in r.problems)


def test_empty_partition_flagged(tmp_path):
    """A partition that exists but has 0 rows must not be treated healthy."""
    p = tmp_path / "raw" / "okx" / "X" / "1H" / "year=2023" / "month=11.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": []}).to_parquet(p)
    r = check_symbol(tmp_path, "X", "1H", now_ms=NOW_MS)
    assert r.status == "empty"


def test_corrupt_partition_reports_missing(tmp_path):
    p = tmp_path / "raw" / "okx" / "X" / "1H" / "year=2023" / "month=11.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not parquet")
    r = check_symbol(tmp_path, "X", "1H", now_ms=NOW_MS)
    assert r.status == "missing"
    assert any("corrupt_parquet" in x for x in r.problems)


# ============================================================
# Healthy path
# ============================================================

def test_healthy_when_dense_and_fresh(tmp_path):
    # 100 bars ending exactly 1 bar before now (well inside grace)
    start = NOW_MS - 100 * INTERVAL_MS["1H"]
    _write_bars(tmp_path, "BTC-USDT-SWAP", "1H", start, 100)
    r = check_symbol(tmp_path, "BTC-USDT-SWAP", "1H", now_ms=NOW_MS,
                     freshness_grace_bars=2)
    assert r.status == "healthy", r.problems
    assert r.rows == 100
    assert r.missing_bars == 0


# ============================================================
# Stale
# ============================================================

def test_stale_when_last_bar_beyond_grace(tmp_path):
    # 100 bars ending 24h ago; grace=2 bars → definitely stale
    start = NOW_MS - 124 * INTERVAL_MS["1H"]
    _write_bars(tmp_path, "BTC-USDT-SWAP", "1H", start, 100)
    r = check_symbol(tmp_path, "BTC-USDT-SWAP", "1H", now_ms=NOW_MS,
                     freshness_grace_bars=2)
    assert r.status == "stale"
    assert any("stale=" in x for x in r.problems)


# ============================================================
# Gaps
# ============================================================

def test_gaps_when_bars_missing(tmp_path):
    """100 bars but skip indices 40..50 → 11 missing, 12-bar gap."""
    start = NOW_MS - 100 * INTERVAL_MS["1H"]
    _write_bars(
        tmp_path, "BTC-USDT-SWAP", "1H", start, 100,
        gaps=set(range(40, 51)),
    )
    r = check_symbol(tmp_path, "BTC-USDT-SWAP", "1H", now_ms=NOW_MS,
                     freshness_grace_bars=2)
    assert r.status == "gaps"
    assert r.missing_bars == 11
    # max_gap should be at least 12 bars * 1H
    assert r.max_gap_ms >= 12 * INTERVAL_MS["1H"]


# ============================================================
# check_all + summarize
# ============================================================

def test_check_all_iterates_symbols_and_intervals(tmp_path):
    for sym in ("BTC-USDT-SWAP", "ETH-USDT-SWAP"):
        _write_bars(tmp_path, sym, "1H",
                    NOW_MS - 20 * INTERVAL_MS["1H"], 20)
    reports = check_all(
        tmp_path, symbols=["BTC-USDT-SWAP", "ETH-USDT-SWAP"],
        intervals=["1H"], now_ms=NOW_MS, freshness_grace_bars=2,
    )
    assert len(reports) == 2
    assert all(r.status == "healthy" for r in reports)


def test_summarize_counts_by_status(tmp_path):
    _write_bars(tmp_path, "GOOD", "1H",
                NOW_MS - 20 * INTERVAL_MS["1H"], 20)
    # No partitions for BAD → missing
    reports = check_all(
        tmp_path, symbols=["GOOD", "BAD"],
        intervals=["1H"], now_ms=NOW_MS,
    )
    stats = summarize(reports)
    assert stats["total"] == 2
    assert stats["healthy"] == 1
    assert stats["unhealthy"] == 1
    assert stats["by_status"]["missing"] == 1


def test_to_row_iso_timestamps(tmp_path):
    _write_bars(tmp_path, "BTC-USDT-SWAP", "1H",
                NOW_MS - 20 * INTERVAL_MS["1H"], 20)
    r = check_symbol(tmp_path, "BTC-USDT-SWAP", "1H", now_ms=NOW_MS)
    row = r.to_row()
    # ISO-formatted, ends with +00:00
    assert row["last_ts"].endswith("+00:00")
    assert "T" in row["last_ts"]
