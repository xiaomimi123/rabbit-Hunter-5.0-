"""Data-pipeline health check — the "will my next backtest reveal a
silent gap?" tool.

For each configured symbol × interval, checks:
  1. File presence — any partition files exist under the expected root.
  2. Freshness — max(timestamp) vs a caller-supplied `now_ms`, with an
     interval-aware threshold (2 bars grace so a lagging OKX fetch
     doesn't false-positive).
  3. Gaps — count of missing bars between min and max timestamp, given
     the interval's expected step (1H = 3_600_000 ms).
  4. Row-count sanity — at least a handful of bars must exist per
     partition (guards against corrupt writes that produced empty files).

Design decisions:
  - Read-only. The tool never rewrites parquet — a report tells you what
    to fix; you decide whether to re-fetch, re-ingest, or ignore.
  - Aggregate per symbol first, then across symbols, so the report can
    still say "SOL is broken; others are healthy" instead of one all-or-
    nothing pass/fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


INTERVAL_MS: dict[str, int] = {
    "1H": 3_600_000, "1h": 3_600_000,
    "15m": 900_000,
    "5m": 300_000,
    "1D": 86_400_000,
}


@dataclass
class HealthReport:
    """Per-(symbol, interval) health snapshot."""
    symbol: str
    interval: str
    status: str                # "healthy" | "stale" | "gaps" | "missing" | "empty"
    rows: int
    first_ts_ms: int | None
    last_ts_ms: int | None
    missing_bars: int
    max_gap_ms: int
    partition_count: int
    problems: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        def dt(ts):
            if ts is None:
                return ""
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "status": self.status,
            "rows": self.rows,
            "first_ts": dt(self.first_ts_ms),
            "last_ts": dt(self.last_ts_ms),
            "missing_bars": self.missing_bars,
            "max_gap_hours": round(self.max_gap_ms / 3_600_000, 2),
            "partitions": self.partition_count,
            "problems": ";".join(self.problems),
        }


def _symbol_dir(root: Path, symbol: str, interval: str) -> Path:
    return root / "raw" / "okx" / symbol / interval


def _list_partitions(root: Path, symbol: str, interval: str) -> list[Path]:
    d = _symbol_dir(root, symbol, interval)
    if not d.exists():
        return []
    return sorted(d.glob("year=*/month=*.parquet"))


def check_symbol(
    root: Path,
    symbol: str,
    interval: str,
    now_ms: int | None = None,
    freshness_grace_bars: int = 2,
) -> HealthReport:
    """Compute a HealthReport for one (symbol, interval)."""
    if now_ms is None:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    interval_ms = INTERVAL_MS.get(interval)
    if interval_ms is None:
        return HealthReport(
            symbol=symbol, interval=interval, status="missing",
            rows=0, first_ts_ms=None, last_ts_ms=None,
            missing_bars=0, max_gap_ms=0, partition_count=0,
            problems=[f"unknown interval={interval}"],
        )

    partitions = _list_partitions(root, symbol, interval)
    if not partitions:
        return HealthReport(
            symbol=symbol, interval=interval, status="missing",
            rows=0, first_ts_ms=None, last_ts_ms=None,
            missing_bars=0, max_gap_ms=0, partition_count=0,
            problems=[f"no partitions under {_symbol_dir(root, symbol, interval)}"],
        )

    frames: list[pd.DataFrame] = []
    empty_partitions: list[Path] = []
    for p in partitions:
        try:
            df = pd.read_parquet(p, columns=["timestamp"])
        except Exception as e:
            return HealthReport(
                symbol=symbol, interval=interval, status="missing",
                rows=0, first_ts_ms=None, last_ts_ms=None,
                missing_bars=0, max_gap_ms=0,
                partition_count=len(partitions),
                problems=[f"corrupt_parquet: {p.name}: {e}"],
            )
        if df.empty:
            empty_partitions.append(p)
        else:
            frames.append(df)

    if not frames:
        return HealthReport(
            symbol=symbol, interval=interval, status="empty",
            rows=0, first_ts_ms=None, last_ts_ms=None,
            missing_bars=0, max_gap_ms=0,
            partition_count=len(partitions),
            problems=[f"all {len(partitions)} partitions empty"],
        )

    ts = pd.concat(frames)["timestamp"].sort_values().reset_index(drop=True)
    ts = ts.drop_duplicates()
    first_ts = int(ts.iloc[0])
    last_ts = int(ts.iloc[-1])
    span = last_ts - first_ts
    expected = span // interval_ms + 1
    actual = len(ts)
    missing = max(0, int(expected - actual))
    diffs = ts.diff().dropna().astype("int64")
    max_gap = int(diffs.max()) if not diffs.empty else 0

    problems: list[str] = []
    if empty_partitions:
        problems.append(
            f"empty_partitions={[p.name for p in empty_partitions]}"
        )
    if missing > 0:
        problems.append(f"missing_bars={missing}")
    if max_gap > interval_ms * 5:  # any gap >5 bars is suspicious
        problems.append(f"largest_gap={max_gap / 3_600_000:.1f}h")

    minutes_stale = (now_ms - last_ts) / 60_000.0
    stale_bars = (now_ms - last_ts) / interval_ms
    if stale_bars > freshness_grace_bars:
        problems.append(
            f"stale={minutes_stale:.0f}min (>{freshness_grace_bars} bars grace)"
        )

    if problems and any("stale=" in x for x in problems):
        status = "stale"
    elif missing > 0 or max_gap > interval_ms * 5:
        status = "gaps"
    elif empty_partitions:
        status = "empty"
    else:
        status = "healthy"

    return HealthReport(
        symbol=symbol, interval=interval, status=status,
        rows=int(actual),
        first_ts_ms=first_ts, last_ts_ms=last_ts,
        missing_bars=missing, max_gap_ms=max_gap,
        partition_count=len(partitions),
        problems=problems,
    )


def check_all(
    root: Path,
    symbols: list[str],
    intervals: list[str],
    now_ms: int | None = None,
    freshness_grace_bars: int = 2,
) -> list[HealthReport]:
    reports: list[HealthReport] = []
    for symbol in symbols:
        for interval in intervals:
            reports.append(check_symbol(
                root=root, symbol=symbol, interval=interval,
                now_ms=now_ms, freshness_grace_bars=freshness_grace_bars,
            ))
    return reports


def summarize(reports: list[HealthReport]) -> dict:
    """Aggregate stats used by the CLI exit-code decision."""
    by_status: dict[str, int] = {}
    for r in reports:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    return {
        "total": len(reports),
        "by_status": by_status,
        "healthy": by_status.get("healthy", 0),
        "unhealthy": len(reports) - by_status.get("healthy", 0),
    }
