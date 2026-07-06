"""Heartbeat watchdog for the shadow runner.

Given a shadow state directory, checks whether the runner is still ticking
by inspecting the last row of `state/metrics_history.parquet`:

  - If `timestamp_ms` of the last row is within `max_silence_seconds`
    of now, the runner is HEALTHY.
  - If it's older, the runner is STALE.
  - If the file is missing, the runner is DOWN (never started, or
    state wiped).
  - If the file exists but is unreadable, the runner is DOWN with a
    corruption reason.

The primary use is a cron entry that runs `rabbit shadow watchdog`
periodically (say every 5 min) and pages / emails / whatever when the
process exits non-zero. It never talks to the runner; the runner never
knows about it. This makes the watchdog immune to bugs in the runner
itself — even a corrupted state parquet is a signal, not a hang.

Design specifically avoids:
  - Reading the whole history (only the last row).
  - Long file locks (opens read-only, closes immediately).
  - Touching the ledger (a bug in ledger persistence must not mask a
    dead runner).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class WatchdogResult:
    status: str                        # "healthy" | "stale" | "down"
    last_tick_ts_ms: int | None
    seconds_since_last_tick: float | None
    threshold_seconds: float
    reason: str

    @property
    def ok(self) -> bool:
        return self.status == "healthy"

    def as_line(self) -> str:
        if self.status == "healthy":
            return (f"watchdog: OK — last tick "
                    f"{self.seconds_since_last_tick:.0f}s ago "
                    f"(threshold {self.threshold_seconds:.0f}s)")
        if self.status == "stale":
            return (f"watchdog: STALE — last tick "
                    f"{self.seconds_since_last_tick:.0f}s ago "
                    f"exceeds threshold {self.threshold_seconds:.0f}s")
        return f"watchdog: DOWN — {self.reason}"


def check(
    state_dir: Path,
    max_silence_seconds: float = 300.0,
    now_ms: int | None = None,
) -> WatchdogResult:
    """Read the last metrics row and decide whether the runner is alive."""
    if now_ms is None:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    metrics_path = state_dir / "state" / "metrics_history.parquet"
    if not metrics_path.exists():
        return WatchdogResult(
            status="down", last_tick_ts_ms=None,
            seconds_since_last_tick=None,
            threshold_seconds=max_silence_seconds,
            reason=f"no metrics_history at {metrics_path}",
        )
    try:
        # Only the last row is needed; but parquet doesn't support tail
        # cheaply, so read the timestamp column and take the max.
        df = pd.read_parquet(metrics_path, columns=["timestamp_ms"])
    except Exception as e:
        return WatchdogResult(
            status="down", last_tick_ts_ms=None,
            seconds_since_last_tick=None,
            threshold_seconds=max_silence_seconds,
            reason=f"unreadable metrics_history: {e}",
        )
    if df.empty:
        return WatchdogResult(
            status="down", last_tick_ts_ms=None,
            seconds_since_last_tick=None,
            threshold_seconds=max_silence_seconds,
            reason="metrics_history is empty",
        )
    last_ts = int(df["timestamp_ms"].max())
    seconds_since = max(0.0, (now_ms - last_ts) / 1000.0)
    if seconds_since <= max_silence_seconds:
        return WatchdogResult(
            status="healthy", last_tick_ts_ms=last_ts,
            seconds_since_last_tick=seconds_since,
            threshold_seconds=max_silence_seconds,
            reason="",
        )
    return WatchdogResult(
        status="stale", last_tick_ts_ms=last_ts,
        seconds_since_last_tick=seconds_since,
        threshold_seconds=max_silence_seconds,
        reason=f"last tick was {seconds_since:.0f}s ago",
    )
