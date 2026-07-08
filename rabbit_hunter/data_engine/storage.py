from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import duckdb


# Plausibility bounds. Crypto perpetual markets on OKX predate 2019 by
# months at earliest; any row with a timestamp before 2019-01-01 is
# either seed/test data or a fetcher bug. And anything more than a day
# in the future is a clock-drift artifact. Both classes get dropped
# before hitting disk so bad rows can never poison the archive.
_MIN_PLAUSIBLE_MS = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _now_plus_grace_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000) + 86_400_000  # +1 day


def _partition_path(root: Path, symbol: str, interval: str, year: int, month: int) -> Path:
    return root / "raw" / "okx" / symbol / interval / f"year={year}" / f"month={month:02d}.parquet"


def write_ohlcv(df: pd.DataFrame, root: Path, symbol: str, interval: str) -> list[Path]:
    if df.empty:
        return []
    df = df.copy()
    # Reject rows outside the plausible timestamp window BEFORE we touch
    # disk. Without this guard, a fetcher that momentarily returned
    # zero-timestamp seed rows (as happened once with a stub) would
    # write a year=1970 partition and permanently break gap-detection
    # in data/health.
    upper = _now_plus_grace_ms()
    df = df[(df["timestamp"] >= _MIN_PLAUSIBLE_MS) & (df["timestamp"] <= upper)]
    if df.empty:
        return []
    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    written: list[Path] = []
    for (year, month), grp in df.groupby(["_year", "_month"], sort=True):
        out = _partition_path(root, symbol, interval, int(year), int(month))
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            existing = pd.read_parquet(out)
            combined = pd.concat([existing, grp.drop(columns=["_year", "_month"])])
            combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        else:
            combined = grp.drop(columns=["_year", "_month"]).sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(out, index=False)
        written.append(out)
    return written


def read_ohlcv(root: Path, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    glob = str(root / "raw" / "okx" / symbol / interval / "year=*" / "month=*.parquet")
    con = duckdb.connect()
    q = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM read_parquet('{glob}')
        WHERE timestamp >= {start_ms} AND timestamp < {end_ms}
        ORDER BY timestamp
    """
    df = con.execute(q).fetch_df()
    con.close()
    return df.reset_index(drop=True)
