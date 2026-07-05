from __future__ import annotations
from pathlib import Path
import pandas as pd
import duckdb


def _partition_path(root: Path, symbol: str, interval: str, year: int, month: int) -> Path:
    return root / "raw" / "okx" / symbol / interval / f"year={year}" / f"month={month:02d}.parquet"


def write_ohlcv(df: pd.DataFrame, root: Path, symbol: str, interval: str) -> list[Path]:
    if df.empty:
        return []
    df = df.copy()
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
