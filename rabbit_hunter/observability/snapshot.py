from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def _jsonify_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize columns holding dict/list values to JSON strings.

    Parquet (via pyarrow) cannot reliably write struct columns whose values
    have inconsistent/empty field sets across rows (e.g. a ``{}`` alongside
    a ``{"k": 1}``), which happens once records are split into per-day
    groups. JSON-encoding those columns up front sidesteps that entirely.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    return df


class SnapshotWriter:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._buffer: list[dict] = []

    def append(self, records: list[dict]) -> None:
        self._buffer.extend(records)

    def flush(self) -> list[Path]:
        if not self._buffer:
            return []
        df = pd.DataFrame(self._buffer)
        df["_day"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
        df = _jsonify_nested_columns(df)
        written: list[Path] = []
        for day, grp in df.groupby("_day", sort=True):
            out = self.root / day / "snapshot.parquet"
            out.parent.mkdir(parents=True, exist_ok=True)
            grp.drop(columns=["_day"]).to_parquet(out, index=False)
            written.append(out)
        self._buffer.clear()
        return written
