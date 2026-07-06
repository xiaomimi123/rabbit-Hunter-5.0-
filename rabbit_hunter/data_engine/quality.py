from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

_INTERVAL_MS = {"1H": 3_600_000, "15m": 900_000, "5m": 300_000, "1D": 86_400_000}


@dataclass
class QualityReport:
    clean_df: pd.DataFrame
    issues: list[dict] = field(default_factory=list)

    @property
    def is_ok(self) -> bool:
        return len(self.issues) == 0


def check_ohlcv(df: pd.DataFrame, interval: str) -> QualityReport:
    issues: list[dict] = []
    step = _INTERVAL_MS[interval]

    # 排序
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # 重复时间戳
    dup_mask = df["timestamp"].duplicated(keep="first")
    if dup_mask.any():
        for ts in df.loc[dup_mask, "timestamp"].tolist():
            issues.append({"type": "duplicate_timestamp", "timestamp": int(ts)})
    df = df[~dup_mask].reset_index(drop=True)

    # 无效价格 / 无效成交量 / NaN
    price_cols = ["open", "high", "low", "close"]
    bad_price_mask = (
        df[price_cols].le(0).any(axis=1)
        | df[price_cols].isna().any(axis=1)
        | df["volume"].lt(0)
        | df["volume"].isna()
    )
    if bad_price_mask.any():
        for ts in df.loc[bad_price_mask, "timestamp"].tolist():
            issues.append({"type": "invalid_price", "timestamp": int(ts)})
        df = df[~bad_price_mask].reset_index(drop=True)

    # High/Low 关系检查
    hilo_bad = (df["high"] < df["low"]) | (df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))
    if hilo_bad.any():
        for ts in df.loc[hilo_bad, "timestamp"].tolist():
            issues.append({"type": "invalid_hilo", "timestamp": int(ts)})
        df = df[~hilo_bad].reset_index(drop=True)

    # 跳空
    if len(df) >= 2:
        diffs = df["timestamp"].diff().iloc[1:]
        gap_mask = diffs > step
        for pos, is_gap in enumerate(gap_mask, start=1):
            if is_gap:
                issues.append({
                    "type": "gap",
                    "before_ts": int(df["timestamp"].iloc[pos - 1]),
                    "after_ts": int(df["timestamp"].iloc[pos]),
                    "missing_bars": int(diffs.iloc[pos - 1] // step) - 1,
                })

    return QualityReport(clean_df=df, issues=issues)
