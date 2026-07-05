from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

from .indicators import compute_indicators
from .price_action import compute_price_action
from .regime import compute_regime


def _align_1h_on_15m(main_1h: pd.DataFrame, confirm_15m: pd.DataFrame) -> pd.DataFrame:
    """把 1H 的 ema20/adx 前向填充到 15m 时间轴。confirm_15m 必须含 timestamp。"""
    right = main_1h[["timestamp", "ema20", "adx"]].rename(
        columns={"ema20": "ema20_1h_on_15m", "adx": "adx_1h_on_15m"}
    ).sort_values("timestamp")
    left = confirm_15m[["timestamp"]].sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    return merged


def build_features(
    raw: pd.DataFrame,
    confirm: pd.DataFrame | None = None,
    engine_version: str = "0.1.0",
) -> pd.DataFrame:
    df = raw.copy().reset_index(drop=True)
    df = compute_indicators(df)
    df = compute_price_action(df)
    df = compute_regime(df)

    if "funding_rate" not in df.columns:
        df["funding_rate"] = np.nan
    if "oi" in df.columns:
        df["oi_change_pct"] = df["oi"].pct_change().fillna(0.0)
    else:
        df["oi_change_pct"] = np.nan

    if confirm is not None and not confirm.empty:
        # 计算 15m 上的 indicators
        confirm_ind = compute_indicators(confirm.copy().reset_index(drop=True))
        aligned = _align_1h_on_15m(df, confirm_ind)
        df = df.merge(aligned, on="timestamp", how="left")
    else:
        df["ema20_1h_on_15m"] = np.nan
        df["adx_1h_on_15m"] = np.nan

    df.attrs["engine_version"] = engine_version
    return df


def _cache_path(root: Path, symbol: str, interval: str, engine_version: str) -> Path:
    return root / "features" / symbol / interval / f"features_v{engine_version}.parquet"


def load_or_compute_features(
    root: Path,
    symbol: str,
    interval: str,
    engine_version: str,
    fetch_raw: Callable[[], pd.DataFrame],
    fetch_confirm: Callable[[], pd.DataFrame] | None = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    cache = _cache_path(root, symbol, interval, engine_version)
    if cache.exists() and not force_recompute:
        return pd.read_parquet(cache)
    raw = fetch_raw()
    confirm = fetch_confirm() if fetch_confirm is not None else None
    feats = build_features(raw, confirm, engine_version=engine_version)
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache, index=False)
    return feats
