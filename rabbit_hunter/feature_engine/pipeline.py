from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd

from .indicators import compute_indicators
from .price_action import compute_price_action
from .regime import compute_regime


def _align_confirm_on_main(main_ts_df: pd.DataFrame, confirm_indicators: pd.DataFrame) -> pd.DataFrame:
    """Align 15m confirm indicators onto the 1H main timestamps via backward merge_asof.

    NOTE: column names keep the historical "_1h_on_15m" suffix for backward
    compatibility, but the values now genuinely come from the 15m confirm
    timeframe's own indicators (computed on `confirm_indicators`), not the
    1H main frame's own ema20/adx.
    """
    right = confirm_indicators[["timestamp", "ema20", "adx"]].rename(
        columns={"ema20": "ema20_1h_on_15m", "adx": "adx_1h_on_15m"}
    ).sort_values("timestamp")
    left = main_ts_df[["timestamp"]].sort_values("timestamp")
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
        # 计算 15m 上的 indicators，再用 backward merge_asof 对齐到 1H 主时间轴
        confirm_ind = compute_indicators(confirm.copy().reset_index(drop=True))
        aligned = _align_confirm_on_main(df, confirm_ind)
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
    # Raw read is cheap (local parquet via duckdb); the expensive part is
    # build_features. Always load raw so we can validate cache freshness —
    # a cache whose newest bar is older than the raw data's newest bar is
    # stale (new bars were archived since it was written) and must be
    # recomputed. Without this check an extended backtest silently reuses
    # last week's features and reports byte-identical results.
    raw = fetch_raw()
    if cache.exists() and not force_recompute:
        feats = pd.read_parquet(cache)
        cache_fresh = (
            len(feats) > 0 and len(raw) > 0
            and int(feats["timestamp"].max()) >= int(raw["timestamp"].max())
        )
        if cache_fresh:
            return feats
    confirm = fetch_confirm() if fetch_confirm is not None else None
    feats = build_features(raw, confirm, engine_version=engine_version)
    cache.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(cache, index=False)
    return feats
