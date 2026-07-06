import numpy as np
import pandas as pd


def gen_synthetic(n_bars: int = 24 * 90, base: float = 50_000.0, seed: int = 42) -> pd.DataFrame:
    """3 个月 1H 数据，含随机波动 + 缓慢上涨趋势。"""
    rng = np.random.default_rng(seed)
    ts = [i * 3_600_000 for i in range(n_bars)]
    trend = np.linspace(0, base * 0.5, n_bars)
    noise = rng.normal(0, base * 0.005, n_bars).cumsum()
    close = base + trend + noise
    close = np.clip(close, 1.0, None)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0, base * 0.002, n_bars)
    low = np.minimum(open_, close) - rng.uniform(0, base * 0.002, n_bars)
    low = np.clip(low, 1.0, None)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.uniform(100, 1000, n_bars),
        "funding_rate": rng.uniform(-0.0001, 0.0002, n_bars),
        "oi": np.linspace(1000, 1500, n_bars),
    })
