import json
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from rabbit_hunter.config.loader import load_config
from rabbit_hunter.shadow.runner import ShadowRunner, ShadowConfig


class _StubStrategy:
    """A minimum BaseStrategy-shaped stub for wiring tests."""
    name = "stub"
    version = "0.1.0"

    def __init__(self, long: float = 0.0, short: float = 0.0):
        self.long = long
        self.short = short

    def score(self, features_row, features_history):
        from rabbit_hunter.scoring_engine.base import ScoreOutput
        return ScoreOutput(
            long=self.long, short=self.short,
            components={"stub": 0.0}, metadata={},
        )


def _mk_feats(n=250, base_price=100.0):
    """Synthesize a Feature Engine output frame — same columns build_features
    would produce."""
    ts = [i * 3_600_000 for i in range(n)]
    close = np.linspace(base_price, base_price + 20, n)
    high = close + 1.0
    low = close - 1.0
    open_ = close - 0.5
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "volume": np.full(n, 100.0),
        "ema20": close - 0.5, "ema60": close - 1.0, "ema200": close - 2.0,
        "ema20_slope": np.full(n, 0.1),
        "adx": np.full(n, 25.0),
        "di_plus": np.full(n, 20.0), "di_minus": np.full(n, 15.0),
        "rsi_14": np.full(n, 55.0),
        "bb_upper": close + 2.0, "bb_middle": close, "bb_lower": close - 2.0,
        "bb_width": np.full(n, 0.04), "bb_pct": np.full(n, 0.6),
        "zscore_20": np.full(n, 0.5),
        "atr_14": np.full(n, 1.0), "atr_pct": np.full(n, 0.01),
        "volume_ratio_20": np.full(n, 1.0),
        "funding_rate": np.full(n, 0.0001),
        "oi_change_pct": np.full(n, 0.0),
        "ema20_1h_on_15m": close - 0.5, "adx_1h_on_15m": np.full(n, 25.0),
        "quote_volume_24h": np.full(n, 5_000_000.0),
        "atr_pct_baseline": np.full(n, 0.01),
        "regime": ["trending"] * n,
        "session": ["us"] * n,
        "day_of_week": [0] * n,
        "pattern_engulfing_bull": [0] * n, "pattern_engulfing_bear": [0] * n,
        "pattern_pinbar": [0] * n, "pattern_inside_bar": [0] * n,
        "pattern_doji": [0] * n, "pattern_marubozu": [0] * n,
        "swing_high_last": close + 5.0, "swing_low_last": close - 5.0,
        "structure_regime": ["uptrend"] * n,
        "bos_flag": [0] * n, "choch_flag": [0] * n,
    })


def test_shadow_runner_creates_state_dir(tmp_path):
    cfg = load_config("configs/default.yaml")
    strategies = [_StubStrategy()]
    _ = ShadowRunner(cfg, strategies, ShadowConfig(state_dir=tmp_path / "s"))
    assert (tmp_path / "s" / "state").exists()


def test_shadow_runner_persistence_roundtrip(tmp_path):
    """A fresh runner should load a new Ledger; a second instance in the same
    state_dir should resume the persisted one."""
    cfg = load_config("configs/default.yaml")
    strategies = [_StubStrategy()]
    r1 = ShadowRunner(cfg, strategies, ShadowConfig(state_dir=tmp_path / "s"))
    # Muck with the ledger + save
    r1.ledger.equity = 12345.67
    r1.last_seen_ts = {"BTC-USDT-SWAP": 1234567890}
    r1._save_ledger()
    r1._save_last_seen_ts()

    r2 = ShadowRunner(cfg, strategies, ShadowConfig(state_dir=tmp_path / "s"))
    assert r2.ledger.equity == 12345.67
    assert r2.last_seen_ts == {"BTC-USDT-SWAP": 1234567890}


def test_shadow_snapshot_writes_parquet_per_day(tmp_path):
    cfg = load_config("configs/default.yaml")
    strategies = [_StubStrategy()]
    r = ShadowRunner(cfg, strategies, ShadowConfig(state_dir=tmp_path / "s"))

    # Two records on different days
    r._write_snapshot({
        "timestamp": 1_700_000_000_000,  # 2023-11-14
        "symbol": "BTC-USDT-SWAP", "action": "wait", "conviction": 0.0,
    })
    r._write_snapshot({
        "timestamp": 1_700_086_400_000,  # 2023-11-15
        "symbol": "BTC-USDT-SWAP", "action": "open_long", "conviction": 0.7,
    })
    # And an append to the first day
    r._write_snapshot({
        "timestamp": 1_700_003_600_000,  # 2023-11-14 later
        "symbol": "ETH-USDT-SWAP", "action": "wait", "conviction": 0.0,
    })

    d1 = tmp_path / "s" / "2023-11-14" / "snapshots.parquet"
    d2 = tmp_path / "s" / "2023-11-15" / "snapshots.parquet"
    assert d1.exists() and d2.exists()

    df1 = pd.read_parquet(d1)
    assert len(df1) == 2
    df2 = pd.read_parquet(d2)
    assert len(df2) == 1
    assert df2.iloc[0]["action"] == "open_long"


def test_shadow_tick_skips_when_no_new_bar(tmp_path):
    """If last_seen_ts is already at (or past) the latest bar, tick should be
    a no-op — no snapshot written."""
    cfg = load_config("configs/default.yaml")
    strategies = [_StubStrategy()]
    r = ShadowRunner(cfg, strategies, ShadowConfig(state_dir=tmp_path / "s"))

    feats = _mk_feats(n=250)
    latest_ts = int(feats.iloc[-1]["timestamp"])
    # Pretend we've already seen it
    r.last_seen_ts = {sym: latest_ts + 1 for sym in cfg.data.symbols}

    with patch.object(r, "_fetch_recent_features", return_value=feats):
        r.tick()

    # No snapshot dir under state root
    day_dirs = [d for d in (tmp_path / "s").iterdir() if d.is_dir() and d.name != "state"]
    assert day_dirs == []


def test_shadow_tick_skips_early_warmup_bars(tmp_path):
    """Bars with NaN atr_14 / ema200 (warmup) must not be handled."""
    cfg = load_config("configs/default.yaml")
    strategies = [_StubStrategy()]
    r = ShadowRunner(cfg, strategies, ShadowConfig(state_dir=tmp_path / "s"))

    # Feats with NaN atr on last row
    feats = _mk_feats(n=250)
    feats.loc[feats.index[-1], "atr_14"] = np.nan

    with patch.object(r, "_fetch_recent_features", return_value=feats):
        r.tick()

    # Nothing processed — snapshot dir empty
    day_dirs = [d for d in (tmp_path / "s").iterdir() if d.is_dir() and d.name != "state"]
    assert day_dirs == []
