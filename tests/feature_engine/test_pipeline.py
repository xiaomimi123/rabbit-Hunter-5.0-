from pathlib import Path
import numpy as np
import pandas as pd
from rabbit_hunter.feature_engine.pipeline import build_features, load_or_compute_features


def _mk_raw(n=400, base=100.0):
    ts = [i * 3_600_000 for i in range(n)]
    close = np.linspace(base, base + 100, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 10.0),
        "funding_rate": np.full(n, 0.0001),
        "oi": np.linspace(1000, 1500, n),
    })


def test_build_features_columns_present():
    raw = _mk_raw()
    feats = build_features(raw)
    for col in [
        "ema20", "adx", "rsi_14", "atr_14",
        "pattern_engulfing_bull", "structure_regime",
        "regime", "session", "day_of_week",
        "funding_rate", "oi_change_pct",
    ]:
        assert col in feats.columns, f"missing {col}"
    assert len(feats) == len(raw)


def test_no_lookahead_prefix_matches():
    raw = _mk_raw()
    full = build_features(raw)
    prefix = build_features(raw.iloc[:-10])
    for col in ["ema20", "adx", "rsi_14", "atr_14"]:
        np.testing.assert_allclose(
            full[col].iloc[:-10].to_numpy(),
            prefix[col].to_numpy(),
            equal_nan=True,
        )


def test_cache_hit_returns_same(tmp_path):
    raw = _mk_raw()

    def fetch():
        return raw

    a = load_or_compute_features(
        root=tmp_path, symbol="TEST-SWAP", interval="1H",
        engine_version="0.1.0", fetch_raw=fetch,
    )
    # 第二次调用不应触发 fetch（用异常检测）
    def fetch_should_not_run():
        raise AssertionError("cache should have hit")

    b = load_or_compute_features(
        root=tmp_path, symbol="TEST-SWAP", interval="1H",
        engine_version="0.1.0", fetch_raw=fetch_should_not_run,
    )
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


def test_baseline_snapshot_stable():
    """
    锁死特征值。任何影响历史特征的改动都必须显式更新 baseline。
    """
    raw = _mk_raw(n=250)
    feats = build_features(raw)
    baseline_path = Path(__file__).resolve().parents[1] / "baselines" / "features_v0_1_0.csv"
    check_cols = ["timestamp", "ema20", "ema60", "adx", "rsi_14", "atr_14", "regime"]
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        feats[check_cols].tail(50).to_csv(baseline_path, index=False)
    expected = pd.read_csv(baseline_path)
    actual = feats[check_cols].tail(50).reset_index(drop=True)
    for c in ["ema20", "ema60", "adx", "rsi_14", "atr_14"]:
        np.testing.assert_allclose(actual[c].to_numpy(), expected[c].to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True)
    assert (actual["regime"].to_numpy() == expected["regime"].to_numpy()).all()
