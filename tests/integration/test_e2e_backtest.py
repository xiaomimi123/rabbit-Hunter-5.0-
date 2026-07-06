from pathlib import Path
import pandas as pd
import yaml

from rabbit_hunter.config.loader import load_config
from rabbit_hunter.feature_engine.pipeline import build_features
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
from rabbit_hunter.backtest.engine import BacktestEngine
from rabbit_hunter.backtest.report import ReportBuilder
from rabbit_hunter.observability.snapshot import SnapshotWriter

from .fixtures.gen_synthetic_ohlcv import gen_synthetic


def test_full_pipeline_smoke(tmp_path):
    """Run the full pipeline (features -> scoring -> backtest -> report) on
    synthetic OHLCV data and verify the six deliverable files are produced.

    No network access is required: OHLCV data is generated in-process.
    """
    cfg = load_config(Path("configs/default.yaml"))
    cfg.backtest.start = "2025-01-01"
    cfg.backtest.end = "2025-04-01"

    features_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in cfg.data.symbols:
        raw = gen_synthetic(seed=42 if symbol == "BTC-USDT-SWAP" else 123)
        feats = build_features(raw, confirm=None, engine_version=cfg.feature_engine.version)
        features_by_symbol[symbol] = feats

    tf_cfg = yaml.safe_load(Path("configs/strategies/trend_following.yaml").read_text(encoding="utf-8"))
    tf = TrendFollowing(TFParams(**tf_cfg["params"]))

    engine = BacktestEngine(cfg, [tf])
    result = engine.run(features_by_symbol, open_action_threshold=0.3)

    sw = SnapshotWriter(root=tmp_path / "snapshots")
    sw.append(result.snapshots.to_dict(orient="records"))
    sw.flush()

    builder = ReportBuilder(cfg, features_by_symbol)
    out_dir = builder.build(result, output_root=tmp_path / "reports", git_commit="test")

    for f in ("report.md", "ai_context.md", "trades.parquet", "snapshots.parquet",
              "config_snapshot.yaml", "charts/equity_curve.png"):
        p = out_dir / f
        assert p.exists() and p.stat().st_size > 0, f"missing or empty: {f}"

    report_md = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "# Rabbit Hunter" in report_md
    ai_md = (out_dir / "ai_context.md").read_text(encoding="utf-8")
    assert "Baseline Comparisons" in ai_md
