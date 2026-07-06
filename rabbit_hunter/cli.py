from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import typer
import pandas as pd
from rabbit_hunter.config.loader import load_config
from rabbit_hunter.observability.logger import configure_logger, get_logger

app = typer.Typer(help="Rabbit Hunter V5.1 Phase 1a CLI")
data_app = typer.Typer(help="Data engine commands")
app.add_typer(data_app, name="data")
features_app = typer.Typer(help="Feature engine commands")
app.add_typer(features_app, name="features")
ai_app = typer.Typer(help="AI Review Agent commands")
app.add_typer(ai_app, name="ai")
ml_app = typer.Typer(help="ML training commands")
app.add_typer(ml_app, name="ml")


def _iso_to_ms(s: str) -> int:
    return int(datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


@app.command()
def fetch(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
):
    """拉配置里所有 symbols × intervals 的 K 线 + funding + OI。"""
    configure_logger()
    log = get_logger("cli.fetch")
    cfg = load_config(config)
    from rabbit_hunter.data_engine.okx_fetcher import (
        fetch_ohlcv, fetch_open_interest_history,
    )
    from rabbit_hunter.data_engine.binance_funding import fetch_funding_rate_history_binance
    from rabbit_hunter.data_engine.quality import check_ohlcv
    from rabbit_hunter.data_engine.storage import write_ohlcv

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)

    for symbol in cfg.data.symbols:
        for interval in (cfg.data.main_interval, cfg.data.confirm_interval):
            log.info("fetch_start", symbol=symbol, interval=interval)
            df = fetch_ohlcv(symbol, interval, start_ms, end_ms)
            qr = check_ohlcv(df, interval)
            paths = write_ohlcv(qr.clean_df, data_root, symbol, interval)
            log.info("fetch_done", symbol=symbol, interval=interval,
                     rows=len(qr.clean_df), issues=len(qr.issues), files=[str(p) for p in paths])

        # funding from Binance (deep history: 3+ years vs OKX ~90 days).
        # OI still from OKX (matches trading target; OKX-only limitation).
        log.info("fetch_funding_start", symbol=symbol, source="binance")
        fr = fetch_funding_rate_history_binance(symbol, start_ms, end_ms)
        oi = fetch_open_interest_history(symbol, start_ms, end_ms)
        (data_root / "raw" / "okx" / symbol).mkdir(parents=True, exist_ok=True)
        fr.to_parquet(data_root / "raw" / "okx" / symbol / "funding.parquet", index=False)
        oi.to_parquet(data_root / "raw" / "okx" / symbol / "oi.parquet", index=False)
        log.info("fetch_funding_done", symbol=symbol, funding_rows=len(fr), oi_rows=len(oi))
    typer.echo("fetch done")


@data_app.command("quality")
def data_quality(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
    report_out: Path = typer.Option(Path("data/quality_report.md")),
):
    """扫描已下载数据的质量并输出 md 报告。"""
    configure_logger()
    cfg = load_config(config)
    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.data_engine.quality import check_ohlcv

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)
    lines = ["# Data Quality Report", ""]
    for symbol in cfg.data.symbols:
        for interval in (cfg.data.main_interval, cfg.data.confirm_interval):
            df = read_ohlcv(data_root, symbol, interval, start_ms, end_ms)
            qr = check_ohlcv(df, interval)
            lines.append(f"## {symbol} @ {interval}")
            lines.append(f"- rows: {len(qr.clean_df)}")
            lines.append(f"- issues: {len(qr.issues)}")
            for i in qr.issues[:20]:
                lines.append(f"  - {i}")
            lines.append("")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"quality report written to {report_out}")


@features_app.command("build")
def features_build(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
):
    """预计算并缓存所有 symbol × interval 的特征。"""
    configure_logger()
    cfg = load_config(config)
    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.feature_engine.pipeline import load_or_compute_features

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)
    for symbol in cfg.data.symbols:
        def _raw(sym=symbol):
            df = read_ohlcv(data_root, sym, cfg.data.main_interval, start_ms, end_ms)
            fr_path = data_root / "raw" / "okx" / sym / "funding.parquet"
            oi_path = data_root / "raw" / "okx" / sym / "oi.parquet"
            if fr_path.exists():
                fr = pd.read_parquet(fr_path)
                df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            if oi_path.exists():
                oi = pd.read_parquet(oi_path)
                df = pd.merge_asof(df.sort_values("timestamp"), oi.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            return df

        def _confirm(sym=symbol):
            return read_ohlcv(data_root, sym, cfg.data.confirm_interval, start_ms, end_ms)

        feats = load_or_compute_features(
            root=data_root, symbol=symbol, interval=cfg.data.main_interval,
            engine_version=cfg.feature_engine.version,
            fetch_raw=_raw, fetch_confirm=_confirm,
        )
        typer.echo(f"features for {symbol} @ {cfg.data.main_interval}: {len(feats)} rows")


@app.command()
def backtest(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
    report_root: Path = typer.Option(Path("reports")),
    snapshot_root: Path = typer.Option(Path("snapshots")),
    start: str = typer.Option(None),
    end: str = typer.Option(None),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """端到端跑回测。"""
    configure_logger()
    log = get_logger("cli.backtest")
    cfg = load_config(config)
    if start:
        cfg.backtest.start = start
    if end:
        cfg.backtest.end = end

    if dry_run:
        typer.echo(f"dry-run: config loaded, symbols={cfg.data.symbols}, "
                   f"strategies={list(cfg.strategy_router.enabled_strategies.keys())}")
        raise typer.Exit(code=0)

    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)

    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.feature_engine.pipeline import load_or_compute_features
    from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
    from rabbit_hunter.scoring_engine.strategies.mean_reversion import MeanReversion, MRParams
    from rabbit_hunter.scoring_engine.strategies.price_action import PriceAction, PAParams
    from rabbit_hunter.ml.ml_scoring import MLScoring, MLScoringParams
    from rabbit_hunter.backtest.engine import BacktestEngine
    from rabbit_hunter.backtest.report import ReportBuilder
    from rabbit_hunter.observability.snapshot import SnapshotWriter
    import yaml as _yaml

    # Strategy registry: name → (class, params_class). Adding a new plugin =
    # add one line here + one new file under strategies/ + one YAML.
    _STRATEGY_REGISTRY = {
        "trend_following": (TrendFollowing, TFParams),
        "mean_reversion": (MeanReversion, MRParams),
        "price_action": (PriceAction, PAParams),
        "ml_scoring": (MLScoring, MLScoringParams),
    }

    features_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in cfg.data.symbols:
        def _raw(sym=symbol):
            df = read_ohlcv(data_root, sym, cfg.data.main_interval, start_ms, end_ms)
            fr_path = data_root / "raw" / "okx" / sym / "funding.parquet"
            oi_path = data_root / "raw" / "okx" / sym / "oi.parquet"
            if fr_path.exists():
                fr = pd.read_parquet(fr_path)
                df = pd.merge_asof(df.sort_values("timestamp"), fr.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            if oi_path.exists():
                oi = pd.read_parquet(oi_path)
                df = pd.merge_asof(df.sort_values("timestamp"), oi.sort_values("timestamp"),
                                   on="timestamp", direction="backward")
            return df

        def _confirm(sym=symbol):
            return read_ohlcv(data_root, sym, cfg.data.confirm_interval, start_ms, end_ms)

        feats = load_or_compute_features(
            root=data_root, symbol=symbol, interval=cfg.data.main_interval,
            engine_version=cfg.feature_engine.version,
            fetch_raw=_raw, fetch_confirm=_confirm,
        )
        features_by_symbol[symbol] = feats

    # Instantiate all enabled strategies from config via the registry.
    strategies = []
    for name, entry in cfg.strategy_router.enabled_strategies.items():
        if name not in _STRATEGY_REGISTRY:
            raise typer.BadParameter(
                f"Unknown strategy '{name}' in enabled_strategies. "
                f"Registered: {sorted(_STRATEGY_REGISTRY)}"
            )
        Strat, ParamsCls = _STRATEGY_REGISTRY[name]
        strat_cfg_path = Path("configs") / entry.config_file
        strat_yaml = _yaml.safe_load(strat_cfg_path.read_text(encoding="utf-8"))
        strategies.append(Strat(ParamsCls(**strat_yaml["params"])))
        log.info("strategy_loaded", name=name, config=str(strat_cfg_path))

    engine = BacktestEngine(cfg, strategies)
    result = engine.run(features_by_symbol)

    # 写快照
    sw = SnapshotWriter(root=snapshot_root)
    sw.append(result.snapshots.to_dict(orient="records"))
    sw.flush()

    # 生成报告
    builder = ReportBuilder(cfg, features_by_symbol)
    out_dir = builder.build(result, output_root=report_root, git_commit=_git_commit())
    log.info("backtest_done", report=str(out_dir),
             trades=len(result.ledger.closed_trades),
             final_equity=result.ledger.equity)
    typer.echo(f"report: {out_dir}")


@ai_app.command("review")
def ai_review(
    report_dir: Path = typer.Argument(None, help="Path to reports/YYYY-MM-DD-HHMM/ (default: latest)"),
    out_file: Path = typer.Option(None, "--out", "-o", help="Write prompt to this file (default: stdout)"),
    reports_root: Path = typer.Option(Path("reports"), help="Reports root when --report-dir not given"),
):
    """Build an LLM review prompt from a backtest report.

    Output is a single prompt that can be pasted into any LLM (Claude, GPT,
    DeepSeek, etc.). The LLM returns structured JSON suggestions. Consuming
    those suggestions is a manual human workflow (see architecture § 3.5).
    """
    configure_logger()
    from rabbit_hunter.ai_review import load_report_bundle, build_review_prompt

    if report_dir is None:
        # Default: pick the newest report subdir
        candidates = [d for d in reports_root.iterdir() if d.is_dir()]
        if not candidates:
            raise typer.BadParameter(f"No reports found under {reports_root}")
        report_dir = max(candidates, key=lambda d: d.stat().st_mtime)
        typer.echo(f"# Using latest report: {report_dir}")

    bundle = load_report_bundle(report_dir)
    prompt = build_review_prompt(bundle)

    if out_file:
        out_file.write_text(prompt, encoding="utf-8")
        typer.echo(f"Prompt written to {out_file}")
    else:
        typer.echo(prompt)


@ml_app.command("train")
def ml_train(
    report_dir: Path = typer.Argument(None, help="Backtest report to use as training data (default: latest)"),
    output_root: Path = typer.Option(Path("models"), "--out", help="Where to save the trained model"),
    reports_root: Path = typer.Option(Path("reports"), help="Reports root when --report-dir not given"),
    train_fraction: float = typer.Option(0.7, help="Fraction for walk-forward train split"),
    model_type: str = typer.Option("logistic", "--model", help="Model type: 'logistic' or 'lightgbm'"),
):
    """Train a logistic regression scoring model from a backtest's trades.parquet.

    Produces a versioned model file under models/ that MLScoring can load
    at inference time. Architecture spec § 4.1: models are versioned and
    training is offline; inference NEVER retrains.
    """
    configure_logger()
    log = get_logger("cli.ml_train")

    if report_dir is None:
        candidates = [d for d in reports_root.iterdir() if d.is_dir() and (d / "trades.parquet").exists()]
        if not candidates:
            raise typer.BadParameter(f"No backtest reports with trades.parquet found under {reports_root}")
        report_dir = max(candidates, key=lambda d: d.stat().st_mtime)
        typer.echo(f"# Using latest report: {report_dir}")

    trades_path = report_dir / "trades.parquet"
    if not trades_path.exists():
        raise typer.BadParameter(f"No trades.parquet in {report_dir}")

    trades = pd.read_parquet(trades_path)
    typer.echo(f"# Loaded {len(trades)} trades from {trades_path}")

    from rabbit_hunter.ml.training import train_model

    output_root.mkdir(parents=True, exist_ok=True)
    _, result, model_path = train_model(
        trades=trades,
        output_root=output_root,
        train_fraction=train_fraction,
        model_type=model_type,
    )
    typer.echo(f"# Trained model v{result.model_version}")
    typer.echo(f"# n_train={result.n_train}, n_test={result.n_test}")
    typer.echo(f"# Train AUC={result.train_auc:.3f}, Test AUC={result.test_auc:.3f}")
    typer.echo(f"# Train Acc={result.train_accuracy:.3f}, Test Acc={result.test_accuracy:.3f}")
    typer.echo(f"# Saved to {model_path}")
    log.info("ml_train_done",
             version=result.model_version,
             train_auc=result.train_auc,
             test_auc=result.test_auc,
             n_train=result.n_train,
             n_test=result.n_test,
             model_path=str(model_path))


if __name__ == "__main__":
    app()
