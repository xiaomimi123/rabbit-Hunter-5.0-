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
shadow_app = typer.Typer(help="Shadow-mode (paper trading) commands")
live_app = typer.Typer(help="Live-execution commands (Phase 5)")
config_app = typer.Typer(help="Config version-history commands")
app.add_typer(shadow_app, name="shadow")
app.add_typer(live_app, name="live")
app.add_typer(config_app, name="config")


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

    # Auto-snapshot the config used for this run. Idempotent — subsequent
    # backtest runs with the same config are no-ops. Change → new snapshot.
    try:
        from rabbit_hunter.config.history import snapshot_if_changed
        entry = snapshot_if_changed(config, note="backtest")
        if entry is not None:
            log.info("config_snapshot_new",
                      hash=entry.config_hash, path=entry.snapshot_path)
    except Exception as e:
        # Never let history bookkeeping break a real backtest run.
        log.warning("config_snapshot_failed", error=str(e))

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


@ml_app.command("list")
def ml_list(
    models_root: Path = typer.Option(Path("models"), "--models-root"),
    ml_config: Path = typer.Option(Path("configs/strategies/ml_scoring.yaml"),
                                    "--ml-config"),
):
    """List every trained model with its test AUC + active marker."""
    from rabbit_hunter.ml.registry import list_models, get_active_model
    active = get_active_model(ml_config)
    models = list_models(models_root, active_path=active)
    if not models:
        typer.echo(f"No models under {models_root}")
        return
    typer.echo(f"{'*':<2} {'Version':<20} {'Test AUC':<10} "
               f"{'Train AUC':<10} {'n_train':<10} Trained")
    typer.echo("-" * 80)
    for m in models:
        marker = "*" if m.is_active else " "
        typer.echo(f"{marker:<2} {m.version:<20} "
                   f"{m.test_auc:<10.4f} {m.train_auc:<10.4f} "
                   f"{m.n_train:<10} {m.trained_at}")


@ml_app.command("promote")
def ml_promote(
    model_path: Path = typer.Argument(..., help="Path to model.pkl to activate"),
    ml_config: Path = typer.Option(Path("configs/strategies/ml_scoring.yaml"),
                                    "--ml-config"),
):
    """Point MLScoring at a specific model.pkl (atomic YAML rewrite +
    records the previous active so `rabbit ml rollback` works)."""
    from rabbit_hunter.ml.registry import promote
    if not model_path.exists():
        typer.echo(f"no such model: {model_path}")
        raise typer.Exit(code=1)
    prev = promote(ml_config, model_path)
    typer.echo(f"promoted: {model_path}")
    typer.echo(f"previous: {prev or '(none)'}")


@ml_app.command("rollback")
def ml_rollback(
    ml_config: Path = typer.Option(Path("configs/strategies/ml_scoring.yaml"),
                                    "--ml-config"),
):
    """Restore the previously-active model (the one active before the
    last `rabbit ml promote` or `rabbit ml retrain`)."""
    from rabbit_hunter.ml.registry import rollback
    try:
        restored = rollback(ml_config)
    except FileNotFoundError as e:
        typer.echo(f"rollback failed: {e}")
        raise typer.Exit(code=1)
    typer.echo(f"restored: {restored}")


@ml_app.command("retrain")
def ml_retrain(
    source: str = typer.Option("backtest", "--source",
                                help="'backtest' or 'shadow'"),
    report_dir: Path = typer.Option(None, "--report-dir",
                                      help="Backtest report (default: latest)"),
    reports_root: Path = typer.Option(Path("reports"), "--reports-root"),
    state_dir: Path = typer.Option(Path("shadows"), "--state-dir",
                                    help="Shadow state root (when --source shadow)"),
    models_root: Path = typer.Option(Path("models"), "--models-root"),
    ml_config: Path = typer.Option(Path("configs/strategies/ml_scoring.yaml"),
                                    "--ml-config"),
    model_type: str = typer.Option("lightgbm", "--model-type"),
    promote_margin: float = typer.Option(
        0.01, "--promote-margin",
        help="New AUC must beat prior by ≥ this to auto-promote"),
    min_test_auc: float = typer.Option(
        0.52, "--min-test-auc",
        help="Absolute floor — reject even if there's no prior"),
    train_fraction: float = typer.Option(0.7, "--train-fraction"),
    fail_on_reject: bool = typer.Option(False, "--fail-on-reject",
                                         help="Exit non-zero when candidate rejected"),
):
    """Retrain the ML model from backtest or shadow data. Auto-promotes
    the candidate if its test AUC beats the prior by --promote-margin.

    Meant for cron / CI. Every run — promoted or rejected — writes an
    audit entry to models/retrain_log.jsonl.
    """
    from rabbit_hunter.ml.retraining import (
        RetrainConfig, retrain,
        load_trades_from_backtest, load_trades_from_shadow,
    )
    if source == "backtest":
        if report_dir is None:
            candidates = [
                d for d in reports_root.iterdir()
                if d.is_dir() and (d / "trades.parquet").exists()
            ]
            if not candidates:
                typer.echo(f"no backtest reports under {reports_root}")
                raise typer.Exit(code=1)
            report_dir = max(candidates, key=lambda d: d.stat().st_mtime)
            typer.echo(f"# latest report: {report_dir}")
        trades = load_trades_from_backtest(report_dir)
    elif source == "shadow":
        trades = load_trades_from_shadow(state_dir)
        if trades.empty:
            typer.echo(f"no shadow trades in {state_dir}")
            raise typer.Exit(code=1)
    else:
        raise typer.BadParameter(f"source must be 'backtest' or 'shadow'")

    typer.echo(f"# training on {len(trades)} trades from {source}")

    outcome = retrain(
        trades=trades,
        models_root=models_root,
        ml_config_path=ml_config,
        cfg=RetrainConfig(
            promote_margin_auc=promote_margin,
            min_test_auc=min_test_auc,
            train_fraction=train_fraction,
            model_type=model_type,
        ),
    )
    d = outcome.decision
    typer.echo(f"decision: {d.action}")
    typer.echo(f"  candidate test_auc={d.candidate_test_auc:.4f}")
    if d.prior_test_auc is not None:
        typer.echo(f"  prior     test_auc={d.prior_test_auc:.4f}")
    typer.echo(f"  reason:   {d.reason}")
    typer.echo(f"# audit log: {models_root / 'retrain_log.jsonl'}")

    if fail_on_reject and not d.action.startswith(("promote", "no_prior")):
        raise typer.Exit(code=1)


@shadow_app.command("run")
def shadow_run(
    config: Path = typer.Option(Path("configs/default.yaml")),
    state_dir: Path = typer.Option(Path("shadows"), help="Where ledger+snapshots persist"),
    poll_interval: int = typer.Option(60, help="Seconds between OKX poll ticks"),
    lookback_bars: int = typer.Option(600, help="Bars back to fetch per tick"),
):
    """Run the shadow-mode paper-trading loop.

    Fetches K-lines from OKX every poll_interval seconds, runs the same
    Feature Engine → Scoring → Router → Risk → PortfolioRisk pipeline as
    backtest, and records simulated fills to `state_dir/`. Never touches
    a real exchange order endpoint.

    Ctrl-C to stop — state (ledger + last-seen-ts) persists across runs.
    """
    configure_logger()
    cfg = load_config(config)

    # Auto-snapshot the config on shadow launch so mid-run drift is caught.
    try:
        from rabbit_hunter.config.history import snapshot_if_changed
        snapshot_if_changed(config, note="shadow_run")
    except Exception:
        pass  # never crash a real runner on bookkeeping

    from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
    from rabbit_hunter.scoring_engine.strategies.mean_reversion import MeanReversion, MRParams
    from rabbit_hunter.scoring_engine.strategies.price_action import PriceAction, PAParams
    from rabbit_hunter.ml.ml_scoring import MLScoring, MLScoringParams
    from rabbit_hunter.shadow import ShadowRunner, ShadowConfig
    import yaml as _yaml

    _STRATEGY_REGISTRY = {
        "trend_following": (TrendFollowing, TFParams),
        "mean_reversion": (MeanReversion, MRParams),
        "price_action": (PriceAction, PAParams),
        "ml_scoring": (MLScoring, MLScoringParams),
    }
    strategies = []
    for name, entry in cfg.strategy_router.enabled_strategies.items():
        if name not in _STRATEGY_REGISTRY:
            raise typer.BadParameter(f"Unknown strategy {name}")
        Strat, ParamsCls = _STRATEGY_REGISTRY[name]
        strat_cfg_path = Path("configs") / entry.config_file
        strat_yaml = _yaml.safe_load(strat_cfg_path.read_text(encoding="utf-8"))
        strategies.append(Strat(ParamsCls(**strat_yaml["params"])))

    runner = ShadowRunner(
        app_config=cfg, strategies=strategies,
        shadow_config=ShadowConfig(
            poll_interval_seconds=poll_interval,
            lookback_bars=lookback_bars,
            state_dir=state_dir,
        ),
    )
    runner.run_forever()


@shadow_app.command("status")
def shadow_status(
    state_dir: Path = typer.Option(Path("shadows"), help="State root to inspect"),
):
    """Show current ledger + last-seen-ts state. Read-only."""
    import pickle as _pickle
    import json as _json

    ledger_p = state_dir / "state" / "ledger.pkl"
    lastseen_p = state_dir / "state" / "last_seen_ts.json"

    if not ledger_p.exists():
        typer.echo(f"No ledger state at {ledger_p}. Has `rabbit shadow run` been started?")
        raise typer.Exit(code=1)

    with ledger_p.open("rb") as f:
        ledger = _pickle.load(f)
    typer.echo(f"Ledger loaded from: {ledger_p}")
    typer.echo(f"  initial_capital: ${ledger.initial_capital:,.2f}")
    typer.echo(f"  equity:          ${ledger.equity:,.2f}")
    ret_pct = (ledger.equity / ledger.initial_capital - 1) * 100
    typer.echo(f"  return_pct:      {ret_pct:+.2f}%")
    typer.echo(f"  open_positions:  {len(ledger.open_positions)}")
    for sym, pos in ledger.open_positions.items():
        typer.echo(f"    - {sym}: {pos.side} size={pos.size:.4f} @ ${pos.entry_price:,.2f}, "
                   f"stop=${pos.stop:,.2f}, tp=${pos.take_profit:,.2f}")
    typer.echo(f"  closed_trades:   {len(ledger.closed_trades)}")

    if lastseen_p.exists():
        seen = _json.loads(lastseen_p.read_text(encoding="utf-8"))
        typer.echo(f"Last-seen timestamps:")
        for sym, ts in seen.items():
            dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat()
            typer.echo(f"  {sym}: {dt}")


@data_app.command("health")
def data_health(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
    intervals: str = typer.Option("1H",
                                   help="Comma-separated intervals to check"),
    grace_bars: int = typer.Option(2,
                                    help="Freshness grace period in bars"),
    fail_on_unhealthy: bool = typer.Option(False, "--fail-on-unhealthy",
                                            help="Exit non-zero if any check fails"),
):
    """Scan the local data directory for freshness / gap / corruption issues.

    Prints a per-(symbol, interval) status table. With --fail-on-unhealthy,
    exits non-zero when any check is not "healthy" — usable in cron / CI.
    """
    from rabbit_hunter.data_engine.health import check_all, summarize
    cfg = load_config(config)
    interval_list = [x.strip() for x in intervals.split(",") if x.strip()]
    reports = check_all(
        root=data_root, symbols=list(cfg.data.symbols),
        intervals=interval_list, freshness_grace_bars=grace_bars,
    )
    header = ("Symbol", "Interval", "Status", "Rows", "Last ts", "Missing", "Max gap (h)")
    typer.echo("  ".join(f"{h:<18}" for h in header))
    for r in reports:
        row = r.to_row()
        last_ts = row["last_ts"][:19] if row["last_ts"] else ""
        typer.echo(f"{row['symbol']:<18}  {row['interval']:<18}  "
                   f"{row['status']:<18}  {row['rows']:<18}  "
                   f"{last_ts:<18}  {row['missing_bars']:<18}  "
                   f"{row['max_gap_hours']}")
        for p in r.problems:
            typer.echo(f"    - {p}")
    stats = summarize(reports)
    typer.echo(f"\nsummary: {stats['healthy']}/{stats['total']} healthy "
               f"({stats['by_status']})")
    if fail_on_unhealthy and stats["unhealthy"] > 0:
        raise typer.Exit(code=1)


@app.command("baseline-features")
def baseline_features_cmd(
    config: Path = typer.Option(Path("configs/default.yaml")),
    data_root: Path = typer.Option(Path("data")),
    out: Path = typer.Option(Path("baselines/features_latest.json")),
    tag: str = typer.Option("baseline", "--tag"),
):
    """Snapshot per-feature distributions (mean/std/quantiles) over the
    config's backtest window and symbols. Later `rabbit shadow
    feature-drift` compares live feature distributions to this file to
    detect regime changes BEFORE they show up in trade outcomes.
    """
    from rabbit_hunter.data_engine.storage import read_ohlcv
    from rabbit_hunter.feature_engine.pipeline import load_or_compute_features
    from rabbit_hunter.analytics.feature_stability import (
        build_baseline_from_features, save,
    )
    cfg = load_config(config)
    end_ms = _iso_to_ms(cfg.backtest.end)
    start_ms = _iso_to_ms(cfg.backtest.start)

    frames: list[pd.DataFrame] = []
    for symbol in cfg.data.symbols:
        def _raw(sym=symbol):
            df = read_ohlcv(data_root, sym, cfg.data.main_interval,
                            start_ms, end_ms)
            fr_path = data_root / "raw" / "okx" / sym / "funding.parquet"
            if fr_path.exists():
                fr = pd.read_parquet(fr_path)
                df = pd.merge_asof(df.sort_values("timestamp"),
                                    fr.sort_values("timestamp"),
                                    on="timestamp", direction="backward")
            oi_path = data_root / "raw" / "okx" / sym / "oi.parquet"
            if oi_path.exists():
                oi = pd.read_parquet(oi_path)
                df = pd.merge_asof(df.sort_values("timestamp"),
                                    oi.sort_values("timestamp"),
                                    on="timestamp", direction="backward")
            return df
        def _confirm(sym=symbol):
            return read_ohlcv(data_root, sym, cfg.data.confirm_interval,
                              start_ms, end_ms)
        feats = load_or_compute_features(
            root=data_root, symbol=symbol,
            interval=cfg.data.main_interval,
            engine_version=cfg.feature_engine.version,
            fetch_raw=_raw, fetch_confirm=_confirm,
        )
        frames.append(feats)
    combined = pd.concat(frames, ignore_index=True)
    snap = build_baseline_from_features(
        combined, tag=tag,
        source=f"{cfg.data.symbols} @ {cfg.backtest.start}..{cfg.backtest.end}",
    )
    save(snap, out)
    typer.echo(f"features baseline: {out} "
               f"(features={len(snap.features)}, rows_used={len(combined)})")


@shadow_app.command("feature-drift")
def shadow_feature_drift(
    baseline_path: Path = typer.Option(
        Path("baselines/features_latest.json"),
        help="Feature baseline JSON to compare against"),
    state_dir: Path = typer.Option(Path("shadows"),
                                    help="Shadow state root"),
    mean_shift_sigmas: float = typer.Option(
        2.0, help="|Δμ|/σ threshold that triggers"),
    std_ratio: float = typer.Option(
        2.0, help="σ_live/σ_baseline band threshold"),
    min_samples: int = typer.Option(
        200, help="Min live samples per feature"),
    fail_on_alert: bool = typer.Option(False, "--fail-on-alert"),
):
    """Compare live shadow feature distributions to a backtest baseline.

    Reads state/features_log.parquet (written on every processed bar),
    then per-feature: |Δμ|/σ ≥ threshold or σ_ratio outside band → alert.
    Cron this alongside `rabbit shadow drift` — feature drift often
    precedes trade drift by hours or days.
    """
    from rabbit_hunter.analytics.feature_stability import (
        load, compare, FeatureStabilityThresholds,
    )
    if not baseline_path.exists():
        typer.echo(f"no baseline at {baseline_path}")
        raise typer.Exit(code=2)
    log_path = state_dir / "state" / "features_log.parquet"
    if not log_path.exists():
        typer.echo(f"no features_log at {log_path}")
        raise typer.Exit(code=2)
    baseline = load(baseline_path)
    live = pd.read_parquet(log_path)
    report = compare(
        baseline, live,
        FeatureStabilityThresholds(
            mean_shift_sigmas=mean_shift_sigmas,
            std_ratio_alert=std_ratio,
            min_live_samples=min_samples,
        ),
    )
    for line in report.as_lines():
        typer.echo(line)
    if fail_on_alert and not report.ok:
        raise typer.Exit(code=1)


@app.command("baseline")
def baseline_cmd(
    report_dir: Path = typer.Argument(..., help="Backtest report to snapshot"),
    out: Path = typer.Option(Path("baselines/latest.json"),
                              help="Output JSON path"),
    tag: str = typer.Option("baseline", "--tag",
                             help="Tag identifier (e.g. v0.1.3)"),
):
    """Snapshot per-cluster baseline stats from a backtest report.

    Later `rabbit shadow drift` compares live shadow performance against
    this baseline. Baselines are checked into git (JSON, not pickle) so
    they diff cleanly across strategy versions.
    """
    from rabbit_hunter.analytics.cluster_performance import analyze
    from rabbit_hunter.analytics.baseline import (
        build_baseline_from_report, save,
    )
    trades_path = report_dir / "trades.parquet"
    if not trades_path.exists():
        typer.echo(f"no trades.parquet at {trades_path}")
        raise typer.Exit(code=1)
    df = pd.read_parquet(trades_path)
    report = analyze(df, schema="backtest")
    snap = build_baseline_from_report(
        report, tag=tag, source_report_dir=str(report_dir),
    )
    save(snap, out)
    typer.echo(f"baseline: {out} (clusters={len(snap.clusters)}, "
               f"total_trades={snap.total_trades})")


@shadow_app.command("drift")
def shadow_drift(
    baseline_path: Path = typer.Option(Path("baselines/latest.json"),
                                        help="Baseline JSON to compare against"),
    state_dir: Path = typer.Option(Path("shadows"), help="Shadow state root"),
    winrate_delta_alert: float = typer.Option(0.15,
                                                help="|Δ WR| pp that triggers"),
    zscore_alert: float = typer.Option(2.5,
                                        help="|z| of avg PnL that triggers"),
    min_live_trades: int = typer.Option(20,
                                          help="Min live n before a cluster can trigger"),
    fail_on_alert: bool = typer.Option(False, "--fail-on-alert",
                                        help="Exit non-zero if drift detected"),
):
    """Check whether live shadow performance has drifted from the baseline.

    Loads baseline.json, runs cluster analysis on the shadow ledger, and
    reports per-cluster Δ WR and z-score of avg PnL. Meant for cron —
    a non-zero exit becomes a page.
    """
    import pickle
    from rabbit_hunter.analytics.baseline import load
    from rabbit_hunter.analytics.cluster_performance import analyze
    from rabbit_hunter.analytics.drift import compare, DriftThresholds

    if not baseline_path.exists():
        typer.echo(f"no baseline at {baseline_path}")
        raise typer.Exit(code=2)
    ledger_p = state_dir / "state" / "ledger.pkl"
    if not ledger_p.exists():
        typer.echo(f"no ledger at {ledger_p}")
        raise typer.Exit(code=2)

    baseline = load(baseline_path)
    with ledger_p.open("rb") as f:
        ledger = pickle.load(f)
    if not ledger.closed_trades:
        typer.echo("No shadow closed trades yet — nothing to compare.")
        raise typer.Exit(code=0)
    live = analyze(pd.DataFrame(ledger.closed_trades), schema="shadow")
    report = compare(baseline, live, DriftThresholds(
        winrate_delta_alert=winrate_delta_alert,
        avg_pnl_zscore_alert=zscore_alert,
        min_live_trades=min_live_trades,
    ))
    for line in report.as_lines():
        typer.echo(line)
    if fail_on_alert and not report.ok:
        raise typer.Exit(code=1)


@app.command()
def clusters(
    report_dir: Path = typer.Argument(..., help="Backtest report dir (contains trades.parquet)"),
    out: Path = typer.Option(None, "--out",
                              help="Write markdown report here (else stdout)"),
):
    """Per-cluster performance breakdown of a backtest.

    Classifies every trade into one of six cluster (momentum_breakdown,
    momentum_breakout, range_breakout, trend_continuation, trend_reversal,
    other) using entry-time features, then aggregates winrate / PF /
    Sharpe / etc per cluster.
    """
    from rabbit_hunter.analytics.cluster_performance import analyze, render_markdown
    trades_path = report_dir / "trades.parquet"
    if not trades_path.exists():
        typer.echo(f"no trades.parquet at {trades_path}")
        raise typer.Exit(code=1)
    df = pd.read_parquet(trades_path)
    md = render_markdown(analyze(df, schema="backtest"))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"clusters: {out}")
    else:
        typer.echo(md)


@shadow_app.command("performance")
def shadow_performance(
    state_dir: Path = typer.Option(Path("shadows"), help="State root"),
    out: Path = typer.Option(None, "--out",
                              help="Write markdown here (else stdout)"),
):
    """Per-cluster performance breakdown of shadow-mode closed trades.

    Reads state/ledger.pkl and classifies each closed_trade by its
    entry_snapshot, then aggregates. Useful for spotting concept drift
    — e.g. Cluster-1 momentum_breakdown had 62% WR in backtest; if
    shadow shows 40%, something changed.
    """
    import pickle
    from rabbit_hunter.analytics.cluster_performance import analyze, render_markdown
    ledger_p = state_dir / "state" / "ledger.pkl"
    if not ledger_p.exists():
        typer.echo(f"no ledger at {ledger_p}")
        raise typer.Exit(code=1)
    with ledger_p.open("rb") as f:
        ledger = pickle.load(f)
    if not ledger.closed_trades:
        typer.echo("No closed trades in shadow ledger yet.")
        raise typer.Exit(code=0)
    df = pd.DataFrame(ledger.closed_trades)
    md = render_markdown(analyze(df, schema="shadow"))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"performance: {out}")
    else:
        typer.echo(md)


@app.command()
def compare(
    reports: list[Path] = typer.Argument(..., help="Two or more report dirs"),
    label: list[str] = typer.Option(None, "--label", "-l",
                                     help="Override display names (order = reports order)"),
    out_md: Path = typer.Option(None, "--out-md",
                                 help="Write markdown to this file"),
    out_html: Path = typer.Option(None, "--out-html",
                                   help="Write HTML comparison to this file"),
    initial_capital: float = typer.Option(10_000.0, help="Base for DD %"),
    summary_only: bool = typer.Option(False, "--summary-only",
                                        help="Markdown: emit only the summary metrics table"),
    top_trades: int = typer.Option(20, "--top-trades",
                                     help="Markdown: number of top-|PnL| trades to list"),
):
    """Side-by-side comparison of ≥2 backtest reports.

    Reads trades.parquet from each report dir, computes standard metrics,
    and emits both formats:
      - Markdown (default): summary metrics + per-symbol breakdown +
        top-N trades. LLM-friendly, git-diffable, no rendering required.
      - HTML (--out-html): same content plus interactive sort/filter.
    """
    from rabbit_hunter.backtest.compare import (
        compare as _compare, render_markdown, render_html,
        _compute_metrics,
    )
    if len(reports) < 2:
        raise typer.BadParameter("need at least 2 report directories")
    if label and len(label) != len(reports):
        raise typer.BadParameter(
            f"--label count ({len(label)}) must match reports count ({len(reports)})"
        )
    labels = label or [d.name for d in reports]
    metrics = [
        _compute_metrics(l, d, initial_capital=initial_capital)
        for l, d in zip(labels, reports)
    ]
    md = render_markdown(metrics,
                          include_sections=not summary_only,
                          top_trades_n=top_trades)
    html_out = render_html(metrics)
    if out_md:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        typer.echo(f"markdown: {out_md}")
    else:
        typer.echo(md)
    if out_html:
        out_html.parent.mkdir(parents=True, exist_ok=True)
        out_html.write_text(html_out, encoding="utf-8")
        typer.echo(f"html: {out_html}")


@live_app.command("reconcile")
def live_reconcile(
    config: Path = typer.Option(Path("configs/default.yaml")),
    state_dir: Path = typer.Option(Path("shadows"),
                                    help="State root to load ledger from"),
    fail_on_mismatch: bool = typer.Option(False, "--fail-on-mismatch",
                                           help="Exit non-zero if drift found"),
):
    """Compare the shadow ledger's open positions against exchange state.

    Read-only. Never modifies the ledger or places orders. Meant to run in
    cron / watchdog so drift is caught before the next entry decision.
    """
    import pickle
    from rabbit_hunter.execution_engine.live_executor import LiveExecutor
    from rabbit_hunter.execution_engine.reconciliation import reconcile_positions
    cfg = load_config(config)
    ledger_p = state_dir / "state" / "ledger.pkl"
    if not ledger_p.exists():
        typer.echo(f"no ledger at {ledger_p}")
        raise typer.Exit(code=2)
    with ledger_p.open("rb") as f:
        ledger = pickle.load(f)
    executor = LiveExecutor(cfg.execution, cfg.live_execution)
    exchange_positions = executor.fetch_exchange_positions()
    report = reconcile_positions(ledger.open_positions, exchange_positions)
    for line in report.as_lines():
        typer.echo(line)
    if fail_on_mismatch and not report.ok:
        raise typer.Exit(code=1)


@shadow_app.command("watchdog")
def shadow_watchdog(
    state_dir: Path = typer.Option(Path("shadows"),
                                    help="State root the runner writes to"),
    max_silence_seconds: float = typer.Option(300.0,
                                               help="Alert when last tick older than this"),
):
    """Check that the shadow runner is still ticking.

    Exits 0 = HEALTHY, 1 = STALE, 2 = DOWN. Meant for cron / alerting
    integrations — a wrapper script decides what happens on non-zero.
    """
    from rabbit_hunter.shadow.watchdog import check
    result = check(state_dir=state_dir,
                   max_silence_seconds=max_silence_seconds)
    typer.echo(result.as_line())
    if result.status == "healthy":
        raise typer.Exit(code=0)
    if result.status == "stale":
        raise typer.Exit(code=1)
    raise typer.Exit(code=2)


@shadow_app.command("digest")
def shadow_digest(
    state_dir: Path = typer.Option(Path("shadows"),
                                    help="State root to summarize"),
    trade_baseline: Path = typer.Option(
        None, "--trade-baseline",
        help="baselines/*.json for trade-outcome drift"),
    feature_baseline: Path = typer.Option(
        None, "--feature-baseline",
        help="baselines/features_*.json for feature-distribution drift"),
    out: Path = typer.Option(None, "--out",
                              help="Write markdown here (else stdout)"),
):
    """One-shot daily digest: health + cluster performance + trade drift +
    feature drift + recent alerts. Meant for cron; pipe to Slack/email.
    """
    from rabbit_hunter.shadow.digest import render
    md = render(state_dir=state_dir,
                trade_baseline_path=trade_baseline,
                feature_baseline_path=feature_baseline)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        typer.echo(f"digest: {out}")
    else:
        typer.echo(md)


@shadow_app.command("dashboard")
def shadow_dashboard(
    state_dir: Path = typer.Option(Path("shadows"), help="State root to render"),
    out: Path = typer.Option(Path("shadow_dashboard.html"),
                             help="Path to write the HTML file"),
):
    """Render a self-contained HTML dashboard from the shadow state.

    Reads state/ledger.pkl + state/metrics_history.parquet + recent daily
    snapshots.parquet and emits ONE HTML file (no external CSS/JS/CDN).
    Open with `open <path>` locally, or scp off a server.
    """
    from rabbit_hunter.shadow.dashboard import write_dashboard
    p = write_dashboard(state_dir=state_dir, out_path=out)
    typer.echo(f"dashboard: {p}")


@config_app.command("history")
def config_history(
    source: Path = typer.Option(None, "--source",
                                  help="Filter to a specific config file"),
    history_dir: Path = typer.Option(
        Path("configs/.history"), "--history-dir"),
    limit: int = typer.Option(0, "--limit",
                                help="Show only the last N entries (0 = all)"),
):
    """List recorded config snapshots — one line per change."""
    from rabbit_hunter.config.history import history as _hist
    entries = _hist(history_dir=history_dir, source_path=source)
    if not entries:
        typer.echo(f"no history under {history_dir}")
        return
    if limit > 0:
        entries = entries[-limit:]
    typer.echo(f"{'Idx':<5} {'Hash':<18} {'Timestamp (UTC)':<28} "
                f"{'Source':<40} Note")
    typer.echo("-" * 110)
    for i, e in enumerate(entries):
        typer.echo(f"{i:<5} {e.config_hash:<18} {e.timestamp_utc:<28} "
                    f"{e.source_path:<40} {e.note}")


@config_app.command("diff")
def config_diff(
    rev_a: str = typer.Argument("previous"),
    rev_b: str = typer.Argument("latest"),
    history_dir: Path = typer.Option(
        Path("configs/.history"), "--history-dir"),
    source: Path = typer.Option(None, "--source",
                                  help="Filter to a specific config file"),
):
    """Show diff between two snapshots. Revisions: `latest`, `previous`,
    a hash-prefix, or an integer index (see `rabbit config history`)."""
    from rabbit_hunter.config.history import diff as _diff
    try:
        text = _diff(rev_a, rev_b,
                      history_dir=history_dir, source_path=source)
    except ValueError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1)
    if not text:
        typer.echo("(revisions identical)")
        return
    typer.echo(text)


@config_app.command("snapshot")
def config_snapshot(
    config: Path = typer.Argument(Path("configs/default.yaml")),
    history_dir: Path = typer.Option(
        Path("configs/.history"), "--history-dir"),
    note: str = typer.Option("", "--note",
                              help="Free-text label for this snapshot"),
):
    """Force-snapshot a config file NOW (bypassing the change check).

    Useful before an experiment: `rabbit config snapshot --note pre-tune`
    then run the tuning, then `rabbit config diff pre-tune latest`.
    """
    from rabbit_hunter.config.history import snapshot_if_changed
    entry = snapshot_if_changed(config, history_dir=history_dir, note=note)
    if entry is None:
        typer.echo("no change — snapshot skipped (idempotent)")
    else:
        typer.echo(f"snapshot: {entry.snapshot_path}")
        typer.echo(f"hash:     {entry.config_hash}")


@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port"),
    root: Path = typer.Option(Path("."), "--root",
                                help="Project root — where reports/, shadows/, "
                                     "models/, baselines/, configs/ live"),
    reload: bool = typer.Option(False, "--reload",
                                  help="Enable uvicorn auto-reload (dev only)"),
):
    """Start the operator console — HTTP API + dark-themed SPA at http://host:port/

    Reads state from the mounted project tree (reports/, shadows/, models/,
    baselines/, configs/). Read-only — mutations still happen via CLI.
    """
    import uvicorn
    from rabbit_hunter.web.app import create_app
    from rabbit_hunter.web.paths import Paths
    web_app = create_app(Paths(root=root))
    typer.echo(f"→ http://{host}:{port}  (root={root.resolve()})")
    uvicorn.run(web_app, host=host, port=port, reload=reload,
                log_level="info", access_log=False)


if __name__ == "__main__":
    app()
