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
        fetch_ohlcv, fetch_funding_rate_history, fetch_open_interest_history,
    )
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

        # funding + OI 每个 symbol 只拉一次（1H 时序）
        fr = fetch_funding_rate_history(symbol, start_ms, end_ms)
        oi = fetch_open_interest_history(symbol, start_ms, end_ms)
        (data_root / "raw" / "okx" / symbol).mkdir(parents=True, exist_ok=True)
        fr.to_parquet(data_root / "raw" / "okx" / symbol / "funding.parquet", index=False)
        oi.to_parquet(data_root / "raw" / "okx" / symbol / "oi.parquet", index=False)
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
    from rabbit_hunter.backtest.engine import BacktestEngine
    from rabbit_hunter.backtest.report import ReportBuilder
    from rabbit_hunter.observability.snapshot import SnapshotWriter
    import yaml as _yaml

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

    tf_cfg_path = Path("configs") / cfg.strategy_router.enabled_strategies["trend_following"].config_file
    tf_yaml = _yaml.safe_load(tf_cfg_path.read_text(encoding="utf-8"))
    tf = TrendFollowing(TFParams(**tf_yaml["params"]))

    engine = BacktestEngine(cfg, [tf])
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


if __name__ == "__main__":
    app()
