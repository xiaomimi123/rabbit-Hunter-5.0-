"""Shadow daily digest — one markdown report summarizing the current
shadow-mode state across every analytic we have.

Combines:
  - Runtime health (last tick, uptime signal)
  - Live PnL / equity / drawdown / open positions
  - Per-cluster performance (analytics/cluster_performance)
  - Trade-outcome drift vs baseline (analytics/drift), if baseline provided
  - Feature-distribution drift vs baseline (analytics/feature_stability),
    if features baseline + features_log provided
  - Recent alerts from metrics_history

Meant to run in cron once/day. The stdout output is fine for logs; a
wrapper script can pipe the markdown to Slack / Discord / email / SMS.

If any input is missing (no baseline, no features_log, no ledger yet)
the corresponding section is muted with a one-line note — the digest
never crashes on partial inputs.
"""
from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _load_ledger(state_dir: Path):
    p = state_dir / "state" / "ledger.pkl"
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


def _load_metrics_history(state_dir: Path) -> pd.DataFrame:
    p = state_dir / "state" / "metrics_history.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


def _fmt_dt_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) \
        .strftime("%Y-%m-%d %H:%M UTC")


def _section_health(hist: pd.DataFrame) -> list[str]:
    if hist.empty:
        return ["## Runtime health", "", "_No metrics history yet._", ""]
    latest = hist.iloc[-1]
    return [
        "## Runtime health",
        "",
        f"- Last tick: {_fmt_dt_utc(int(latest['timestamp_ms']))}",
        f"- Equity: ${latest['equity']:,.2f}",
        f"- PnL: {latest['total_pnl']:+,.2f} "
        f"({latest['pnl_pct']*100:+.2f}%)",
        f"- Drawdown from peak: {latest['drawdown_from_peak_pct']*100:.2f}%",
        f"- Open positions: {int(latest['open_positions'])}",
        f"- Closed trades: {int(latest['total_closed_trades'])}",
        f"- Consecutive errors: {int(latest['consecutive_errors'])}",
        "",
    ]


def _section_recent_alerts(hist: pd.DataFrame,
                             lookback_hours: int = 24) -> list[str]:
    lines = ["## Recent alerts (last 24h)", ""]
    if hist.empty or "alert_count" not in hist.columns:
        lines.append("_No alerts._")
        lines.append("")
        return lines
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    cutoff = now_ms - lookback_hours * 3_600_000
    recent = hist[(hist["timestamp_ms"] >= cutoff) & (hist["alert_count"] > 0)]
    if recent.empty:
        lines.append("_All clear — no alerts in the last 24h._")
        lines.append("")
        return lines
    for _, r in recent.tail(10).iterrows():
        lines.append(f"- `{_fmt_dt_utc(int(r['timestamp_ms']))}` "
                     f"{r['alerts']}")
    lines.append("")
    return lines


def _section_clusters(ledger) -> list[str]:
    if ledger is None or not ledger.closed_trades:
        return ["## Per-cluster performance", "",
                "_No closed trades to classify yet._", ""]
    from rabbit_hunter.analytics.cluster_performance import (
        analyze, render_markdown,
    )
    df = pd.DataFrame(ledger.closed_trades)
    report = analyze(df, schema="shadow")
    return ["## Per-cluster performance", "",
            render_markdown(report), ""]


def _section_trade_drift(baseline_path: Path | None,
                           ledger) -> list[str]:
    lines = ["## Trade-outcome drift vs baseline", ""]
    if baseline_path is None:
        lines.append("_No trade baseline specified._")
        lines.append("")
        return lines
    if not baseline_path.exists():
        lines.append(f"_Baseline missing at `{baseline_path}`._")
        lines.append("")
        return lines
    if ledger is None or not ledger.closed_trades:
        lines.append("_No shadow closed trades yet — nothing to compare._")
        lines.append("")
        return lines
    from rabbit_hunter.analytics.baseline import load
    from rabbit_hunter.analytics.cluster_performance import analyze
    from rabbit_hunter.analytics.drift import compare, DriftThresholds
    baseline = load(baseline_path)
    live = analyze(pd.DataFrame(ledger.closed_trades), schema="shadow")
    report = compare(baseline, live)
    marker = "OK" if report.ok else "**ALERT**"
    lines.append(f"Status: {marker} — baseline `{baseline.tag}`")
    lines.append("")
    for f in report.findings:
        symbol = "✅" if not f.triggered else "🚨"
        lines.append(
            f"- {symbol} `{f.cluster}` — "
            f"WR {f.baseline_winrate:.1%} → {f.live_winrate:.1%} "
            f"(Δ{f.winrate_delta*100:+.1f}pp, z={f.avg_pnl_zscore:+.2f}, "
            f"n={f.live_n}): {f.reason}"
        )
    for c in report.baseline_only:
        lines.append(f"- ⏳ `{c}` — baseline had this cluster but shadow "
                     f"has 0 trades")
    for c in report.live_only:
        lines.append(f"- ❓ `{c}` — shadow has trades but baseline had 0 "
                     f"(unexpected cluster in live)")
    lines.append("")
    return lines


def _section_feature_drift(baseline_path: Path | None,
                             state_dir: Path) -> list[str]:
    lines = ["## Feature-distribution drift vs baseline", ""]
    if baseline_path is None:
        lines.append("_No feature baseline specified._")
        lines.append("")
        return lines
    if not baseline_path.exists():
        lines.append(f"_Baseline missing at `{baseline_path}`._")
        lines.append("")
        return lines
    log_path = state_dir / "state" / "features_log.parquet"
    if not log_path.exists():
        lines.append(f"_No features_log at `{log_path}` yet._")
        lines.append("")
        return lines
    from rabbit_hunter.analytics.feature_stability import (
        load, compare, FeatureStabilityThresholds,
    )
    baseline = load(baseline_path)
    live = pd.read_parquet(log_path)
    report = compare(baseline, live)
    marker = "OK" if report.ok else "**ALERT**"
    lines.append(f"Status: {marker} — baseline `{baseline.tag}`, "
                 f"live_rows={len(live)}")
    lines.append("")
    for f in report.findings:
        symbol = "✅" if not f.triggered else "🚨"
        lines.append(
            f"- {symbol} `{f.feature}` — "
            f"μ {f.baseline_mean:.4g} → {f.live_mean:.4g} "
            f"(Δ={f.mean_shift_sigmas:+.2f}σ), "
            f"σ_ratio={f.std_ratio:.2f}, n={f.live_n}: {f.reason}"
        )
    for c in report.baseline_only:
        lines.append(f"- ⏳ `{c}` — baseline saw this feature but live has "
                     f"none yet")
    for c in report.live_only:
        lines.append(f"- ❓ `{c}` — live has this feature but baseline "
                     f"never saw it")
    lines.append("")
    return lines


def render(
    state_dir: Path,
    trade_baseline_path: Path | None = None,
    feature_baseline_path: Path | None = None,
) -> str:
    """Render the full digest as one Markdown string."""
    ledger = _load_ledger(state_dir)
    hist = _load_metrics_history(state_dir)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts: list[str] = [
        f"# Rabbit Hunter — Shadow Digest",
        "",
        f"_generated {now} · state_dir={state_dir}_",
        "",
    ]
    parts += _section_health(hist)
    parts += _section_recent_alerts(hist)
    parts += _section_clusters(ledger)
    parts += _section_trade_drift(trade_baseline_path, ledger)
    parts += _section_feature_drift(feature_baseline_path, state_dir)
    return "\n".join(parts)
