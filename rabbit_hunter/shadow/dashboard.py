"""Shadow dashboard — self-contained HTML report of the current shadow run.

The dashboard reads three sources from a shadow state_dir:
  - `state/ledger.pkl` — current equity, open positions, closed trades
  - `state/metrics_history.parquet` — the metrics tick-log
  - `YYYY-MM-DD/snapshots.parquet` — decision snapshots for the last few days

...and emits ONE self-contained HTML file with no external assets — no CDN,
no fonts, no <script src>. It's meant to be scp'd off a server or opened
via `open shadow.html`, not served.

Charts are rendered as inline SVG. Simple, cheap, no runtime deps beyond
the stdlib + pandas.
"""
from __future__ import annotations

import html
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Data-loading helpers — each returns None if source is missing so the
# dashboard renders partial info instead of crashing.
# ============================================================

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


def _load_recent_snapshots(state_dir: Path, days: int = 3) -> pd.DataFrame:
    """Load snapshots from the last N daily parquet files."""
    day_dirs = sorted(
        (d for d in state_dir.iterdir()
         if d.is_dir() and d.name != "state" and len(d.name) == 10),
        reverse=True,
    )[:days]
    frames = []
    for d in day_dirs:
        p = d / "snapshots.parquet"
        if not p.exists():
            continue
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ============================================================
# SVG chart — one function per chart. Kept dumb: no axes ticks
# beyond min/max labels, no legend. The point is a glance value.
# ============================================================

def _svg_line_chart(
    values: list[float],
    width: int = 720,
    height: int = 180,
    stroke: str = "#0a7d34",
    fill: str = "#0a7d3422",
    label_prefix: str = "",
) -> str:
    if len(values) < 2:
        return (f'<svg width="{width}" height="{height}" '
                f'style="background:#f6f6f6;border-radius:6px">'
                f'<text x="12" y="24" fill="#666" font-size="12">'
                f'not enough data</text></svg>')
    y_min = min(values)
    y_max = max(values)
    y_range = y_max - y_min or 1.0
    n = len(values)
    pad = 12
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad
    pts = []
    for i, v in enumerate(values):
        x = pad + (i / (n - 1)) * inner_w
        y = pad + (1 - (v - y_min) / y_range) * inner_h
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    # Filled area under the line
    area = f"{pad},{pad + inner_h} " + poly + f" {pad + inner_w},{pad + inner_h}"
    return f'''<svg width="{width}" height="{height}" style="background:#f6f6f6;border-radius:6px">
  <polygon points="{area}" fill="{fill}" stroke="none"/>
  <polyline points="{poly}" fill="none" stroke="{stroke}" stroke-width="1.5"/>
  <text x="{pad}" y="14" fill="#666" font-size="10">{label_prefix}max {y_max:,.2f}</text>
  <text x="{pad}" y="{height - 4}" fill="#666" font-size="10">{label_prefix}min {y_min:,.2f}</text>
</svg>'''


# ============================================================
# HTML section builders
# ============================================================

def _pct(x: float) -> str:
    return f"{x*100:+.2f}%"


def _fmt(x: float) -> str:
    return f"${x:,.2f}"


def _kpi_card(label: str, value: str, color: str = "#111") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value" style="color:{color}">{html.escape(value)}</div></div>')


def _build_kpi_row(ledger, latest_metrics) -> str:
    if ledger is None:
        return ""
    equity = ledger.equity
    initial = ledger.initial_capital
    pnl = equity - initial
    pnl_pct = pnl / initial if initial else 0.0
    dd = latest_metrics.get("drawdown_from_peak_pct", 0.0) if latest_metrics else 0.0
    n_open = len(ledger.open_positions)
    n_closed = len(ledger.closed_trades)
    equity_color = "#0a7d34" if pnl >= 0 else "#c62828"
    dd_color = "#c62828" if dd >= 0.10 else "#111"

    cards = [
        _kpi_card("Equity", _fmt(equity), equity_color),
        _kpi_card("PnL", f"{_fmt(pnl)} ({_pct(pnl_pct)})", equity_color),
        _kpi_card("Drawdown", _pct(dd), dd_color),
        _kpi_card("Open positions", str(n_open)),
        _kpi_card("Closed trades", str(n_closed)),
    ]
    return f'<div class="kpi-row">{"".join(cards)}</div>'


def _build_alerts_section(hist: pd.DataFrame) -> str:
    if hist.empty or "alerts" not in hist.columns:
        return '<section><h2>Alerts</h2><p class="muted">No metrics history yet.</p></section>'
    with_alerts = hist[hist["alert_count"] > 0]
    if with_alerts.empty:
        return '<section><h2>Alerts</h2><p class="ok">All clear — no alerts in history.</p></section>'
    recent = with_alerts.tail(15).iloc[::-1]
    rows = []
    for _, r in recent.iterrows():
        dt = datetime.fromtimestamp(int(r["timestamp_ms"]) / 1000,
                                     tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rows.append(f"<tr><td>{dt}</td><td>{html.escape(str(r['alerts']))}</td></tr>")
    return f'''<section><h2>Alerts (recent)</h2>
<table>
  <thead><tr><th>Time (UTC)</th><th>Details</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></section>'''


def _build_equity_curve(hist: pd.DataFrame) -> str:
    if hist.empty:
        return ''
    equity_series = hist["equity"].tolist()
    dd_series = (hist["drawdown_from_peak_pct"] * 100).tolist() \
        if "drawdown_from_peak_pct" in hist.columns else []
    equity_svg = _svg_line_chart(equity_series, label_prefix="$")
    dd_svg = _svg_line_chart(dd_series, stroke="#c62828", fill="#c6282822",
                             label_prefix="") if dd_series else ""
    return f'''<section>
<h2>Equity curve ({len(equity_series)} ticks)</h2>
{equity_svg}
<h2 style="margin-top:24px">Drawdown from peak (%)</h2>
{dd_svg}
</section>'''


def _build_positions_table(ledger) -> str:
    if ledger is None or not ledger.open_positions:
        return '<section><h2>Open positions</h2><p class="muted">None.</p></section>'
    rows = []
    for sym, pos in ledger.open_positions.items():
        entry_dt = datetime.fromtimestamp(int(pos.entry_time) / 1000,
                                           tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        rows.append(
            f"<tr><td>{html.escape(sym)}</td><td>{html.escape(pos.side)}</td>"
            f"<td>{pos.size:.4f}</td><td>{_fmt(pos.entry_price)}</td>"
            f"<td>{_fmt(pos.stop)}</td><td>{_fmt(pos.take_profit)}</td>"
            f"<td>{entry_dt}</td></tr>"
        )
    return f'''<section><h2>Open positions</h2>
<table>
  <thead><tr><th>Symbol</th><th>Side</th><th>Size</th><th>Entry</th>
    <th>Stop</th><th>TP</th><th>Opened (UTC)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></section>'''


def _build_recent_trades_table(ledger, n: int = 20) -> str:
    if ledger is None or not ledger.closed_trades:
        return '<section><h2>Recent closed trades</h2><p class="muted">None yet.</p></section>'
    recent = ledger.closed_trades[-n:][::-1]
    rows = []
    for t in recent:
        pnl = t.get("pnl_after_fees", 0.0)
        color = "#0a7d34" if pnl > 0 else "#c62828" if pnl < 0 else "#666"
        exit_dt = datetime.fromtimestamp(int(t.get("exit_time", 0)) / 1000,
                                          tz=timezone.utc).strftime("%Y-%m-%d %H:%M") \
            if t.get("exit_time") else ""
        rows.append(
            f"<tr><td>{html.escape(str(t.get('symbol', '')))}</td>"
            f"<td>{html.escape(str(t.get('side', '')))}</td>"
            f"<td style='color:{color}'><b>{_fmt(pnl)}</b></td>"
            f"<td>{html.escape(str(t.get('exit_reason', '')))}</td>"
            f"<td>{exit_dt}</td></tr>"
        )
    return f'''<section><h2>Recent closed trades ({len(recent)})</h2>
<table>
  <thead><tr><th>Symbol</th><th>Side</th><th>PnL</th>
    <th>Exit reason</th><th>Exit (UTC)</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table></section>'''


def _build_health_section(hist: pd.DataFrame) -> str:
    if hist.empty:
        return ''
    latest = hist.iloc[-1].to_dict()
    ts = datetime.fromtimestamp(int(latest["timestamp_ms"]) / 1000,
                                 tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    last_bar = latest.get("last_bar_ts_ms")
    last_bar_str = (datetime.fromtimestamp(int(last_bar) / 1000,
                     tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    if last_bar and not pd.isna(last_bar) else "n/a")
    return f'''<section><h2>Runtime health</h2>
<table>
  <tr><th>Last tick</th><td>{ts}</td></tr>
  <tr><th>Last processed bar</th><td>{last_bar_str}</td></tr>
  <tr><th>Minutes since last bar</th><td>{latest.get("minutes_since_last_bar") or "n/a"}</td></tr>
  <tr><th>Consecutive errors</th><td>{int(latest.get("consecutive_errors", 0))}</td></tr>
  <tr><th>Winrate</th><td>{latest.get("winrate", 0.0):.2%}</td></tr>
  <tr><th>Profit factor</th><td>{latest.get("profit_factor", 0.0):.2f}</td></tr>
</table></section>'''


# ============================================================
# Top-level renderer
# ============================================================

def render_dashboard(state_dir: Path) -> str:
    ledger = _load_ledger(state_dir)
    hist = _load_metrics_history(state_dir)
    latest_metrics = hist.iloc[-1].to_dict() if not hist.empty else {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    kpi = _build_kpi_row(ledger, latest_metrics)
    alerts = _build_alerts_section(hist)
    curve = _build_equity_curve(hist)
    positions = _build_positions_table(ledger)
    trades = _build_recent_trades_table(ledger)
    health = _build_health_section(hist)

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Rabbit Hunter · Shadow Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #fafafa; color: #111; line-height: 1.4; }}
  header {{ background: #111; color: #fff; padding: 16px 24px; }}
  header h1 {{ margin: 0; font-size: 20px; font-weight: 500; }}
  header .subtitle {{ font-size: 12px; opacity: 0.7; margin-top: 4px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 20px 24px 60px; }}
  section {{ background: #fff; border: 1px solid #eee; border-radius: 8px;
             padding: 20px 24px; margin-bottom: 20px; }}
  section h2 {{ font-size: 15px; margin: 0 0 12px 0; color: #333; font-weight: 600;
                text-transform: uppercase; letter-spacing: 0.05em; }}
  .kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: #fff; border: 1px solid #eee; border-radius: 8px;
          padding: 14px 18px; }}
  .kpi-label {{ font-size: 11px; color: #999; text-transform: uppercase;
                letter-spacing: 0.06em; }}
  .kpi-value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #f0f0f0; }}
  th {{ background: #fafafa; font-weight: 600; color: #666; font-size: 11px;
        text-transform: uppercase; letter-spacing: 0.05em; }}
  .muted {{ color: #999; }}
  .ok    {{ color: #0a7d34; }}
</style>
</head>
<body>
<header>
  <h1>Rabbit Hunter · Shadow Dashboard</h1>
  <div class="subtitle">state_dir: {html.escape(str(state_dir))} · rendered {now}</div>
</header>
<main>
{kpi}
{alerts}
{health}
{curve}
{positions}
{trades}
</main>
</body>
</html>'''


def write_dashboard(state_dir: Path, out_path: Path) -> Path:
    """Render + write. Returns the path written."""
    html_content = render_dashboard(state_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
