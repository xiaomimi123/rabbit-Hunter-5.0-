"""Multi-report backtest comparison — the "am I improving vs my last tag?" tool.

Takes ≥2 report directories (each with trades.parquet), computes standard
metrics per report, and emits:
  - markdown table (deterministic, git-diffable, LLM-readable)
  - side-by-side HTML with an inline SVG equity curve overlay

Two-input version is optimized (shows Δ column). Three+ shows plain columns.
Labels override the report path in headers so a user can pass
`--label v0.1g v0.1.4` to get readable names in the output.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Metric computation — single source of truth
# ============================================================

@dataclass
class ReportMetrics:
    label: str
    path: str
    n_trades: int
    winners: int
    losers: int
    winrate: float
    total_pnl: float
    avg_pnl: float
    best_pnl: float
    worst_pnl: float
    profit_factor: float
    sharpe_est: float          # per-trade mean / std × sqrt(365)
    max_drawdown: float        # in absolute PnL cumulative
    max_drawdown_pct: float    # / initial capital (best-effort — needs known base)
    avg_bars_held: float
    trades_df: pd.DataFrame    # kept for equity-curve rendering


def _compute_metrics(label: str, report_dir: Path,
                     initial_capital: float = 10_000.0) -> ReportMetrics:
    trades_path = report_dir / "trades.parquet"
    if not trades_path.exists():
        raise FileNotFoundError(f"no trades.parquet at {trades_path}")
    df = pd.read_parquet(trades_path)
    pnl = df["pnl_after_fees"] if "pnl_after_fees" in df.columns else df["pnl"]
    n = len(df)
    if n == 0:
        return ReportMetrics(
            label=label, path=str(report_dir), n_trades=0,
            winners=0, losers=0, winrate=0.0, total_pnl=0.0,
            avg_pnl=0.0, best_pnl=0.0, worst_pnl=0.0,
            profit_factor=0.0, sharpe_est=0.0,
            max_drawdown=0.0, max_drawdown_pct=0.0,
            avg_bars_held=0.0, trades_df=df,
        )
    winners = int((pnl > 0).sum())
    losers = int((pnl < 0).sum())
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    if gross_loss > 0:
        pf = gross_win / gross_loss
    elif gross_win > 0:
        pf = float("inf")
    else:
        pf = 0.0
    std = float(pnl.std())
    sharpe = float(pnl.mean()) / std * np.sqrt(365) if std > 0 else 0.0
    cum = pnl.cumsum()
    dd = float((cum - cum.cummax()).min())
    bars_col = "bars_held" if "bars_held" in df.columns else None
    avg_bars = float(df[bars_col].mean()) if bars_col else 0.0

    return ReportMetrics(
        label=label, path=str(report_dir),
        n_trades=n, winners=winners, losers=losers,
        winrate=winners / n,
        total_pnl=float(pnl.sum()), avg_pnl=float(pnl.mean()),
        best_pnl=float(pnl.max()), worst_pnl=float(pnl.min()),
        profit_factor=pf, sharpe_est=sharpe,
        max_drawdown=dd,
        max_drawdown_pct=dd / initial_capital if initial_capital else 0.0,
        avg_bars_held=avg_bars,
        trades_df=df,
    )


# ============================================================
# Markdown renderer
# ============================================================

def _fmt_pnl(x: float) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x:+,.2f}"


def _fmt_num(x: float, places: int = 2) -> str:
    if x == float("inf"):
        return "∞"
    return f"{x:.{places}f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100:+.2f}%" if x else "0.00%"


def _row_val(m: ReportMetrics, key: str) -> str:
    v = getattr(m, key)
    fmt = {
        "n_trades": lambda x: f"{int(x)}",
        "winners": lambda x: f"{int(x)}",
        "losers": lambda x: f"{int(x)}",
        "winrate": lambda x: f"{x*100:.1f}%",
        "total_pnl": _fmt_pnl,
        "avg_pnl": _fmt_pnl,
        "best_pnl": _fmt_pnl,
        "worst_pnl": _fmt_pnl,
        "profit_factor": lambda x: _fmt_num(x, 2),
        "sharpe_est": lambda x: _fmt_num(x, 2),
        "max_drawdown": _fmt_pnl,
        "max_drawdown_pct": _fmt_pct,
        "avg_bars_held": lambda x: _fmt_num(x, 1),
    }[key]
    return fmt(v)


_ROW_LABELS = [
    ("Trades",        "n_trades"),
    ("Winners",       "winners"),
    ("Losers",        "losers"),
    ("Winrate",       "winrate"),
    ("Total PnL",     "total_pnl"),
    ("Avg PnL",       "avg_pnl"),
    ("Best trade",    "best_pnl"),
    ("Worst trade",   "worst_pnl"),
    ("Profit Factor", "profit_factor"),
    ("Sharpe (est)",  "sharpe_est"),
    ("Max DD",        "max_drawdown"),
    ("Max DD %",      "max_drawdown_pct"),
    ("Avg bars held", "avg_bars_held"),
]


def render_markdown(metrics: list[ReportMetrics]) -> str:
    if len(metrics) < 2:
        raise ValueError("compare needs ≥2 reports")

    header = ["Metric"] + [m.label for m in metrics]
    if len(metrics) == 2:
        header.append("Δ (B - A)")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for label, key in _ROW_LABELS:
        cells = [label] + [_row_val(m, key) for m in metrics]
        if len(metrics) == 2:
            a = getattr(metrics[0], key)
            b = getattr(metrics[1], key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                    and a != float("inf") and b != float("inf"):
                delta = b - a
                if key in ("winrate", "max_drawdown_pct"):
                    delta_str = f"{delta*100:+.2f}pp"
                elif key in ("n_trades", "winners", "losers"):
                    delta_str = f"{int(delta):+d}"
                else:
                    delta_str = f"{delta:+.2f}"
            else:
                delta_str = "n/a"
            cells.append(delta_str)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ============================================================
# HTML renderer — reuses the same SVG chart style as shadow dashboard
# ============================================================

def _svg_overlay_equity_curves(metrics: list[ReportMetrics],
                                width: int = 900, height: int = 260) -> str:
    """Overlay each report's cumulative PnL as a line — X is normalized
    trade index so paths of different lengths line up."""
    colors = ["#0a7d34", "#1976d2", "#c62828", "#7b1fa2"]
    all_pnl = []
    for m in metrics:
        pnl_col = "pnl_after_fees" if "pnl_after_fees" in m.trades_df.columns else "pnl"
        if pnl_col in m.trades_df.columns and not m.trades_df.empty:
            all_pnl.append(m.trades_df[pnl_col].cumsum().to_numpy())
        else:
            all_pnl.append(np.array([0.0]))
    y_min = min(float(a.min()) for a in all_pnl if len(a))
    y_max = max(float(a.max()) for a in all_pnl if len(a))
    y_range = y_max - y_min or 1.0
    pad = 20
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    polylines = []
    legend_items = []
    for i, (m, cum) in enumerate(zip(metrics, all_pnl)):
        color = colors[i % len(colors)]
        n = len(cum)
        if n < 2:
            continue
        pts = []
        for j, v in enumerate(cum):
            x = pad + (j / (n - 1)) * inner_w
            y = pad + (1 - (v - y_min) / y_range) * inner_h
            pts.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" '
            f'stroke-width="1.5"/>'
        )
        final = f"{cum[-1]:+.0f}"
        legend_items.append(
            f'<span style="color:{color};margin-right:16px">'
            f'■ {html.escape(m.label)} ({final})</span>'
        )
    return (
        f'<div style="margin-top:8px">{"".join(legend_items)}</div>'
        f'<svg width="{width}" height="{height}" '
        f'style="background:#f6f6f6;border-radius:6px">'
        f'<text x="{pad}" y="14" fill="#666" font-size="10">max PnL {y_max:+,.0f}</text>'
        f'<text x="{pad}" y="{height - 4}" fill="#666" font-size="10">'
        f'min PnL {y_min:+,.0f}</text>'
        f'{"".join(polylines)}</svg>'
    )


def render_html(metrics: list[ReportMetrics]) -> str:
    header_cells = "".join(f"<th>{html.escape(m.label)}</th>" for m in metrics)
    if len(metrics) == 2:
        header_cells += "<th>Δ (B − A)</th>"
    rows_html = []
    for label, key in _ROW_LABELS:
        cells = "".join(f"<td>{_row_val(m, key)}</td>" for m in metrics)
        if len(metrics) == 2:
            a = getattr(metrics[0], key)
            b = getattr(metrics[1], key)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                    and a != float("inf") and b != float("inf"):
                delta = b - a
                if key in ("winrate", "max_drawdown_pct"):
                    delta_txt = f"{delta*100:+.2f}pp"
                elif key in ("n_trades", "winners", "losers"):
                    delta_txt = f"{int(delta):+d}"
                else:
                    delta_txt = f"{delta:+.2f}"
                color = ("#0a7d34" if _is_positive_improvement(key, delta)
                         else "#c62828" if _is_negative_regression(key, delta) else "#666")
                cells += f'<td style="color:{color}"><b>{delta_txt}</b></td>'
            else:
                cells += "<td>n/a</td>"
        rows_html.append(f"<tr><th>{label}</th>{cells}</tr>")

    curve = _svg_overlay_equity_curves(metrics)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    labels_joined = html.escape(" vs ".join(m.label for m in metrics))
    path_list = "<br>".join(
        f'<code>{html.escape(m.label)}: {html.escape(m.path)}</code>'
        for m in metrics
    )

    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Rabbit Hunter · Compare · {labels_joined}</title>
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
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #f0f0f0; }}
  th:first-child {{ text-align: left; }}
  th {{ background: #fafafa; font-weight: 600; color: #444; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.05em; }}
  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<header>
  <h1>Rabbit Hunter · Report Comparison</h1>
  <div class="subtitle">{labels_joined} · rendered {now}</div>
</header>
<main>
<section>
  <h2>Sources</h2>
  {path_list}
</section>
<section>
  <h2>Side-by-side metrics</h2>
  <table>
    <thead><tr><th>Metric</th>{header_cells}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
</section>
<section>
  <h2>Cumulative PnL overlay</h2>
  {curve}
</section>
</main>
</body>
</html>'''


def _is_positive_improvement(key: str, delta: float) -> bool:
    """Should this delta be shown green?"""
    positive_when_up = {"winners", "winrate", "total_pnl", "avg_pnl",
                        "best_pnl", "profit_factor", "sharpe_est"}
    positive_when_down = {"losers", "worst_pnl"}  # less negative is better
    if key in positive_when_up:
        return delta > 0
    if key in positive_when_down:
        return delta > 0    # e.g. worst_pnl became less negative
    if key == "max_drawdown":
        return delta > 0    # DD is negative; less negative is better
    if key == "max_drawdown_pct":
        return delta > 0
    return False


def _is_negative_regression(key: str, delta: float) -> bool:
    positive_when_up = {"winners", "winrate", "total_pnl", "avg_pnl",
                        "best_pnl", "profit_factor", "sharpe_est"}
    if key in positive_when_up:
        return delta < 0
    if key in {"max_drawdown", "max_drawdown_pct"}:
        return delta < 0
    return False


# ============================================================
# Public entry
# ============================================================

def compare(
    report_dirs: list[Path],
    labels: list[str] | None = None,
    initial_capital: float = 10_000.0,
) -> tuple[str, str]:
    """Compute metrics for each report, return (markdown, html) side-by-side."""
    if labels is None:
        labels = [d.name for d in report_dirs]
    if len(labels) != len(report_dirs):
        raise ValueError("labels count must match report_dirs count")
    metrics = [
        _compute_metrics(label, d, initial_capital=initial_capital)
        for label, d in zip(labels, report_dirs)
    ]
    return render_markdown(metrics), render_html(metrics)
