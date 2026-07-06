"""Unit tests for the multi-report compare tool.

The tool's job is: given ≥2 trades.parquet files, emit a table where each
row is a metric and each column is a run. Correctness = the metrics match
what you'd compute by hand from the input; robustness = missing files
error clearly, empty inputs don't crash.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from rabbit_hunter.backtest.compare import (
    ReportMetrics, _compute_metrics, compare, render_html, render_markdown,
)


def _mk_trades(pnls: list[float], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "symbol": ["BTC-USDT-SWAP"] * len(pnls),
        "side": ["long"] * len(pnls),
        "pnl_after_fees": pnls,
        "entry_time": [1_700_000_000_000 + i * 3_600_000 for i in range(len(pnls))],
        "exit_time":  [1_700_000_000_000 + (i + 5) * 3_600_000 for i in range(len(pnls))],
        "bars_held":  [5] * len(pnls),
        "exit_reason": ["take_profit" if p > 0 else "stop_loss" for p in pnls],
    })
    df.to_parquet(report_dir / "trades.parquet")


# ============================================================
# Metric computation
# ============================================================

def test_metrics_from_simple_trades(tmp_path):
    _mk_trades([+100.0, +50.0, -30.0, -20.0], tmp_path / "run_a")
    m = _compute_metrics("A", tmp_path / "run_a", initial_capital=10_000.0)
    assert m.n_trades == 4
    assert m.winners == 2
    assert m.losers == 2
    assert m.winrate == 0.5
    assert m.total_pnl == pytest.approx(100.0)
    assert m.profit_factor == pytest.approx(150.0 / 50.0)   # 3.0
    assert m.best_pnl == 100.0
    assert m.worst_pnl == -30.0


def test_metrics_no_losses_yields_infinite_pf(tmp_path):
    _mk_trades([+10.0, +20.0], tmp_path / "run_b")
    m = _compute_metrics("B", tmp_path / "run_b")
    assert m.profit_factor == float("inf")


def test_metrics_empty_trades_returns_zeros(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    pd.DataFrame({"pnl_after_fees": []}).to_parquet(d / "trades.parquet")
    m = _compute_metrics("E", d)
    assert m.n_trades == 0
    assert m.total_pnl == 0.0
    assert m.profit_factor == 0.0


def test_metrics_max_drawdown_matches_manual_calc(tmp_path):
    """cumsum: [100, 50, -10, -80]. Peak = 100. Trough = -80. DD = -180."""
    _mk_trades([+100.0, -50.0, -60.0, -70.0], tmp_path / "run_dd")
    m = _compute_metrics("DD", tmp_path / "run_dd", initial_capital=1000.0)
    assert m.max_drawdown == pytest.approx(-180.0)
    assert m.max_drawdown_pct == pytest.approx(-0.18)


def test_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _compute_metrics("X", tmp_path / "no-such-dir")


# ============================================================
# Markdown rendering
# ============================================================

def test_markdown_includes_per_symbol_section(tmp_path):
    """Rich markdown output (default) includes a per-symbol breakdown."""
    _mk_trades([+10.0, -5.0], tmp_path / "a")
    df_b = pd.read_parquet(tmp_path / "a" / "trades.parquet").copy()
    df_b["symbol"] = "ETH-USDT-SWAP"
    (tmp_path / "b").mkdir()
    df_b.to_parquet(tmp_path / "b" / "trades.parquet")
    md, _ = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "## Per-symbol breakdown" in md
    assert "BTC-USDT-SWAP" in md
    assert "ETH-USDT-SWAP" in md


def test_markdown_includes_top_trades_section(tmp_path):
    _mk_trades([+100.0, +50.0, -30.0], tmp_path / "a")
    _mk_trades([+80.0, -20.0], tmp_path / "b")
    md, _ = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "## Top 20 trades by |PnL|" in md
    # The largest-|PnL| trade must appear as the first row after headers
    assert "+100.00" in md


def test_markdown_summary_only_skips_extra_sections(tmp_path):
    """`include_sections=False` reproduces the legacy one-table output."""
    from rabbit_hunter.backtest.compare import _compute_metrics, render_markdown
    _mk_trades([+10.0, -5.0], tmp_path / "a")
    _mk_trades([+15.0, -3.0], tmp_path / "b")
    mA = _compute_metrics("A", tmp_path / "a")
    mB = _compute_metrics("B", tmp_path / "b")
    md = render_markdown([mA, mB], include_sections=False)
    assert "Per-symbol breakdown" not in md
    assert "Top" not in md
    # But summary table is still there
    assert "| A |" in md.splitlines()[0]


def test_markdown_top_trades_n_configurable(tmp_path):
    """--top-trades caps how many trade rows appear in the section."""
    from rabbit_hunter.backtest.compare import _compute_metrics, render_markdown
    _mk_trades([+100.0, +90.0, -80.0, -70.0, +60.0], tmp_path / "a")
    _mk_trades([+50.0], tmp_path / "b")
    md = render_markdown(
        [_compute_metrics("A", tmp_path / "a"),
         _compute_metrics("B", tmp_path / "b")],
        top_trades_n=2,
    )
    # Section header reflects the cap
    assert "## Top 2 trades by |PnL|" in md
    # And the section body should not include the smaller-|PnL| trades
    # (+50 and +60 have smaller |PnL| than +100 and +90)
    section = md.split("## Top 2 trades")[1]
    assert "+100.00" in section
    assert "+90.00" in section or "-80.00" in section
    # -70 and +60 should NOT be there
    assert "+60.00" not in section
    assert "-70.00" not in section


def test_markdown_two_reports_has_delta_column(tmp_path):
    _mk_trades([+100.0, +50.0], tmp_path / "a")
    _mk_trades([+120.0, +40.0], tmp_path / "b")
    md, _ = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    header = md.splitlines()[0]
    assert "| A |" in header
    assert "| B |" in header
    assert "Δ (B - A)" in header


def test_markdown_three_reports_omits_delta(tmp_path):
    for name in ("a", "b", "c"):
        _mk_trades([+10.0, +20.0], tmp_path / name)
    md, _ = compare(
        [tmp_path / "a", tmp_path / "b", tmp_path / "c"],
        labels=["A", "B", "C"],
    )
    header = md.splitlines()[0]
    assert "Δ" not in header


def test_markdown_default_labels_use_dir_name(tmp_path):
    _mk_trades([+1.0], tmp_path / "run_2026-01-01")
    _mk_trades([+2.0], tmp_path / "run_2026-01-02")
    md, _ = compare(
        [tmp_path / "run_2026-01-01", tmp_path / "run_2026-01-02"],
    )
    header = md.splitlines()[0]
    assert "run_2026-01-01" in header
    assert "run_2026-01-02" in header


def test_markdown_delta_shows_pp_for_winrate(tmp_path):
    _mk_trades([+1.0, -1.0], tmp_path / "a")     # WR 50%
    _mk_trades([+1.0, +1.0, +1.0, -1.0], tmp_path / "b")  # WR 75%
    md, _ = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    winrate_line = [l for l in md.splitlines() if l.startswith("| Winrate ")][0]
    assert "+25.00pp" in winrate_line


def test_render_markdown_requires_two_reports(tmp_path):
    _mk_trades([+1.0], tmp_path / "a")
    m = _compute_metrics("A", tmp_path / "a")
    with pytest.raises(ValueError):
        render_markdown([m])


# ============================================================
# HTML rendering
# ============================================================

def test_html_self_contained_no_external_urls(tmp_path):
    _mk_trades([+10.0, -5.0], tmp_path / "a")
    _mk_trades([+20.0, -5.0], tmp_path / "b")
    _, out = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "http://" not in out
    assert "https://" not in out
    assert re.search(r'<script[^>]*src=', out) is None
    assert re.search(r'<link[^>]+rel=["\']?stylesheet', out) is None


def test_html_contains_both_labels(tmp_path):
    _mk_trades([+10.0], tmp_path / "a")
    _mk_trades([+20.0], tmp_path / "b")
    _, out = compare([tmp_path / "a", tmp_path / "b"], labels=["v0.1g", "v0.1.4"])
    assert "v0.1g" in out
    assert "v0.1.4" in out


def test_html_has_svg_overlay_chart(tmp_path):
    _mk_trades([+10.0, +5.0, -3.0], tmp_path / "a")
    _mk_trades([+15.0, -2.0, +8.0], tmp_path / "b")
    _, out = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "<svg" in out
    # Two lines drawn (one per report)
    assert out.count("<polyline") == 2


# ============================================================
# Label mismatch validation
# ============================================================

def test_html_includes_per_symbol_breakdown(tmp_path):
    """Two reports with different symbols → per-symbol section shows both."""
    _mk_trades([+10.0, -5.0], tmp_path / "a")
    df_b = pd.read_parquet(tmp_path / "a" / "trades.parquet").copy()
    df_b["symbol"] = "ETH-USDT-SWAP"
    (tmp_path / "b").mkdir()
    df_b.to_parquet(tmp_path / "b" / "trades.parquet")
    _, out = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "Per-symbol breakdown" in out
    assert "BTC-USDT-SWAP" in out
    assert "ETH-USDT-SWAP" in out


def test_html_includes_trades_explorer_with_filters(tmp_path):
    """The trades explorer emits filter <select> elements per column."""
    _mk_trades([+10.0, -5.0], tmp_path / "a")
    _mk_trades([+15.0, -3.0], tmp_path / "b")
    _, out = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "Trades explorer" in out
    assert 'data-filter="report"' in out
    assert 'data-filter="symbol"' in out
    assert 'data-filter="side"' in out


def test_html_has_inline_sortable_js(tmp_path):
    """The interactive layer must be inline (no CDN) — proves the sort
    handler is bundled into the emitted HTML."""
    _mk_trades([+10.0, -5.0], tmp_path / "a")
    _mk_trades([+15.0, -3.0], tmp_path / "b")
    _, out = compare([tmp_path / "a", tmp_path / "b"], labels=["A", "B"])
    assert "<script>" in out
    assert "makeSortable" in out or "sortable" in out
    # Still self-contained (no CDN JS/CSS)
    assert 'src="http' not in out


def test_html_trades_explorer_caps_row_count(tmp_path):
    """Explorer table caps to 200 top-|pnl| rows so a 100k-trade backtest
    doesn't emit 100k <tr>s."""
    from rabbit_hunter.backtest.compare import _trades_explorer, _compute_metrics
    # Make 500 trades in one report
    n = 500
    dfa = pd.DataFrame({
        "symbol": ["BTC-USDT-SWAP"] * n,
        "side":   ["long"] * n,
        "pnl_after_fees": [i - n // 2 for i in range(n)],
        "entry_time": list(range(n)),
        "exit_time":  list(range(n)),
        "bars_held":  [5] * n,
        "exit_reason": ["tp"] * n,
    })
    (tmp_path / "run").mkdir()
    dfa.to_parquet(tmp_path / "run" / "trades.parquet")
    m = _compute_metrics("R", tmp_path / "run")
    html_frag = _trades_explorer([m], max_rows=50)
    # Count <tr> rows — should be at most 50 body rows + 1 header
    assert html_frag.count("<tr>") <= 51


def test_label_count_mismatch_raises(tmp_path):
    _mk_trades([+1.0], tmp_path / "a")
    _mk_trades([+1.0], tmp_path / "b")
    with pytest.raises(ValueError):
        compare([tmp_path / "a", tmp_path / "b"], labels=["only_one"])
