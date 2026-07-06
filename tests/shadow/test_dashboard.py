"""Unit tests for the shadow HTML dashboard.

The dashboard's job is to be robust to missing files (freshly started run,
crashed writer, corrupt parquet) and to be truly self-contained (no
external asset references). Both are pinned here.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd

from rabbit_hunter.backtest.ledger import Ledger, Position
from rabbit_hunter.shadow.dashboard import render_dashboard, write_dashboard


def _empty_state_dir(tmp_path: Path) -> Path:
    state = tmp_path / "shadows"
    (state / "state").mkdir(parents=True)
    return state


def _seed_ledger(state_dir: Path, equity: float = 11_500.0,
                  initial: float = 10_000.0,
                  n_open: int = 1, n_closed: int = 3) -> None:
    ledger = Ledger(initial_capital=initial)
    ledger.equity = equity
    for i in range(n_open):
        sym = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"][i % 2]
        ledger.open_positions[sym] = Position(
            symbol=sym, side="short", entry_time=1_700_000_000_000,
            entry_price=50_000.0, size=0.1, fees_paid=0.0,
            stop=51_000.0, take_profit=47_000.0,
            entry_snapshot={"close": 50_000.0},
            strategy_scores={"trend_following": {"long": 0.0, "short": 0.7}},
        )
    for i in range(n_closed):
        pnl = 100.0 if i % 2 == 0 else -60.0
        ledger.closed_trades.append({
            "symbol": "BTC-USDT-SWAP", "side": "short",
            "pnl_after_fees": pnl, "exit_reason": "take_profit",
            "exit_time": 1_700_000_000_000 + i * 3_600_000,
        })
    with (state_dir / "state" / "ledger.pkl").open("wb") as f:
        pickle.dump(ledger, f)


def _seed_metrics_history(state_dir: Path, n: int = 5,
                           include_alert: bool = False) -> None:
    rows = []
    for i in range(n):
        row = {
            "timestamp_ms": 1_700_000_000_000 + i * 60_000,
            "equity": 10_000.0 + i * 100.0,
            "initial_capital": 10_000.0,
            "total_pnl": i * 100.0,
            "pnl_pct": (i * 100.0) / 10_000.0,
            "peak_equity": 10_000.0 + i * 100.0,
            "drawdown_from_peak_pct": 0.0,
            "open_positions": 0,
            "open_notional": 0.0,
            "open_long_notional": 0.0,
            "open_short_notional": 0.0,
            "total_closed_trades": 0,
            "winners": 0, "losers": 0,
            "winrate": 0.0, "profit_factor": 0.0,
            "last_bar_ts_ms": 1_700_000_000_000 + i * 60_000,
            "minutes_since_last_bar": 0.0,
            "consecutive_errors": 0,
            "alerts": "drawdown_high=12%>=10%" if (include_alert and i == n - 1) else "",
            "alert_count": 1 if (include_alert and i == n - 1) else 0,
        }
        rows.append(row)
    pd.DataFrame(rows).to_parquet(state_dir / "state" / "metrics_history.parquet")


# ============================================================
# Empty state — must still render something usable, not crash
# ============================================================

def test_render_empty_state_dir_returns_html(tmp_path):
    state = _empty_state_dir(tmp_path)
    out = render_dashboard(state)
    assert out.startswith("<!doctype html>")
    assert "Rabbit Hunter" in out
    # No ledger → no KPI row rendered but must NOT throw
    assert "No ledger state" not in out  # dashboard doesn't hard-error


def test_render_missing_state_dir_does_not_crash(tmp_path):
    """Point at a directory that has no state subdir at all."""
    p = tmp_path / "doesnotexist"
    p.mkdir()
    out = render_dashboard(p)
    assert "<!doctype html>" in out


# ============================================================
# Self-containment invariants — protects against future edits
# regressing on the CSP-safe / no-external-assets promise
# ============================================================

def test_html_is_self_contained_no_external_urls(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state)
    _seed_metrics_history(state, n=5)
    out = render_dashboard(state)
    # No http(s):// URLs anywhere (scripts, images, fonts, links)
    assert "http://" not in out
    assert "https://" not in out
    # No <script src=...> or <link rel="stylesheet" href=...>
    assert re.search(r'<script[^>]*src=', out) is None
    assert re.search(r'<link[^>]+rel=["\']?stylesheet', out) is None


def test_html_includes_inline_style(tmp_path):
    """CSS must be inline in a <style> block — proves style tag is present
    so the page has base styling even when opened from disk."""
    state = _empty_state_dir(tmp_path)
    out = render_dashboard(state)
    assert "<style>" in out and "</style>" in out


# ============================================================
# Content — with a seeded ledger + history, the sections show up
# ============================================================

def test_render_with_seeded_ledger_shows_kpi_and_positions(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state, equity=11_500.0, n_open=1, n_closed=3)
    _seed_metrics_history(state, n=10)
    out = render_dashboard(state)
    # KPI values
    assert "$11,500" in out
    assert "PnL" in out
    # Positions table
    assert "BTC-USDT-SWAP" in out
    # Trades table (3 closed)
    assert "take_profit" in out


def test_alert_history_shown_when_present(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state)
    _seed_metrics_history(state, n=5, include_alert=True)
    out = render_dashboard(state)
    assert "drawdown_high" in out


def test_all_clear_when_no_alerts(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state)
    _seed_metrics_history(state, n=5, include_alert=False)
    out = render_dashboard(state)
    assert "All clear" in out


# ============================================================
# write_dashboard — end-to-end file emission
# ============================================================

def test_write_dashboard_creates_file(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state)
    _seed_metrics_history(state, n=5)
    out_path = tmp_path / "out" / "dashboard.html"
    result = write_dashboard(state, out_path)
    assert result == out_path
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")


def _seed_ledger_with_shadow_trades(state_dir: Path) -> None:
    """Seed a ledger with two closed shadow trades so the per-cluster
    section has real content to render."""
    from rabbit_hunter.backtest.ledger import Ledger
    ledger = Ledger(initial_capital=10_000.0)
    ledger.closed_trades = [
        {
            "symbol": "BTC-USDT-SWAP", "side": "short",
            "pnl_after_fees": +100.0, "exit_reason": "take_profit",
            "exit_time": 1_700_000_000_000, "bars_held": 5,
            "entry_snapshot": {
                "rsi_14": 25.0, "zscore_20": 0.0, "bb_pct": 0.5,
                "structure_regime": "range", "bos_flag": 0,
            },
        },
        {
            "symbol": "ETH-USDT-SWAP", "side": "long",
            "pnl_after_fees": +50.0, "exit_reason": "take_profit",
            "exit_time": 1_700_003_600_000, "bars_held": 8,
            "entry_snapshot": {
                "rsi_14": 75.0, "zscore_20": 0.0, "bb_pct": 0.5,
                "structure_regime": "range", "bos_flag": 0,
            },
        },
    ]
    with (state_dir / "state" / "ledger.pkl").open("wb") as f:
        pickle.dump(ledger, f)


def test_dashboard_renders_per_cluster_section(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger_with_shadow_trades(state)
    out = render_dashboard(state)
    # The section header exists
    assert "Per-cluster performance" in out
    # Cluster labels shown
    assert "1_momentum_breakdown" in out
    assert "2_momentum_breakout" in out


def test_dashboard_cluster_section_muted_when_no_trades(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state, n_closed=0)   # ledger with no closed trades
    out = render_dashboard(state)
    assert "No closed trades to classify" in out


def test_corrupt_history_does_not_crash(tmp_path):
    state = _empty_state_dir(tmp_path)
    _seed_ledger(state)
    # Write garbage into history parquet path
    (state / "state" / "metrics_history.parquet").write_bytes(b"not parquet")
    out = render_dashboard(state)
    assert "<!doctype html>" in out
