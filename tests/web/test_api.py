"""Tests for the FastAPI endpoints.

Each test spins up a fresh TestClient against `create_app(Paths(tmp_path))`
so nothing touches production data. Fixtures build the minimum on-disk
state each endpoint reads.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from rabbit_hunter.backtest.ledger import Ledger, Position
from rabbit_hunter.web.app import create_app
from rabbit_hunter.web.paths import Paths


@pytest.fixture
def app_root(tmp_path: Path) -> Path:
    """Skeleton project tree for a test app to read from."""
    for sub in ("reports", "shadows/state", "models", "baselines",
                "configs/strategies", "configs/.history",
                "data/raw/okx"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "default.yaml").write_text(
        yaml.safe_dump({
            "data": {"exchange": "okx", "symbols": ["BTC-USDT-SWAP"],
                      "main_interval": "1H", "confirm_interval": "15m",
                      "history_window_days": 30},
            "feature_engine": {"version": "0.1.0"},
            "strategy_router": {
                "composer": "weighted_avg",
                "enabled_strategies": {
                    "trend_following": {"weight": 1.0,
                                         "config_file": "strategies/trend_following.yaml"},
                },
            },
            "risk": {"risk_per_trade_pct": 1.0, "atr_stop_multiplier": 1.5,
                      "reward_risk_ratio": 2.0, "max_leverage": 3,
                      "daily_max_loss_pct": 3.0, "hold_timeout_bars": 48},
            "execution": {"fees": {"maker": 0.0002, "taker": 0.0005},
                           "slippage_atr_multiplier": 0.1,
                           "funding_settlement": True},
            "backtest": {"start": "2025-01-01", "end": "2025-02-01",
                          "initial_capital": 10_000.0},
            "report": {},
        }),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def client(app_root: Path) -> TestClient:
    return TestClient(create_app(Paths(root=app_root)))


def _seed_ledger(root: Path, equity: float = 11_000.0,
                  n_open: int = 0, n_closed: int = 0) -> None:
    ledger = Ledger(initial_capital=10_000.0)
    ledger.equity = equity
    for i in range(n_open):
        sym = "BTC-USDT-SWAP"
        ledger.open_positions[sym] = Position(
            symbol=sym, side="short", entry_time=1_700_000_000_000,
            entry_price=50_000.0, size=0.01, fees_paid=0.0,
            stop=51_000.0, take_profit=47_000.0,
            entry_snapshot={"close": 50_000.0},
            strategy_scores={"trend_following": {"long": 0.0, "short": 0.7}},
        )
    for i in range(n_closed):
        ledger.closed_trades.append({
            "symbol": "BTC-USDT-SWAP", "side": "short",
            "pnl_after_fees": (100.0 if i % 2 == 0 else -30.0),
            "exit_reason": "take_profit",
            "entry_time": 1_700_000_000_000 + i * 3_600_000,
            "exit_time": 1_700_000_000_000 + (i + 5) * 3_600_000,
            "bars_held": 5,
            "entry_snapshot": {"rsi_14": 25.0, "zscore_20": 0.0,
                                "bb_pct": 0.5, "structure_regime": "range",
                                "bos_flag": 0},
        })
    with (root / "shadows" / "state" / "ledger.pkl").open("wb") as f:
        pickle.dump(ledger, f)


def _seed_metrics_history(root: Path, n: int = 5) -> None:
    rows = []
    for i in range(n):
        rows.append({
            "timestamp_ms": 1_700_000_000_000 + i * 60_000,
            "equity": 10_000.0 + i * 100.0,
            "initial_capital": 10_000.0,
            "total_pnl": i * 100.0,
            "pnl_pct": i * 0.01,
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
            "alerts": "" if i < n - 1 else "drawdown_high=12%>=10%",
            "alert_count": 0 if i < n - 1 else 1,
        })
    pd.DataFrame(rows).to_parquet(
        root / "shadows" / "state" / "metrics_history.parquet"
    )


# ============================================================
# Bootstrap
# ============================================================

def test_root_serves_html(client: TestClient):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_health_endpoint(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


# ============================================================
# Shadow
# ============================================================

def test_shadow_state_empty(client: TestClient):
    r = client.get("/api/shadow/state")
    assert r.status_code == 200
    assert r.json()["has_ledger"] is False


def test_shadow_state_with_ledger(app_root, client: TestClient):
    _seed_ledger(app_root, equity=11_500.0, n_open=1, n_closed=2)
    _seed_metrics_history(app_root)
    r = client.get("/api/shadow/state")
    body = r.json()
    assert body["has_ledger"] is True
    assert body["equity"] == 11_500.0
    assert body["open_positions"] == 1
    assert body["closed_trades"] == 2
    assert body["total_pnl"] == 1_500.0


def test_shadow_metrics_history_returns_points(app_root, client: TestClient):
    _seed_metrics_history(app_root, n=5)
    r = client.get("/api/shadow/metrics-history?hours=0")   # all points
    points = r.json()["points"]
    assert len(points) == 5
    assert "equity" in points[0]


def test_shadow_positions_returns_open(app_root, client: TestClient):
    _seed_ledger(app_root, n_open=1)
    r = client.get("/api/shadow/positions")
    pos = r.json()["positions"]
    assert len(pos) == 1
    assert pos[0]["symbol"] == "BTC-USDT-SWAP"
    assert pos[0]["side"] == "short"


def test_shadow_trades_returns_recent(app_root, client: TestClient):
    _seed_ledger(app_root, n_closed=3)
    r = client.get("/api/shadow/trades?limit=2")
    trades = r.json()["trades"]
    assert len(trades) == 2
    # Reversed order — most recent first
    assert trades[0]["exit_time_ms"] > trades[-1]["exit_time_ms"]


def test_shadow_alerts_only_alerted_rows(app_root, client: TestClient):
    _seed_metrics_history(app_root, n=5)   # last row has an alert
    r = client.get("/api/shadow/alerts?hours=1000")
    alerts = r.json()["alerts"]
    assert len(alerts) == 1
    assert "drawdown_high" in alerts[0]["alerts"]


# ============================================================
# Backtests
# ============================================================

def _seed_report(root: Path, name: str, n: int = 20) -> None:
    d = root / "reports" / name
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "symbol": ["BTC-USDT-SWAP"] * n,
        "side": ["short"] * n,
        "pnl_after_fees": [(i - n // 2) * 10.0 for i in range(n)],
        "entry_time": list(range(n)),
        "exit_time": list(range(n)),
        "bars_held": [5] * n,
        "exit_reason": ["take_profit"] * n,
    })
    df.to_parquet(d / "trades.parquet")


def test_backtest_list_empty(client: TestClient):
    r = client.get("/api/backtests")
    assert r.json() == {"reports": []}


def test_backtest_list_recent_first(app_root, client: TestClient):
    _seed_report(app_root, "run-A")
    _seed_report(app_root, "run-B")
    r = client.get("/api/backtests")
    names = [x["name"] for x in r.json()["reports"]]
    # Sorted reverse by name → B first
    assert names == ["run-B", "run-A"]


def test_backtest_detail(app_root, client: TestClient):
    _seed_report(app_root, "run-A", n=10)
    r = client.get("/api/backtests/run-A")
    body = r.json()
    assert body["n_trades"] == 10
    assert "top_trades" in body
    assert len(body["top_trades"]) <= 20


def test_backtest_detail_bars_held_null_when_column_missing(app_root, client: TestClient):
    """Older trades.parquet files predate the bars_held column. The API
    must surface `bars_held: null` (not omit the field, not error) so
    the frontend can render "—" instead of "undefined"."""
    d = app_root / "reports" / "old-run"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "symbol": ["BTC-USDT-SWAP"] * 5,
        "side": ["short"] * 5,
        "pnl_after_fees": [10.0, -5.0, 20.0, -8.0, 3.0],
        "entry_time": list(range(5)),
        "exit_time": list(range(5)),
        "exit_reason": ["tp"] * 5,
        # no bars_held column — this is the point of the test
    })
    df.to_parquet(d / "trades.parquet")
    body = client.get("/api/backtests/old-run").json()
    assert body["n_trades"] == 5
    assert all("bars_held" in t for t in body["top_trades"])
    assert all(t["bars_held"] is None for t in body["top_trades"])


def test_backtest_detail_404(client: TestClient):
    assert client.get("/api/backtests/no-such").status_code == 404


# ============================================================
# Models
# ============================================================

def _seed_model(root: Path, version: str, test_auc: float = 0.55) -> None:
    d = root / "models" / f"ml_model_v{version}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pkl").write_bytes(b"fake")
    (d / "training_result.json").write_text(json.dumps({
        "model_version": version, "trained_at": "2026-01-01T00:00:00Z",
        "train_auc": 0.7, "test_auc": test_auc,
        "train_accuracy": 0.6, "test_accuracy": 0.55,
        "n_train": 100, "n_test": 30,
        "features_used": [], "hyperparameters": {},
    }), encoding="utf-8")


def test_models_list_empty(client: TestClient):
    assert client.get("/api/models").json() == {"models": []}


def test_models_list_populated(app_root, client: TestClient):
    _seed_model(app_root, "v1", test_auc=0.55)
    _seed_model(app_root, "v2", test_auc=0.60)
    r = client.get("/api/models")
    models = r.json()["models"]
    assert len(models) == 2
    assert models[0]["version"] in {"v1", "v2"}


def test_retrain_log_empty(client: TestClient):
    assert client.get("/api/models/retrain-log").json() == {"entries": []}


def test_retrain_log_populated(app_root, client: TestClient):
    log = app_root / "models" / "retrain_log.jsonl"
    log.write_text(
        json.dumps({"action": "promote", "timestamp": "2026-01-01T00:00:01",
                     "candidate_test_auc": 0.55, "reason": "no prior"}) + "\n"
        + json.dumps({"action": "reject_no_improvement",
                       "timestamp": "2026-01-02T00:00:01",
                       "candidate_test_auc": 0.54,
                       "reason": "no lift"}) + "\n",
        encoding="utf-8",
    )
    entries = client.get("/api/models/retrain-log").json()["entries"]
    assert len(entries) == 2
    # Reversed — most recent first
    assert entries[0]["action"] == "reject_no_improvement"


# ============================================================
# Config
# ============================================================

def test_config_current(app_root, client: TestClient):
    r = client.get("/api/config/current")
    body = r.json()
    assert body["content"] is not None
    assert "symbols" in body["content"]


def test_config_history_empty(client: TestClient):
    assert client.get("/api/config/history").json() == {"entries": []}


def test_config_history_populated(app_root, client: TestClient):
    from rabbit_hunter.config.history import snapshot_if_changed
    snapshot_if_changed(app_root / "configs" / "default.yaml",
                          history_dir=app_root / "configs" / ".history",
                          now_utc="20260101-000001")
    r = client.get("/api/config/history")
    entries = r.json()["entries"]
    assert len(entries) == 1
    assert "config_hash" in entries[0]


# ============================================================
# Analytics
# ============================================================

def test_cluster_shadow_no_trades(client: TestClient):
    body = client.get("/api/analytics/clusters/shadow").json()
    assert body["total_trades"] == 0


def test_cluster_shadow_with_trades(app_root, client: TestClient):
    _seed_ledger(app_root, n_closed=4)
    body = client.get("/api/analytics/clusters/shadow").json()
    assert body["total_trades"] == 4
    assert len(body["clusters"]) == 6   # canonical 6-cluster shape


def test_cluster_report_by_name(app_root, client: TestClient):
    _seed_report(app_root, "run-A")
    r = client.get("/api/analytics/clusters/report/run-A")
    assert r.status_code == 200


def test_cluster_report_404(client: TestClient):
    assert client.get(
        "/api/analytics/clusters/report/no-such"
    ).status_code == 404


def test_baselines_list_empty(client: TestClient):
    assert client.get("/api/analytics/baselines").json() == {"baselines": []}


def test_baselines_list_populated(app_root, client: TestClient):
    (app_root / "baselines" / "v1.json").write_text("{}", encoding="utf-8")
    (app_root / "baselines" / "v2.json").write_text("{}", encoding="utf-8")
    body = client.get("/api/analytics/baselines").json()
    assert len(body["baselines"]) == 2


# ============================================================
# Data health
# ============================================================

def test_data_health_returns_summary(app_root, client: TestClient):
    r = client.get("/api/data/health")
    body = r.json()
    assert "summary" in body
    assert "reports" in body
