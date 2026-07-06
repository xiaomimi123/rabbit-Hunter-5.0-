from pathlib import Path
import numpy as np
import pandas as pd
from rabbit_hunter.backtest.report import (
    compute_stats,
    find_loss_clusters,
    ReportBuilder,
)
from rabbit_hunter.backtest.engine import BacktestResult
from rabbit_hunter.backtest.ledger import Ledger
from rabbit_hunter.config.schema import (
    AppConfig, DataConfig, FeatureEngineConfig, StrategyRouterConfig, StrategyEntry,
    RiskConfig, ExecutionConfig, FeeConfig, BacktestConfig, ReportConfig,
)


def _cfg():
    return AppConfig(
        data=DataConfig(exchange="okx", symbols=["BTC-USDT-SWAP"],
                        main_interval="1H", confirm_interval="15m", history_window_days=30),
        feature_engine=FeatureEngineConfig(version="0.1.0"),
        strategy_router=StrategyRouterConfig(composer="weighted_avg",
                                             enabled_strategies={"trend_following": StrategyEntry(weight=1.0, config_file="strategies/trend_following.yaml")}),
        risk=RiskConfig(risk_per_trade_pct=1.0, atr_stop_multiplier=1.5, reward_risk_ratio=2.0,
                        max_leverage=3, daily_max_loss_pct=3.0, hold_timeout_bars=48),
        execution=ExecutionConfig(fees=FeeConfig(maker=0.0002, taker=0.0005),
                                  slippage_atr_multiplier=0.1, funding_settlement=True),
        backtest=BacktestConfig(start="2025-01-01", end="2025-02-01", initial_capital=10_000.0),
        report=ReportConfig(),
    )


def _mk_trades(n=30):
    return [
        {
            "symbol": "BTC-USDT-SWAP", "side": "long" if i % 2 == 0 else "short",
            "entry_time": i * 3_600_000, "exit_time": (i + 1) * 3_600_000,
            "entry_price": 50_000.0, "exit_price": 50_100.0 if i % 3 else 49_900.0,
            "size": 0.01, "pnl_raw": 1.0 if i % 3 else -1.0,
            "pnl_after_fees": 0.9 if i % 3 else -1.1,
            "fees": 0.1, "funding": 0.0, "slippage": 5.0, "hold_bars": 5,
            "exit_reason": "take_profit" if i % 3 else "stop_loss",
            "entry_snapshot": {"regime": "trending", "adx": 30.0, "rsi_14": 55},
            "exit_snapshot": {"regime": "trending", "adx": 28.0, "rsi_14": 60},
            "strategy_scores": {"trend_following": 0.7},
        }
        for i in range(n)
    ]


def test_compute_stats_returns_expected_keys():
    trades = _mk_trades()
    eq = pd.DataFrame({"timestamp": range(30), "equity": np.linspace(10000, 10100, 30)})
    stats = compute_stats(trades, eq, initial_capital=10_000.0)
    for k in ("total_return_pct", "sharpe", "max_drawdown_pct", "trade_count", "win_rate_pct", "profit_factor"):
        assert k in stats


def test_find_loss_clusters_finds_something_with_many_losses():
    losing_trades = [
        {**t, "pnl_after_fees": -1.0,
         "entry_snapshot": {"regime": "ranging", "adx": 15, "rsi_14": 75},
         "exit_snapshot": {"regime": "ranging", "adx": 15, "rsi_14": 75}}
        for t in _mk_trades(30)
    ]
    df = pd.DataFrame([{
        "regime_t0": t["entry_snapshot"]["regime"],
        "session_t0": "asia",
        "day_of_week_t0": 0,
        "pnl_after_fees": t["pnl_after_fees"],
    } for t in losing_trades])
    clusters = find_loss_clusters(df, min_trades=10, max_winrate=0.5)
    assert len(clusters) > 0


def test_report_builder_writes_all_files(tmp_path):
    result = BacktestResult(
        ledger=Ledger(initial_capital=10_000.0, equity=10_050.0, closed_trades=_mk_trades()),
        snapshots=pd.DataFrame([{"timestamp": 0, "symbol": "BTC-USDT-SWAP",
                                 "action": "wait", "conviction": 0.1, "order_placed": False}]),
        equity_curve=pd.DataFrame({"timestamp": [0, 1, 2], "equity": [10_000.0, 10_020.0, 10_050.0]}),
    )
    fake_feats = pd.DataFrame({
        "timestamp": [0, 3_600_000, 7_200_000],
        "open": [50_000.0, 50_100.0, 50_050.0],
        "high": [50_050.0, 50_150.0, 50_100.0],
        "low": [49_950.0, 50_050.0, 50_000.0],
        "close": [50_000.0, 50_100.0, 50_050.0],
    })
    builder = ReportBuilder(_cfg(), {"BTC-USDT-SWAP": fake_feats})
    out_dir = builder.build(result, output_root=tmp_path, git_commit="test123")

    assert (out_dir / "report.md").exists()
    assert (out_dir / "ai_context.md").exists()
    assert (out_dir / "trades.parquet").exists()
    assert (out_dir / "snapshots.parquet").exists()
    assert (out_dir / "config_snapshot.yaml").exists()
    assert (out_dir / "charts" / "equity_curve.png").exists()
    assert (out_dir / "charts" / "monthly_pnl.png").exists()
