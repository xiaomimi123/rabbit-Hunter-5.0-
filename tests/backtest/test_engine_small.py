import numpy as np
import pandas as pd
from rabbit_hunter.config.loader import load_config
from rabbit_hunter.config.schema import (
    AppConfig, DataConfig, FeatureEngineConfig, StrategyRouterConfig, StrategyEntry,
    RiskConfig, ExecutionConfig, FeeConfig, BacktestConfig, ReportConfig,
)
from rabbit_hunter.feature_engine.pipeline import build_features
from rabbit_hunter.scoring_engine.strategies.trend_following import TrendFollowing, TFParams
from rabbit_hunter.backtest.engine import BacktestEngine


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


def _mk_uptrend_df(n=400, base=100.0):
    ts = [i * 3_600_000 for i in range(n)]
    close = np.linspace(base, base + 100, n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 100.0),
        "funding_rate": np.full(n, 0.0001),
        "oi": np.linspace(1000, 1500, n),
    })


def test_backtest_produces_trades_and_equity_curve():
    cfg = _cfg()
    tf = TrendFollowing(TFParams(
        ema_fast=20, ema_slow=60, ema_trend=200,
        adx_threshold=25, volume_ratio_threshold=0.5,
        confirm_ema_fast=20, confirm_adx_threshold=15,
    ))
    engine = BacktestEngine(cfg, [tf])
    raw = _mk_uptrend_df()
    feats = build_features(raw)
    result = engine.run({"BTC-USDT-SWAP": feats}, open_action_threshold=0.3)

    assert len(result.equity_curve) > 100
    assert result.equity_curve["equity"].iloc[-1] > 0
    # 明显趋势中至少能触发一次开仓
    assert len(result.snapshots) > 0
