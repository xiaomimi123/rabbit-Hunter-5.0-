from pathlib import Path
from rabbit_hunter.config.loader import load_config


def test_load_default_config(tmp_path):
    cfg_path = Path("configs/default.yaml")
    cfg = load_config(cfg_path)
    assert cfg.data.symbols == ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
    assert cfg.data.main_interval == "1H"
    assert cfg.data.confirm_interval == "15m"
    assert cfg.risk.risk_per_trade_pct == 1.0
    assert cfg.backtest.initial_capital == 10000
    assert "trend_following" in cfg.strategy_router.enabled_strategies


import pytest
from pydantic import ValidationError


def test_load_config_rejects_unknown_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "data:\n"
        "  exchange: okx\n"
        "  symbols: [BTC-USDT-SWAP]\n"
        "  main_interval: '1H'\n"
        "  confirm_interval: '15m'\n"
        "  history_window_days: 30\n"
        "  unknown_field: oops\n"
        "feature_engine: {version: '0.1.0'}\n"
        "strategy_router:\n"
        "  composer: weighted_avg\n"
        "  enabled_strategies: {}\n"
        "risk: {risk_per_trade_pct: 1, atr_stop_multiplier: 1.5,"
        " reward_risk_ratio: 2, max_leverage: 3, daily_max_loss_pct: 3,"
        " hold_timeout_bars: 48}\n"
        "execution:\n"
        "  fees: {maker: 0.0002, taker: 0.0005}\n"
        "  slippage_atr_multiplier: 0.1\n"
        "backtest: {start: '2024-01-01', end: '2024-06-01', initial_capital: 1000}\n"
        "report: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(bad)
