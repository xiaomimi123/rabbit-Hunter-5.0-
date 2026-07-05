from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exchange: Literal["okx"] = "okx"
    symbols: list[str]
    main_interval: str
    confirm_interval: str
    history_window_days: int = Field(gt=0)


class FeatureEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    cache_enabled: bool = True


class StrategyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    weight: float = Field(ge=0)
    config_file: str


class StrategyRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    composer: Literal["weighted_avg", "unanimous", "regime_switch", "max_score"] = "weighted_avg"
    enabled_strategies: dict[str, StrategyEntry]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_per_trade_pct: float = Field(gt=0)
    atr_stop_multiplier: float = Field(gt=0)
    reward_risk_ratio: float = Field(gt=0)
    max_leverage: float = Field(gt=0)
    daily_max_loss_pct: float = Field(gt=0)
    hold_timeout_bars: int = Field(gt=0)


class FeeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    maker: float  # 小数形式：0.0002 表示 0.02%
    taker: float


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fees: FeeConfig
    slippage_atr_multiplier: float = Field(ge=0)
    funding_settlement: bool = True


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str  # ISO date
    end: str
    initial_capital: float = Field(gt=0)


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "reports"
    ai_context: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    feature_engine: FeatureEngineConfig
    strategy_router: StrategyRouterConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    report: ReportConfig
