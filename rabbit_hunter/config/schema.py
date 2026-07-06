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
    open_action_threshold: float = Field(ge=0, le=1, default=0.5)
    enabled_strategies: dict[str, StrategyEntry]


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    risk_per_trade_pct: float = Field(gt=0)
    atr_stop_multiplier: float = Field(gt=0)
    reward_risk_ratio: float = Field(gt=0)
    max_leverage: float = Field(gt=0)
    daily_max_loss_pct: float = Field(gt=0)
    hold_timeout_bars: int = Field(gt=0)
    # Trailing-stop (v0.2.0-scalp). All three ignored when trailing_enabled=False.
    trailing_enabled: bool = False
    trailing_activation_r: float = Field(gt=0, default=1.0)
    trailing_atr_multiplier: float = Field(gt=0, default=1.0)


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


class HardRulesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    min_quote_volume_24h: float = 1_000_000.0
    atr_pct_max_multiplier: float = 5.0
    atr_pct_baseline_window: int = 500


class PortfolioRiskConfig(BaseModel):
    """Phase 3 § 3.3: portfolio-level risk gates that see across positions."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # Rolling window used to compute inter-symbol return correlations.
    # 720 bars on 1H ≈ 30 days.
    correlation_window_bars: int = Field(gt=0, default=720)
    # Absolute Pearson correlation above this → treat as "same bet".
    max_correlation_threshold: float = Field(gt=0, le=1, default=0.7)
    # When correlated, multiply candidate order size by this factor
    # (0.0 = full rejection, 1.0 = no adjustment).
    correlated_size_reduction: float = Field(ge=0, le=1, default=0.5)
    # Total (|notional_long| + |notional_short|) / equity cap. Hard reject
    # any order that would push gross leverage past this.
    max_gross_leverage: float = Field(gt=0, default=3.0)


class CircuitBreakerConfig(BaseModel):
    """Phase 3 § 3.3: extreme-volatility watchdog independent of strategy path."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # Trip when current atr_14 >= this × the row's atr_pct_baseline
    # (already computed by Feature Engine's regime module).
    atr_shock_multiplier: float = Field(gt=0, default=3.0)
    # On trip: immediately close all open positions on the offending symbol
    # at the bar close, and refuse new entries this bar. Recovery: next bar
    # the ratio is normal again → normal flow resumes.
    emergency_close_on_shock: bool = True


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: DataConfig
    feature_engine: FeatureEngineConfig
    strategy_router: StrategyRouterConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    report: ReportConfig
    hard_rules: HardRulesConfig = Field(default_factory=HardRulesConfig)
    portfolio_risk: PortfolioRiskConfig = Field(default_factory=PortfolioRiskConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
