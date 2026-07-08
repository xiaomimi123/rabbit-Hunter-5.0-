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


class RegimeRule(BaseModel):
    """Per-structure-regime trade permissions. Derived from the
    structure × side performance grid: e.g. range-longs were the only
    net-losing cell (14 trades, 36% WR, -$112) while range-shorts
    stayed positive — so 'range' can disallow longs but keep shorts."""
    model_config = ConfigDict(extra="forbid")
    allow_long: bool = True
    allow_short: bool = True
    # Multiplies the composed score before thresholding — 0.5 halves
    # conviction in this regime (fewer marginal trades), 1.0 = neutral.
    score_multiplier: float = Field(ge=0, le=1, default=1.0)


class StrategyRouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    composer: Literal["weighted_avg", "unanimous", "regime_switch", "max_score"] = "weighted_avg"
    open_action_threshold: float = Field(ge=0, le=1, default=0.5)
    enabled_strategies: dict[str, StrategyEntry]
    # Keyed by structure_regime value: "uptrend" | "downtrend" | "range".
    # Missing keys default to allow-everything.
    regime_rules: dict[str, RegimeRule] = Field(default_factory=dict)


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


class LiveExecutionConfig(BaseModel):
    """Phase 5 · live-execution safety envelope.

    Two invariants pinned in this schema (before any code runs):

      1. `enabled=False` is the default. Every code path that could touch
         real money checks this flag first and short-circuits when off.

      2. When enabled, credentials MUST come from env vars named here.
         No plaintext keys ever appear in config or version control. If
         the env var is unset, the LiveExecutor refuses to construct —
         we prefer a loud startup crash over silently reverting to paper.
    """
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    exchange: Literal["okx"] = "okx"
    testnet: bool = True
    api_key_env: str = "OKX_API_KEY"
    api_secret_env: str = "OKX_API_SECRET"
    passphrase_env: str = "OKX_API_PASSPHRASE"
    # Max notional per order (belt-and-suspenders on top of position sizing).
    # A misconfigured risk engine cannot blow past this without also editing
    # config — makes accidental fat-finger orders effectively impossible.
    max_notional_per_order: float = Field(gt=0, default=1_000.0)
    # If True, block orders when reconciliation shows a mismatch between
    # ledger positions and exchange positions.
    block_on_reconcile_mismatch: bool = True


class BtcCrashBoostConfig(BaseModel):
    """v0.1.3: on BTC systemic-crash bars, uplift same-side short orders.
    Cluster analysis showed 10 mass-crash days generated $21 per trade avg
    vs $9 on other days — a real edge worth pressing when it appears."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # BTC symbol used as the market beacon.
    btc_symbol: str = "BTC-USDT-SWAP"
    # Trigger when BTC zscore_20 <= -this AND btc close < prior close.
    zscore_threshold: float = Field(gt=0, default=2.0)
    # How much to scale short-side orders on triggered bars.
    boost_multiplier: float = Field(ge=1.0, default=1.2)


class ChopKillSwitchConfig(BaseModel):
    """v0.1.3: pause new entries after a streak of low-winrate closes.
    Cluster analysis showed 2025Q2-Q4 (chop regime) drove 3 consecutive losing
    quarters; a rolling-WR gate exits early instead of pushing through."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    # How many most-recent closed trades to look at.
    window: int = Field(gt=0, default=10)
    # If winrate over the window falls below this, pause.
    wr_threshold: float = Field(ge=0, le=1, default=0.35)
    # Number of main-interval bars to stay paused before evaluating again.
    pause_bars: int = Field(gt=0, default=48)


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
    chop_kill_switch: ChopKillSwitchConfig = Field(default_factory=ChopKillSwitchConfig)
    btc_crash_boost: BtcCrashBoostConfig = Field(default_factory=BtcCrashBoostConfig)
    live_execution: LiveExecutionConfig = Field(default_factory=LiveExecutionConfig)
