from __future__ import annotations
from dataclasses import dataclass
from rabbit_hunter.config.schema import RiskConfig
from rabbit_hunter.strategy_router.router import Intent
from .position_sizing import compute_order, Order
from .daily_circuit import daily_loss_tripped


@dataclass(frozen=True)
class RiskContext:
    equity: float
    atr: float
    price: float
    daily_realized_pnl: float
    initial_capital: float
    open_positions_count: int


class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def size(self, intent: Intent, ctx: RiskContext) -> Order | None:
        if intent.action not in ("open_long", "open_short"):
            return None
        if daily_loss_tripped(ctx.daily_realized_pnl, ctx.initial_capital, self.cfg.daily_max_loss_pct):
            return None
        if ctx.atr <= 0 or ctx.price <= 0:
            return None
        side = "long" if intent.action == "open_long" else "short"
        return compute_order(
            symbol=intent.symbol,
            side=side,
            price=ctx.price,
            atr=ctx.atr,
            equity=ctx.equity,
            risk_per_trade_pct=self.cfg.risk_per_trade_pct,
            atr_stop_multiplier=self.cfg.atr_stop_multiplier,
            reward_risk_ratio=self.cfg.reward_risk_ratio,
            max_leverage=self.cfg.max_leverage,
        )
