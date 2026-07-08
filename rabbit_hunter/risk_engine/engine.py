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

    def _structural_stop_distance(
        self, intent: Intent, side: str, price: float, atr: float,
    ) -> float | None:
        """Distance from price to the structure-invalidation level plus a
        small buffer, clamped to [min, max] × ATR. None → caller falls
        back to the ATR stop (missing swing feature, wrong-side level)."""
        feats = intent.features_snapshot or {}
        level = feats.get("swing_low_last" if side == "long" else "swing_high_last")
        if level is None:
            return None
        try:
            level = float(level)
        except (TypeError, ValueError):
            return None
        if level != level:  # NaN
            return None
        raw = (price - level) if side == "long" else (level - price)
        if raw <= 0:
            # Swing level on the wrong side of price (structure already
            # broken) — structural stop meaningless here.
            return None
        dist = raw + self.cfg.structural_buffer_atr_mult * atr
        lo = self.cfg.structural_min_atr_mult * atr
        hi = self.cfg.structural_max_atr_mult * atr
        return min(max(dist, lo), hi)

    def size(self, intent: Intent, ctx: RiskContext) -> Order | None:
        if intent.action not in ("open_long", "open_short"):
            return None
        if daily_loss_tripped(ctx.daily_realized_pnl, ctx.initial_capital, self.cfg.daily_max_loss_pct):
            return None
        if ctx.atr <= 0 or ctx.price <= 0:
            return None
        side = "long" if intent.action == "open_long" else "short"
        stop_override = None
        if self.cfg.stop_mode == "structural":
            stop_override = self._structural_stop_distance(
                intent, side, ctx.price, ctx.atr,
            )
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
            stop_distance_override=stop_override,
        )
