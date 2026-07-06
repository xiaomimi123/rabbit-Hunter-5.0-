from __future__ import annotations
from typing import Literal
from rabbit_hunter.config.schema import ExecutionConfig
from rabbit_hunter.risk_engine.position_sizing import Order
from .base import BaseExecutor, Fill


class BacktestExecutor(BaseExecutor):
    def __init__(self, cfg: ExecutionConfig):
        self.cfg = cfg

    def _slippage(self, atr: float) -> float:
        return self.cfg.slippage_atr_multiplier * atr

    def submit(self, order: Order, next_bar: dict, atr: float) -> Fill:
        open_price = float(next_bar["open"])
        slip = self._slippage(atr)
        if order.side == "long":
            fill_price = open_price + slip
        else:
            fill_price = open_price - slip
        notional = fill_price * order.size
        fees = notional * self.cfg.fees.taker
        return Fill(
            symbol=order.symbol,
            side=order.side,
            fill_price=fill_price,
            size=order.size,
            timestamp=int(next_bar["timestamp"]),
            fees=fees,
            slippage=slip,
            reason="entry",
        )

    def close_at(
        self,
        symbol: str,
        side: Literal["long", "short"],
        size: float,
        price: float,
        timestamp: int,
        atr: float,
        reason: str,
        is_taker: bool = True,
    ) -> Fill:
        slip = self._slippage(atr)
        # 平多 = 卖 → 减滑点；平空 = 买 → 加滑点
        fill_price = price - slip if side == "long" else price + slip
        rate = self.cfg.fees.taker if is_taker else self.cfg.fees.maker
        fees = fill_price * size * rate
        return Fill(
            symbol=symbol,
            side=side,
            fill_price=fill_price,
            size=size,
            timestamp=timestamp,
            fees=fees,
            slippage=slip,
            reason=reason,
        )

    def apply_funding(self, position_size: float, price: float, funding_rate: float) -> float:
        """position_size > 0 表示多头，< 0 表示空头。返回资金费带来的 pnl delta（多头付/收）。"""
        if not self.cfg.funding_settlement or funding_rate is None:
            return 0.0
        # 惯例：funding 为正 → 多头付空头
        return -position_size * price * funding_rate
