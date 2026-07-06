from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Literal["long", "short"]
    entry_price: float
    stop_price: float
    take_profit_price: float
    size: float
    leverage: float


def compute_order(
    symbol: str,
    side: Literal["long", "short"],
    price: float,
    atr: float,
    equity: float,
    risk_per_trade_pct: float,
    atr_stop_multiplier: float,
    reward_risk_ratio: float,
    max_leverage: float,
) -> Order:
    stop_distance = atr_stop_multiplier * atr
    if side == "long":
        stop = price - stop_distance
        tp = price + reward_risk_ratio * stop_distance
    else:
        stop = price + stop_distance
        tp = price - reward_risk_ratio * stop_distance

    risk_amount = equity * (risk_per_trade_pct / 100.0)
    size = risk_amount / stop_distance if stop_distance > 0 else 0.0

    notional = size * price
    max_notional = equity * max_leverage
    if notional > max_notional:
        size = max_notional / price
        notional = size * price
    leverage = notional / equity if equity > 0 else 0.0

    return Order(
        symbol=symbol, side=side, entry_price=price,
        stop_price=stop, take_profit_price=tp, size=size, leverage=leverage,
    )
