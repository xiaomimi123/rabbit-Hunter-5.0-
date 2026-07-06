"""v0.1.3 · BTC-crash size booster.

Cluster analysis of the 273-trade backtest showed 10 "mass-crash" days
(≥5 symbols entering Cluster-1 same day) delivered $21/trade avg vs $9 on
other days — 2.4× edge. This module presses that edge: on bars where BTC's
zscore_20 is deeply negative, same-direction short orders get their size
scaled up by `boost_multiplier`.

The booster is deliberately narrow: only short-side orders on triggered
bars, only when BTC (the beacon) itself is confirming the crash. It does
NOT engage on isolated alt-independent crashes — those had weaker WR in
the cluster study.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from rabbit_hunter.config.schema import BtcCrashBoostConfig
from rabbit_hunter.risk_engine.position_sizing import Order


@dataclass(frozen=True)
class BtcCrashBoostResult:
    boosted: bool
    multiplier: float          # 1.0 = no boost applied
    adjusted_order: Order      # same object if no change
    reason: str = ""


class BtcCrashBooster:
    def __init__(self, cfg: BtcCrashBoostConfig):
        self.cfg = cfg

    def evaluate(
        self,
        candidate: Order,
        btc_row: dict | None,
        btc_prev_close: float | None,
        equity: float,
    ) -> BtcCrashBoostResult:
        if not self.cfg.enabled or candidate.side != "short":
            return BtcCrashBoostResult(False, 1.0, candidate)
        if btc_row is None:
            return BtcCrashBoostResult(False, 1.0, candidate)

        z = btc_row.get("zscore_20")
        close = btc_row.get("close")
        if z is None or close is None or btc_prev_close is None:
            return BtcCrashBoostResult(False, 1.0, candidate)
        z_val = float(z) if not (isinstance(z, float) and math.isnan(z)) else 0.0
        if z_val > -self.cfg.zscore_threshold:
            return BtcCrashBoostResult(False, 1.0, candidate)
        # Must also be falling (guard against a rebound bar with old z-score).
        if float(close) >= float(btc_prev_close):
            return BtcCrashBoostResult(False, 1.0, candidate)

        mult = self.cfg.boost_multiplier
        new_size = candidate.size * mult
        new_notional = new_size * candidate.entry_price
        new_leverage = new_notional / equity if equity > 0 else 0.0
        adjusted = Order(
            symbol=candidate.symbol,
            side=candidate.side,
            entry_price=candidate.entry_price,
            stop_price=candidate.stop_price,
            take_profit_price=candidate.take_profit_price,
            size=new_size,
            leverage=new_leverage,
        )
        return BtcCrashBoostResult(
            True, mult, adjusted, reason=f"btc_crash_z={z_val:.2f}"
        )
