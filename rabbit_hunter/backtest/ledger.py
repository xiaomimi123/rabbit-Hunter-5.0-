from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
from rabbit_hunter.execution_engine.base import Fill
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor


@dataclass
class Position:
    symbol: str
    side: Literal["long", "short"]
    entry_time: int
    entry_price: float
    size: float
    fees_paid: float
    stop: float
    take_profit: float
    entry_snapshot: dict
    strategy_scores: dict
    bars_held: int = 0
    funding_accum: float = 0.0


@dataclass
class Ledger:
    initial_capital: float
    equity: float = 0.0
    open_positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.equity == 0.0:
            self.equity = self.initial_capital

    def record_entry(
        self,
        fill: Fill,
        entry_snapshot: dict,
        strategy_scores: dict,
        stop: float,
        take_profit: float,
    ):
        pos = Position(
            symbol=fill.symbol,
            side=fill.side,
            entry_time=fill.timestamp,
            entry_price=fill.fill_price,
            size=fill.size,
            fees_paid=fill.fees,
            stop=stop,
            take_profit=take_profit,
            entry_snapshot=entry_snapshot,
            strategy_scores=strategy_scores,
        )
        self.open_positions[fill.symbol] = pos
        self.equity -= fill.fees

    def record_exit(self, fill: Fill, exit_snapshot: dict) -> dict:
        pos = self.open_positions.pop(fill.symbol)
        if pos.side == "long":
            pnl_raw = (fill.fill_price - pos.entry_price) * pos.size
        else:
            pnl_raw = (pos.entry_price - fill.fill_price) * pos.size
        fees_total = pos.fees_paid + fill.fees
        self.equity += pnl_raw - fill.fees
        trade = {
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_time": pos.entry_time,
            "exit_time": fill.timestamp,
            "entry_price": pos.entry_price,
            "exit_price": fill.fill_price,
            "size": pos.size,
            "pnl_raw": pnl_raw,
            "pnl_after_fees": pnl_raw - fees_total + pos.funding_accum,
            "fees": fees_total,
            "funding": pos.funding_accum,
            "slippage": fill.slippage,
            "hold_bars": pos.bars_held,
            "exit_reason": fill.reason,
            "entry_snapshot": pos.entry_snapshot,
            "exit_snapshot": exit_snapshot,
            "strategy_scores": pos.strategy_scores,
        }
        self.closed_trades.append(trade)
        return trade

    def check_exits(
        self,
        symbol: str,
        bar: dict,
        atr: float,
        executor: BacktestExecutor,
        hold_timeout_bars: int,
        exit_snapshot_fn: Callable[[], dict],
    ) -> list[dict]:
        results: list[dict] = []
        if symbol not in self.open_positions:
            return results
        pos = self.open_positions[symbol]
        pos.bars_held += 1

        high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
        ts = int(bar["timestamp"])

        # 检测触发
        if pos.side == "long":
            hit_stop = low <= pos.stop
            hit_tp = high >= pos.take_profit
            price = pos.stop if hit_stop else (pos.take_profit if hit_tp else close)
        else:
            hit_stop = high >= pos.stop
            hit_tp = low <= pos.take_profit
            price = pos.stop if hit_stop else (pos.take_profit if hit_tp else close)

        reason = None
        if hit_stop and hit_tp:
            reason = "stop_loss"  # 保守：同 bar 内两者都触发 → 假设先触发止损
        elif hit_stop:
            reason = "stop_loss"
        elif hit_tp:
            reason = "take_profit"
        elif pos.bars_held >= hold_timeout_bars:
            reason = "timeout"; price = close

        if reason:
            fill = executor.close_at(symbol, pos.side, pos.size, price, ts, atr, reason)
            trade = self.record_exit(fill, exit_snapshot_fn())
            results.append(trade)
        return results

    def apply_funding(self, symbol: str, price: float, funding_rate: float, executor: BacktestExecutor):
        if symbol not in self.open_positions:
            return
        pos = self.open_positions[symbol]
        signed_size = pos.size if pos.side == "long" else -pos.size
        delta = executor.apply_funding(signed_size, price, funding_rate)
        pos.funding_accum += delta
        self.equity += delta

    def mark_to_market(self, prices: dict[str, float]) -> float:
        eq = self.equity
        for sym, pos in self.open_positions.items():
            if sym not in prices:
                continue
            p = prices[sym]
            if pos.side == "long":
                eq += (p - pos.entry_price) * pos.size
            else:
                eq += (pos.entry_price - p) * pos.size
        return eq
