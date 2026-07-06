from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
from rabbit_hunter.execution_engine.base import Fill
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor


@dataclass(frozen=True)
class TrailingConfig:
    """Runtime knobs for trailing-stop behavior. Passed into check_exits so
    Ledger stays config-agnostic — the risk layer decides whether trailing
    is on and at what R multiple / ATR distance."""
    enabled: bool = False
    activation_r: float = 1.0
    atr_multiplier: float = 1.0


TRAILING_OFF = TrailingConfig(enabled=False)


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
    # Trailing-stop bookkeeping (v0.2.0-scalp). initial_stop is frozen at
    # entry as the "safety net" for R-multiple math (activation threshold
    # compares profit against original 1R distance, not the current trailed
    # stop). max_favorable_price is a running high-water mark for the side.
    initial_stop: float = 0.0
    max_favorable_price: float = 0.0
    trailing_active: bool = False


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
            initial_stop=stop,
            max_favorable_price=fill.fill_price,
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
        trailing: TrailingConfig = TRAILING_OFF,
    ) -> list[dict]:
        results: list[dict] = []
        if symbol not in self.open_positions:
            return results
        pos = self.open_positions[symbol]
        pos.bars_held += 1

        high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
        ts = int(bar["timestamp"])

        # --- Trailing-stop update (v0.2.0-scalp) ---
        # Runs BEFORE the stop/TP hit check so trail movement can affect
        # this bar's exit decision. Order within the bar is intraday-lossy
        # (we can't tell whether high or low came first), so we update the
        # HW mark and trailing stop conservatively: HW uses bar.high for
        # long / bar.low for short, and any newly-computed trail stop must
        # not move backward relative to the existing stop.
        if trailing.enabled and atr > 0:
            if pos.side == "long":
                if high > pos.max_favorable_price:
                    pos.max_favorable_price = high
            else:  # short
                if low < pos.max_favorable_price or pos.max_favorable_price == 0.0:
                    # entry-time init leaves max_favorable_price = entry_price
                    # for short we want the running MIN, so first low tick
                    # will overwrite entry price too if low < entry.
                    pos.max_favorable_price = min(pos.max_favorable_price, low) if pos.max_favorable_price > 0 else low

            initial_r = abs(pos.entry_price - pos.initial_stop)
            if initial_r > 0:
                if pos.side == "long":
                    profit_r = pos.max_favorable_price - pos.entry_price
                else:
                    profit_r = pos.entry_price - pos.max_favorable_price
                if not pos.trailing_active and profit_r >= trailing.activation_r * initial_r:
                    pos.trailing_active = True

            if pos.trailing_active:
                trail_dist = trailing.atr_multiplier * atr
                if pos.side == "long":
                    new_trail = pos.max_favorable_price - trail_dist
                    if new_trail > pos.stop:
                        pos.stop = new_trail
                else:
                    new_trail = pos.max_favorable_price + trail_dist
                    if new_trail < pos.stop:
                        pos.stop = new_trail

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
            # Distinguish trailing exits from initial-stop exits in the trade
            # log so the AI reviewer can tell which is which.
            reason = "trailing_stop" if pos.trailing_active else "stop_loss"
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
