"""ShadowMetrics — per-tick snapshot of the shadow runner's health.

At the end of every tick, ShadowRunner asks this module for a metrics
snapshot (equity, positions, PnL, alerts). The snapshot is:
  1. Emitted to stdout via structlog (so operators tailing the container
     see live values, not just event logs).
  2. Appended to `state/metrics_history.parquet` so the HTML dashboard has
     a real time series to plot instead of only current state.

Alert rules are hard thresholds — no ML, no smoothing — so an operator
reading the log can immediately tell what tripped and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rabbit_hunter.backtest.ledger import Ledger


# ============================================================
# Alert thresholds — hard values. Tuned for the current 10k-USD
# shadow config; adjust per deployment via ShadowConfig if needed.
# ============================================================

@dataclass(frozen=True)
class AlertThresholds:
    # Fire if equity has fallen this fraction from the peak seen in this run.
    max_drawdown_pct: float = 0.10  # 10%
    # Fire if a position has been open longer than this many hours.
    stuck_position_hours: float = 72.0
    # Fire if wall clock is this many minutes past the last processed bar.
    stale_data_minutes: float = 120.0
    # Fire if consecutive tick errors exceed this.
    consecutive_error_limit: int = 3


# ============================================================
# Snapshot value type
# ============================================================

@dataclass
class MetricsSnapshot:
    timestamp_ms: int
    equity: float
    initial_capital: float
    total_pnl: float           # equity - initial_capital
    pnl_pct: float             # / initial_capital
    peak_equity: float         # highest equity ever seen
    drawdown_from_peak_pct: float
    open_positions: int
    open_notional: float
    open_long_notional: float
    open_short_notional: float
    total_closed_trades: int
    winners: int
    losers: int
    winrate: float             # 0.0 if no closed trades
    profit_factor: float       # inf if no losses; 0 if no wins
    last_bar_ts_ms: int | None
    minutes_since_last_bar: float | None
    consecutive_errors: int
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "alerts"} | {
            "alerts": ",".join(self.alerts) if self.alerts else "",
            "alert_count": len(self.alerts),
        }


# ============================================================
# Metrics collector — holds cross-tick state (peak equity, error streak)
# ============================================================

class ShadowMetrics:
    """Compute a MetricsSnapshot from live ledger + runner state.

    The collector owns two pieces of state that need to survive across ticks:
      - peak_equity (for max-drawdown tracking)
      - consecutive_errors (for run-health alerting)

    Everything else is derived from the ledger + latest bar timestamps that
    the runner already tracks — no double-counting.
    """

    def __init__(
        self,
        thresholds: AlertThresholds | None = None,
        state_dir: Path | None = None,
    ):
        self.thresholds = thresholds or AlertThresholds()
        self.state_dir = state_dir
        self.peak_equity: float = 0.0
        self.consecutive_errors: int = 0
        # If a state_dir was passed, try to hydrate peak_equity from history
        # so a restart doesn't reset the drawdown reference to today.
        if self.state_dir is not None:
            self._hydrate_peak_from_history()

    def _history_path(self) -> Path | None:
        if self.state_dir is None:
            return None
        return self.state_dir / "state" / "metrics_history.parquet"

    def _hydrate_peak_from_history(self) -> None:
        p = self._history_path()
        if p is None or not p.exists():
            return
        try:
            hist = pd.read_parquet(p, columns=["peak_equity", "equity"])
            if not hist.empty:
                self.peak_equity = float(max(
                    hist["peak_equity"].max(), hist["equity"].max()
                ))
        except Exception:
            # Corrupt history file should not crash startup — start fresh.
            pass

    def note_tick_error(self) -> None:
        self.consecutive_errors += 1

    def note_tick_success(self) -> None:
        self.consecutive_errors = 0

    def snapshot(
        self,
        ledger: Ledger,
        prices: dict[str, float],
        last_seen_ts: dict[str, int],
        now_ms: int | None = None,
    ) -> MetricsSnapshot:
        """Compute the current snapshot. `prices` is a symbol → last-known
        close map used to mark open positions; `last_seen_ts` is what the
        runner already keeps for skip-if-bar-already-seen bookkeeping."""
        if now_ms is None:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Refresh equity mark using current prices (does not modify ledger)
        equity = ledger.mark_to_market(prices)
        if equity > self.peak_equity:
            self.peak_equity = equity

        total_pnl = equity - ledger.initial_capital
        pnl_pct = total_pnl / ledger.initial_capital if ledger.initial_capital > 0 else 0.0
        drawdown = ((self.peak_equity - equity) / self.peak_equity
                    if self.peak_equity > 0 else 0.0)

        long_notional = 0.0
        short_notional = 0.0
        for sym, pos in ledger.open_positions.items():
            mark_px = prices.get(sym, pos.entry_price)
            notional = mark_px * pos.size
            if pos.side == "long":
                long_notional += notional
            else:
                short_notional += notional

        winners = sum(1 for t in ledger.closed_trades if t["pnl_after_fees"] > 0)
        losers = sum(1 for t in ledger.closed_trades if t["pnl_after_fees"] < 0)
        total_closed = len(ledger.closed_trades)
        winrate = winners / total_closed if total_closed else 0.0
        gross_win = sum(t["pnl_after_fees"] for t in ledger.closed_trades
                        if t["pnl_after_fees"] > 0)
        gross_loss = -sum(t["pnl_after_fees"] for t in ledger.closed_trades
                          if t["pnl_after_fees"] < 0)
        if gross_loss > 0:
            profit_factor = gross_win / gross_loss
        elif gross_win > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        last_bar_ts = max(last_seen_ts.values()) if last_seen_ts else None
        minutes_since = ((now_ms - last_bar_ts) / 60_000.0
                         if last_bar_ts is not None else None)

        alerts = self._evaluate_alerts(
            drawdown=drawdown,
            open_positions=ledger.open_positions,
            now_ms=now_ms,
            minutes_since_last_bar=minutes_since,
        )

        return MetricsSnapshot(
            timestamp_ms=now_ms,
            equity=float(equity),
            initial_capital=float(ledger.initial_capital),
            total_pnl=float(total_pnl),
            pnl_pct=float(pnl_pct),
            peak_equity=float(self.peak_equity),
            drawdown_from_peak_pct=float(drawdown),
            open_positions=len(ledger.open_positions),
            open_notional=float(long_notional + short_notional),
            open_long_notional=float(long_notional),
            open_short_notional=float(short_notional),
            total_closed_trades=total_closed,
            winners=winners,
            losers=losers,
            winrate=float(winrate),
            profit_factor=float(profit_factor),
            last_bar_ts_ms=last_bar_ts,
            minutes_since_last_bar=minutes_since,
            consecutive_errors=self.consecutive_errors,
            alerts=alerts,
        )

    def _evaluate_alerts(
        self,
        drawdown: float,
        open_positions: dict,
        now_ms: int,
        minutes_since_last_bar: float | None,
    ) -> list[str]:
        alerts: list[str] = []
        t = self.thresholds

        if drawdown >= t.max_drawdown_pct:
            alerts.append(
                f"drawdown_high={drawdown*100:.2f}%>={t.max_drawdown_pct*100:.0f}%"
            )

        for sym, pos in open_positions.items():
            entry_time = pos.entry_time if pos.entry_time is not None else 0
            hours_open = (now_ms - entry_time) / 3_600_000.0
            if hours_open >= t.stuck_position_hours:
                alerts.append(
                    f"position_stuck:{sym}={hours_open:.1f}h>={t.stuck_position_hours:.0f}h"
                )

        if (minutes_since_last_bar is not None
                and minutes_since_last_bar >= t.stale_data_minutes):
            alerts.append(
                f"stale_data={minutes_since_last_bar:.1f}min>={t.stale_data_minutes:.0f}min"
            )

        if self.consecutive_errors >= t.consecutive_error_limit:
            alerts.append(
                f"consecutive_errors={self.consecutive_errors}>={t.consecutive_error_limit}"
            )

        return alerts

    def append_history(self, snapshot: MetricsSnapshot) -> Path | None:
        """Append this snapshot to metrics_history.parquet. Returns the path
        written (or None if no state_dir configured)."""
        p = self._history_path()
        if p is None:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([snapshot.to_dict()])
        if p.exists():
            existing = pd.read_parquet(p)
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row
        combined.to_parquet(p, index=False)
        return p
