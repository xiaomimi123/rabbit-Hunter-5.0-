"""ShadowRunner — the shadow-mode main loop.

Wraps the same six-layer pipeline used by BacktestEngine:
  circuit_breaker → check_exits → hard_rules → route → risk → paper_execute

The critical invariant is that ALL of Feature Engine / Scoring / Router /
Risk code is byte-for-byte the same as backtest — the only difference is
the data source (REST poll here vs. cached Parquet in backtest) and the
executor class (PaperExecutor here vs. BacktestExecutor there — same
fill logic under the hood).

MVP scope:
  - REST polling every N seconds (not WebSocket). Simpler, more robust.
  - Fetches last ~200 bars on each symbol per tick; the Feature Engine
    handles warmup internally.
  - Persists Ledger + last-seen-timestamp map so a restart resumes cleanly.
  - Snapshot every decision to shadows/YYYY-MM-DD/*.parquet.
  - Structured JSON logs via structlog.

Out of MVP (planned Phase 4+):
  - WebSocket subscription (lower latency, avoids REST rate limits).
  - Watchdog process (independent liveness check).
  - Daily shadow_report.md diffing simulated vs. actual market.
"""
from __future__ import annotations

import json
import pickle
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from rabbit_hunter.config.schema import AppConfig
from rabbit_hunter.scoring_engine.base import BaseStrategy
from rabbit_hunter.scoring_engine import pass_hard_rules, HardRulesParams
from rabbit_hunter.strategy_router.router import StrategyRouter
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext
from rabbit_hunter.risk_engine.portfolio_risk import PortfolioRiskEngine
from rabbit_hunter.risk_engine.circuit_breaker import CircuitBreaker
from rabbit_hunter.execution_engine.paper_executor import PaperExecutor
from rabbit_hunter.backtest.ledger import Ledger, TrailingConfig
from rabbit_hunter.data_engine.okx_fetcher import fetch_ohlcv
from rabbit_hunter.data_engine.binance_funding import fetch_funding_rate_history_binance
from rabbit_hunter.feature_engine.pipeline import build_features
from rabbit_hunter.observability.logger import get_logger


@dataclass
class ShadowConfig:
    """Runtime knobs for the shadow loop. All defaults tuned for MVP-ok."""
    poll_interval_seconds: int = 60
    # How many bars back to fetch on every tick. Must exceed the Feature
    # Engine's slowest warmup (regime baseline uses rolling 500).
    lookback_bars: int = 600
    # State root where ledger + last-seen + snapshot files live.
    state_dir: Path = field(default_factory=lambda: Path("shadows"))


class ShadowRunner:
    def __init__(
        self,
        app_config: AppConfig,
        strategies: list[BaseStrategy],
        shadow_config: ShadowConfig | None = None,
    ):
        self.cfg = app_config
        self.shadow_cfg = shadow_config or ShadowConfig()
        self.log = get_logger("shadow.runner")

        strategy_weights = {
            name: {"weight": entry.weight}
            for name, entry in app_config.strategy_router.enabled_strategies.items()
        }
        self.router = StrategyRouter(
            composer=app_config.strategy_router.composer,
            strategy_weights=strategy_weights,
            strategies=strategies,
        )
        self.risk = RiskEngine(app_config.risk)
        self.executor = PaperExecutor(app_config.execution)
        self._trailing_cfg = TrailingConfig(
            enabled=app_config.risk.trailing_enabled,
            activation_r=app_config.risk.trailing_activation_r,
            atr_multiplier=app_config.risk.trailing_atr_multiplier,
        )
        self.circuit_breaker = CircuitBreaker(app_config.circuit_breaker)
        self.hard_rules_cfg = app_config.hard_rules
        self._hard_rules_params = HardRulesParams(
            min_quote_volume_24h=app_config.hard_rules.min_quote_volume_24h,
            atr_pct_max_multiplier=app_config.hard_rules.atr_pct_max_multiplier,
            atr_pct_baseline_window=app_config.hard_rules.atr_pct_baseline_window,
        )

        # State directories
        self.state_dir = self.shadow_cfg.state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "state").mkdir(exist_ok=True)

        # Persisted state
        self.ledger = self._load_ledger()
        self.last_seen_ts = self._load_last_seen_ts()
        # PortfolioRiskEngine wants features_by_symbol; seeded lazily on first tick
        self._portfolio_risk: PortfolioRiskEngine | None = None
        # Latest features by symbol (needed to compute correlations across ticks)
        self._latest_features: dict[str, pd.DataFrame] = {}

        # Graceful shutdown flag
        self._stop = False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _ledger_path(self) -> Path:
        return self.state_dir / "state" / "ledger.pkl"

    def _last_seen_path(self) -> Path:
        return self.state_dir / "state" / "last_seen_ts.json"

    def _load_ledger(self) -> Ledger:
        p = self._ledger_path()
        if p.exists():
            with p.open("rb") as f:
                ledger = pickle.load(f)
            self.log.info("ledger_resumed",
                          equity=ledger.equity,
                          open_positions=len(ledger.open_positions),
                          closed_trades=len(ledger.closed_trades))
            return ledger
        self.log.info("ledger_new", initial_capital=self.cfg.backtest.initial_capital)
        return Ledger(initial_capital=self.cfg.backtest.initial_capital)

    def _save_ledger(self):
        with self._ledger_path().open("wb") as f:
            pickle.dump(self.ledger, f)

    def _load_last_seen_ts(self) -> dict[str, int]:
        p = self._last_seen_path()
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return {}

    def _save_last_seen_ts(self):
        self._last_seen_path().write_text(
            json.dumps(self.last_seen_ts, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def _snapshot_path(self, ts_ms: int) -> Path:
        day = pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%d")
        d = self.state_dir / day
        d.mkdir(parents=True, exist_ok=True)
        return d / "snapshots.parquet"

    def _write_snapshot(self, record: dict):
        path = self._snapshot_path(record["timestamp"])
        # Append: read existing (if any) + concat + write.
        # For MVP scale (~10-100 snapshots/day) this is fine.
        row = pd.DataFrame([record])
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, row], ignore_index=True)
        else:
            combined = row
        combined.to_parquet(path, index=False)

    # ------------------------------------------------------------------
    # Data fetch — one symbol, latest bars
    # ------------------------------------------------------------------
    def _fetch_recent_features(self, symbol: str) -> pd.DataFrame | None:
        """Pull last N bars via REST and run the Feature Engine on them.

        Returns None if fetch fails or too few bars are available.
        """
        interval_ms = {"1H": 3_600_000, "15m": 900_000}[self.cfg.data.main_interval]
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - self.shadow_cfg.lookback_bars * interval_ms

        try:
            raw = fetch_ohlcv(symbol, self.cfg.data.main_interval, start_ms, end_ms)
        except Exception as e:
            self.log.error("fetch_ohlcv_failed", symbol=symbol, error=str(e))
            return None
        if len(raw) < 200:
            self.log.warning("fetch_ohlcv_too_few_rows", symbol=symbol, n=len(raw))
            return None

        # Confirm timeframe (15m) — best-effort. If missing, features degrade
        # gracefully to no cross-timeframe signal.
        confirm = None
        if self.cfg.data.confirm_interval != self.cfg.data.main_interval:
            confirm_interval_ms = {"1H": 3_600_000, "15m": 900_000}[
                self.cfg.data.confirm_interval
            ]
            confirm_start = end_ms - 200 * confirm_interval_ms
            try:
                confirm = fetch_ohlcv(
                    symbol, self.cfg.data.confirm_interval, confirm_start, end_ms
                )
            except Exception as e:
                self.log.warning("fetch_confirm_failed", symbol=symbol, error=str(e))

        # Funding (best effort — Binance)
        funding = None
        try:
            funding_start = end_ms - 100 * 8 * 3_600_000  # ~100 funding intervals back
            funding = fetch_funding_rate_history_binance(symbol, funding_start, end_ms)
        except Exception as e:
            self.log.warning("fetch_funding_failed", symbol=symbol, error=str(e))

        if funding is not None and not funding.empty:
            raw = pd.merge_asof(
                raw.sort_values("timestamp"),
                funding.sort_values("timestamp"),
                on="timestamp", direction="backward",
            )

        return build_features(
            raw, confirm=confirm,
            engine_version=self.cfg.feature_engine.version,
        )

    # ------------------------------------------------------------------
    # Per-bar decision (mirrors BacktestEngine.run inner loop)
    # ------------------------------------------------------------------
    def _handle_bar(self, symbol: str, features_row: dict,
                    features_history: pd.DataFrame, price: float, atr: float,
                    ts: int):
        """Same decision path as BacktestEngine — no exchange calls, only
        PaperExecutor for the fill simulation."""

        # 1. Circuit breaker
        cb = self.circuit_breaker.check(features_row)
        if cb.tripped:
            if (self.cfg.circuit_breaker.emergency_close_on_shock
                    and symbol in self.ledger.open_positions):
                pos = self.ledger.open_positions[symbol]
                fill = self.executor.close_at(
                    symbol, pos.side, pos.size, price, ts, atr,
                    reason="circuit_breaker",
                )
                self.ledger.record_exit(fill, exit_snapshot=features_row)
            self._write_snapshot({
                "timestamp": ts, "symbol": symbol, "action": "circuit_breaker",
                "conviction": 0.0, "circuit_breaker_reason": cb.reason,
                "atr_ratio": cb.atr_ratio,
            })
            return

        # 2. Check exits on any open position (uses "current bar" here because
        # this is real time — no "next bar" yet)
        if symbol in self.ledger.open_positions:
            trades = self.ledger.check_exits(
                symbol=symbol,
                bar={"timestamp": ts, "open": features_row["open"],
                     "high": features_row["high"], "low": features_row["low"],
                     "close": features_row["close"]},
                atr=atr,
                executor=self.executor,
                hold_timeout_bars=self.cfg.risk.hold_timeout_bars,
                exit_snapshot_fn=lambda r=features_row: r,
                trailing=self._trailing_cfg,
            )
            if trades:
                self.log.info("shadow_exit", symbol=symbol,
                              trades=[t["exit_reason"] for t in trades],
                              equity=self.ledger.equity)

        if symbol in self.ledger.open_positions:
            # Still holding — don't open new
            return

        # 3. Hard rules gate
        if self.hard_rules_cfg.enabled:
            ok, reasons = pass_hard_rules(features_row, self._hard_rules_params)
            if not ok:
                self._write_snapshot({
                    "timestamp": ts, "symbol": symbol, "action": "hard_reject",
                    "conviction": 0.0, "hard_reject_reasons": reasons,
                })
                return

        # 4. Route strategies
        intent = self.router.route(
            symbol=symbol,
            features_row=features_row,
            features_history=features_history,
            open_action_threshold=self.cfg.strategy_router.open_action_threshold,
        )
        # 5. Risk sizing
        ctx = RiskContext(
            equity=self.ledger.equity,
            atr=atr, price=price,
            daily_realized_pnl=0.0,  # MVP: no daily circuit in shadow (state doesn't span days yet)
            initial_capital=self.cfg.backtest.initial_capital,
            open_positions_count=len(self.ledger.open_positions),
        )
        order = self.risk.size(intent, ctx)

        # 6. Portfolio-level risk
        portfolio_reasons: list[str] = []
        portfolio_mult = 1.0
        if order is not None and self._portfolio_risk is not None:
            pr = self._portfolio_risk.evaluate(
                candidate=order, open_positions=self.ledger.open_positions,
                equity=self.ledger.equity,
            )
            portfolio_reasons = pr.reasons
            portfolio_mult = pr.size_multiplier
            order = pr.adjusted_order if pr.accepted else None

        self._write_snapshot({
            "timestamp": ts, "symbol": symbol,
            "action": intent.action, "conviction": intent.conviction,
            "long_score": json.dumps(intent.contributing_strategies, default=str),
            "order_placed": order is not None,
            "portfolio_reasons": ",".join(portfolio_reasons) if portfolio_reasons else "",
            "portfolio_multiplier": portfolio_mult,
        })

        if order is None:
            return

        # 7. Paper execute using CURRENT bar (real-time — no "next bar" available)
        current_bar = {
            "timestamp": ts, "open": features_row["close"],  # fill at current close as proxy
            "high": features_row["high"], "low": features_row["low"],
            "close": features_row["close"],
        }
        fill = self.executor.submit(order, current_bar, atr)
        self.ledger.record_entry(
            fill=fill, entry_snapshot=features_row,
            strategy_scores=intent.contributing_strategies,
            stop=order.stop_price, take_profit=order.take_profit_price,
        )
        self.log.info("shadow_entry",
                      symbol=symbol, side=fill.side, price=fill.fill_price,
                      size=fill.size, equity=self.ledger.equity)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def tick(self):
        """One iteration: for each symbol, fetch latest features, process
        any new bar, save state."""
        symbols = self.cfg.data.symbols
        newly_processed = 0

        for symbol in symbols:
            feats = self._fetch_recent_features(symbol)
            if feats is None or feats.empty:
                continue
            self._latest_features[symbol] = feats

            latest = feats.iloc[-1].to_dict()
            latest_ts = int(latest["timestamp"])

            if self.last_seen_ts.get(symbol, 0) >= latest_ts:
                continue  # already processed this bar

            # Skip early bars still in warmup — indicators are NaN
            if pd.isna(latest.get("atr_14")) or pd.isna(latest.get("ema200")):
                continue

            price = float(latest["close"])
            atr = float(latest.get("atr_14") or 0.0)
            history = feats.iloc[-200:].reset_index(drop=True)

            # Ensure PortfolioRiskEngine has all symbols by now (recreate each tick;
            # cheap since correlation matrix is small)
            self._portfolio_risk = PortfolioRiskEngine(
                self.cfg.portfolio_risk, dict(self._latest_features)
            )

            self._handle_bar(
                symbol=symbol, features_row=latest,
                features_history=history, price=price, atr=atr, ts=latest_ts,
            )
            self.last_seen_ts[symbol] = latest_ts
            newly_processed += 1

        if newly_processed > 0:
            self._save_ledger()
            self._save_last_seen_ts()
            self.log.info("tick_done",
                          processed=newly_processed,
                          equity=self.ledger.equity,
                          open_positions=len(self.ledger.open_positions))

    def _install_signal_handlers(self):
        def handler(signum, frame):
            self.log.info("shadow_stop_requested", signal=signum)
            self._stop = True
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def run_forever(self):
        """Run the tick loop until interrupted (Ctrl-C or SIGTERM)."""
        self._install_signal_handlers()
        self.log.info("shadow_started",
                      symbols=list(self.cfg.data.symbols),
                      main_interval=self.cfg.data.main_interval,
                      poll_interval_seconds=self.shadow_cfg.poll_interval_seconds)
        try:
            while not self._stop:
                try:
                    self.tick()
                except Exception as e:
                    # Never exit the loop on an unexpected error — log and continue
                    self.log.error("tick_error", error=str(e), type=type(e).__name__)
                time.sleep(self.shadow_cfg.poll_interval_seconds)
        finally:
            self._save_ledger()
            self._save_last_seen_ts()
            self.log.info("shadow_stopped",
                          equity=self.ledger.equity,
                          open_positions=len(self.ledger.open_positions),
                          closed_trades=len(self.ledger.closed_trades))
