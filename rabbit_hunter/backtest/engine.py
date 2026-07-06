from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd
from rabbit_hunter.config.schema import AppConfig
from rabbit_hunter.scoring_engine.base import BaseStrategy
from rabbit_hunter.scoring_engine import pass_hard_rules, HardRulesParams
from rabbit_hunter.strategy_router.router import StrategyRouter
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext
from rabbit_hunter.risk_engine.portfolio_risk import PortfolioRiskEngine
from rabbit_hunter.risk_engine.circuit_breaker import CircuitBreaker
from rabbit_hunter.risk_engine.btc_crash_booster import BtcCrashBooster
from rabbit_hunter.execution_engine.backtest_executor import BacktestExecutor
from .ledger import Ledger, TrailingConfig


@dataclass
class BacktestResult:
    ledger: Ledger
    snapshots: pd.DataFrame  # 每次决策一行
    equity_curve: pd.DataFrame  # timestamp, equity


class BacktestEngine:
    def __init__(
        self,
        app_config: AppConfig,
        strategies: list[BaseStrategy],
    ):
        self.cfg = app_config
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
        self.executor = BacktestExecutor(app_config.execution)
        self.hard_rules_cfg = app_config.hard_rules
        self._hard_rules_params = HardRulesParams(
            min_quote_volume_24h=app_config.hard_rules.min_quote_volume_24h,
            atr_pct_max_multiplier=app_config.hard_rules.atr_pct_max_multiplier,
            atr_pct_baseline_window=app_config.hard_rules.atr_pct_baseline_window,
        )
        self._trailing_cfg = TrailingConfig(
            enabled=app_config.risk.trailing_enabled,
            activation_r=app_config.risk.trailing_activation_r,
            atr_multiplier=app_config.risk.trailing_atr_multiplier,
        )
        self._circuit_breaker = CircuitBreaker(app_config.circuit_breaker)
        # PortfolioRiskEngine needs features_by_symbol which arrives in run();
        # defer construction to run() so we can seed the correlation matrix.
        self._portfolio_risk: PortfolioRiskEngine | None = None
        # v0.1.3 chop-market kill switch: rolling WR of most recent N closed
        # trades. When it drops below wr_threshold, new entries are blocked
        # for pause_bars main-interval bars.
        self._chop_cfg = app_config.chop_kill_switch
        self._recent_pnl: deque[float] = deque(maxlen=self._chop_cfg.window)
        self._paused_until_ts: int | None = None
        # 1H = 3_600_000 ms (main_interval currently hard-locked to 1H).
        self._bar_ms = 3_600_000
        # v0.1.3 BTC-crash size booster: press the edge on systemic drops.
        self._btc_booster = BtcCrashBooster(app_config.btc_crash_boost)

    def run(
        self,
        features_by_symbol: dict[str, pd.DataFrame],
        open_action_threshold: float | None = None,
    ) -> BacktestResult:
        # If caller doesn't pass an explicit threshold, use the router config
        # value. Explicit param still wins so tests can override.
        if open_action_threshold is None:
            open_action_threshold = self.cfg.strategy_router.open_action_threshold
        ledger = Ledger(initial_capital=self.cfg.backtest.initial_capital)
        # Seed portfolio risk with the features frame (needed for correlations)
        self._portfolio_risk = PortfolioRiskEngine(
            self.cfg.portfolio_risk, features_by_symbol
        )

        # 合并所有 symbol 的 timestamps 并按顺序处理
        all_ts = sorted({int(ts) for df in features_by_symbol.values() for ts in df["timestamp"]})

        # 建索引：{symbol: {ts: row_idx}}
        indexes = {sym: {int(ts): i for i, ts in enumerate(df["timestamp"])} for sym, df in features_by_symbol.items()}

        snapshots: list[dict] = []
        equity_points: list[dict] = []
        daily_realized: dict[str, float] = {}
        last_day: str | None = None

        for ts in all_ts:
            day_str = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d")
            if day_str != last_day:
                daily_realized[day_str] = 0.0
                last_day = day_str

            prices_at_ts: dict[str, float] = {}

            for symbol, feats in features_by_symbol.items():
                idx = indexes[symbol].get(ts)
                if idx is None or idx + 1 >= len(feats):
                    continue
                row = feats.iloc[idx].to_dict()
                next_bar = feats.iloc[idx + 1].to_dict()
                price = float(row["close"])
                atr_raw = row.get("atr_14")
                atr = float(atr_raw) if pd.notna(atr_raw) else 0.0
                prices_at_ts[symbol] = price

                # === Phase 3: Circuit breaker (extreme volatility) ===
                # Runs BEFORE anything else. If tripped, emergency-close any
                # position on this symbol and refuse new entries this bar.
                cb_result = self._circuit_breaker.check(row)
                if cb_result.tripped:
                    if (self.cfg.circuit_breaker.emergency_close_on_shock
                            and symbol in ledger.open_positions):
                        pos = ledger.open_positions[symbol]
                        fill = self.executor.close_at(
                            symbol=symbol, side=pos.side, size=pos.size,
                            price=price, timestamp=ts, atr=atr,
                            reason="circuit_breaker",
                        )
                        trade = ledger.record_exit(fill, exit_snapshot=row)
                        daily_realized[day_str] += trade["pnl_after_fees"]
                        self._on_trade_closed(trade, ts)
                    snapshots.append({
                        "timestamp": ts,
                        "symbol": symbol,
                        "action": "circuit_breaker",
                        "conviction": 0.0,
                        "long_score": {},
                        "order_placed": False,
                        "circuit_breaker_reason": cb_result.reason,
                        "atr_ratio": cb_result.atr_ratio,
                    })
                    continue

                # 结算 funding（每 8 小时）
                dt = pd.to_datetime(ts, unit="ms", utc=True)
                if dt.hour % 8 == 0 and dt.minute == 0:
                    fr = row.get("funding_rate")
                    if fr is not None and pd.notna(fr):
                        ledger.apply_funding(symbol, price, float(fr), self.executor)

                # 检查已有仓位是否触发止损/止盈/超时/移动止损
                if symbol in ledger.open_positions:
                    exit_snapshot_fn = lambda r=row: r
                    trades = ledger.check_exits(
                        symbol=symbol,
                        bar=next_bar,
                        atr=atr,
                        executor=self.executor,
                        hold_timeout_bars=self.cfg.risk.hold_timeout_bars,
                        exit_snapshot_fn=exit_snapshot_fn,
                        trailing=self._trailing_cfg,
                    )
                    for t in trades:
                        daily_realized[day_str] += t["pnl_after_fees"]
                        self._on_trade_closed(t, ts)

                # 如果已有该 symbol 仓位，不再开新单
                if symbol in ledger.open_positions:
                    continue

                # v0.1.3 chop-market kill switch: block new entries while
                # paused. Existing positions still exit normally above.
                if self._chop_cfg.enabled and self._paused_until_ts is not None \
                        and ts < self._paused_until_ts:
                    snapshots.append({
                        "timestamp": ts,
                        "symbol": symbol,
                        "action": "chop_paused",
                        "conviction": 0.0,
                        "long_score": {},
                        "order_placed": False,
                        "chop_paused_until_ts": self._paused_until_ts,
                    })
                    continue

                # 硬规则闸门：流动性/极端波动率过滤，拒绝的行不算作交易决策
                if self.hard_rules_cfg.enabled:
                    ok, reasons = pass_hard_rules(row, self._hard_rules_params)
                    if not ok:
                        snapshots.append({
                            "timestamp": ts,
                            "symbol": symbol,
                            "action": "hard_reject",
                            "conviction": 0.0,
                            "long_score": {},
                            "order_placed": False,
                            "hard_reject_reasons": reasons,
                        })
                        continue

                intent = self.router.route(
                    symbol=symbol,
                    features_row=row,
                    features_history=feats.iloc[max(0, idx - 200): idx + 1],
                    open_action_threshold=open_action_threshold,
                )
                ctx = RiskContext(
                    equity=ledger.equity,
                    atr=atr,
                    price=price,
                    daily_realized_pnl=daily_realized[day_str],
                    initial_capital=self.cfg.backtest.initial_capital,
                    open_positions_count=len(ledger.open_positions),
                )
                order = self.risk.size(intent, ctx)

                # === Phase 3: Portfolio-level risk gate ===
                portfolio_reasons: list[str] = []
                portfolio_multiplier = 1.0
                if order is not None and self._portfolio_risk is not None:
                    pr = self._portfolio_risk.evaluate(
                        candidate=order,
                        open_positions=ledger.open_positions,
                        equity=ledger.equity,
                    )
                    portfolio_reasons = pr.reasons
                    portfolio_multiplier = pr.size_multiplier
                    if not pr.accepted:
                        order = None
                    else:
                        order = pr.adjusted_order

                # === v0.1.3: BTC crash size booster ===
                # Runs AFTER portfolio_risk so its output size respects the
                # gross-leverage cap. We fetch the current-bar BTC row from
                # features_by_symbol (may be missing if BTC not in the basket).
                btc_boost_reason = ""
                btc_boost_mult = 1.0
                if order is not None:
                    btc_sym = self.cfg.btc_crash_boost.btc_symbol
                    btc_row = None
                    btc_prev_close = None
                    btc_feats = features_by_symbol.get(btc_sym)
                    if btc_feats is not None:
                        btc_idx = indexes.get(btc_sym, {}).get(ts)
                        if btc_idx is not None and btc_idx > 0:
                            btc_row = btc_feats.iloc[btc_idx].to_dict()
                            btc_prev_close = float(btc_feats.iloc[btc_idx - 1]["close"])
                    boost = self._btc_booster.evaluate(
                        candidate=order,
                        btc_row=btc_row,
                        btc_prev_close=btc_prev_close,
                        equity=ledger.equity,
                    )
                    if boost.boosted:
                        order = boost.adjusted_order
                        btc_boost_reason = boost.reason
                        btc_boost_mult = boost.multiplier

                snap = {
                    "timestamp": ts,
                    "symbol": symbol,
                    "action": intent.action,
                    "conviction": intent.conviction,
                    "long_score": intent.contributing_strategies,
                    "order_placed": order is not None,
                    "portfolio_risk_reasons": portfolio_reasons,
                    "portfolio_size_multiplier": portfolio_multiplier,
                    "btc_boost_reason": btc_boost_reason,
                    "btc_boost_multiplier": btc_boost_mult,
                }
                snapshots.append(snap)

                if order is None:
                    continue
                fill = self.executor.submit(order, next_bar, atr)
                ledger.record_entry(
                    fill=fill,
                    entry_snapshot=row,
                    strategy_scores=intent.contributing_strategies,
                    stop=order.stop_price,
                    take_profit=order.take_profit_price,
                )

            eq_now = ledger.mark_to_market(prices_at_ts)
            equity_points.append({"timestamp": ts, "equity": eq_now})

        snap_df = pd.DataFrame(snapshots)
        eq_df = pd.DataFrame(equity_points)
        return BacktestResult(ledger=ledger, snapshots=snap_df, equity_curve=eq_df)

    def _on_trade_closed(self, trade: dict, ts: int) -> None:
        """Update the rolling-WR window and, if it drops below threshold,
        engage the chop kill switch for the configured pause_bars."""
        if not self._chop_cfg.enabled:
            return
        pnl = float(trade.get("pnl_after_fees", 0.0))
        self._recent_pnl.append(pnl)
        # Only evaluate once the window is full — otherwise a single early
        # loss trips it immediately.
        if len(self._recent_pnl) < self._chop_cfg.window:
            return
        wr = sum(1 for x in self._recent_pnl if x > 0) / self._chop_cfg.window
        if wr < self._chop_cfg.wr_threshold:
            self._paused_until_ts = ts + self._chop_cfg.pause_bars * self._bar_ms
            # Reset window so the next evaluation is measured on trades that
            # closed AFTER the pause — otherwise the same losers keep re-tripping.
            self._recent_pnl.clear()
