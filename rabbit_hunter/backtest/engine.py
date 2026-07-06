from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd
from rabbit_hunter.config.schema import AppConfig
from rabbit_hunter.scoring_engine.base import BaseStrategy
from rabbit_hunter.scoring_engine import pass_hard_rules, HardRulesParams
from rabbit_hunter.strategy_router.router import StrategyRouter
from rabbit_hunter.risk_engine.engine import RiskEngine, RiskContext
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

                # 如果已有该 symbol 仓位，不再开新单
                if symbol in ledger.open_positions:
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

                snap = {
                    "timestamp": ts,
                    "symbol": symbol,
                    "action": intent.action,
                    "conviction": intent.conviction,
                    "long_score": intent.contributing_strategies,
                    "order_placed": order is not None,
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
