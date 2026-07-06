from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
import pandas as pd
from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput


Action = Literal["open_long", "open_short", "close", "wait"]


@dataclass(frozen=True)
class Intent:
    symbol: str
    action: Action
    conviction: float
    contributing_strategies: dict[str, float] = field(default_factory=dict)
    features_snapshot: dict = field(default_factory=dict)
    score_components: dict = field(default_factory=dict)


class StrategyRouter:
    def __init__(
        self,
        composer: str,
        strategy_weights: dict[str, dict],
        strategies: list[BaseStrategy],
    ):
        if composer != "weighted_avg":
            # Phase 1a 只实现 weighted_avg；其他 composer 留给 1b/1c
            raise NotImplementedError(f"composer {composer} not implemented in Phase 1a")
        self.composer = composer
        self.weights = {name: float(cfg["weight"]) for name, cfg in strategy_weights.items()}
        self.strategies = [s for s in strategies if s.name in self.weights]

    def route(
        self,
        symbol: str,
        features_row: dict,
        features_history: pd.DataFrame,
        open_action_threshold: float = 0.5,
    ) -> Intent:
        outputs: dict[str, ScoreOutput] = {}
        for s in self.strategies:
            outputs[s.name] = s.score(features_row, features_history)

        total_w = sum(self.weights[n] for n in outputs) or 1.0
        long_sum = sum(outputs[n].long * self.weights[n] for n in outputs) / total_w
        short_sum = sum(outputs[n].short * self.weights[n] for n in outputs) / total_w

        # contributing_strategies = 每个策略的 long 分（用于报告）
        contributing_display = {n: outputs[n].long for n in outputs}

        if long_sum >= open_action_threshold and long_sum > short_sum:
            action: Action = "open_long"
            conviction = long_sum
        elif short_sum >= open_action_threshold and short_sum > long_sum:
            action = "open_short"
            conviction = short_sum
        else:
            action = "wait"
            conviction = max(long_sum, short_sum)

        # 合并所有策略的 components
        merged_components: dict[str, Any] = {}
        for n, out in outputs.items():
            for k, v in out.components.items():
                merged_components[f"{n}.{k}"] = v

        return Intent(
            symbol=symbol,
            action=action,
            conviction=conviction,
            contributing_strategies=contributing_display,
            features_snapshot=dict(features_row),
            score_components=merged_components,
        )
