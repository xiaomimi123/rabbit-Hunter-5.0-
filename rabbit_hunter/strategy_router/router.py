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


_SUPPORTED_COMPOSERS = {"weighted_avg", "max_score"}


class StrategyRouter:
    def __init__(
        self,
        composer: str,
        strategy_weights: dict[str, dict],
        strategies: list[BaseStrategy],
        regime_rules: dict[str, Any] | None = None,
    ):
        if composer not in _SUPPORTED_COMPOSERS:
            raise NotImplementedError(
                f"composer {composer} not implemented. Supported: {sorted(_SUPPORTED_COMPOSERS)}"
            )
        self.composer = composer
        self.weights = {name: float(cfg["weight"]) for name, cfg in strategy_weights.items()}
        self.strategies = [s for s in strategies if s.name in self.weights]
        # Per-structure-regime gates. Values may be pydantic RegimeRule
        # objects or plain dicts (tests) — normalized at access time in
        # _apply_regime_rules.
        self.regime_rules = regime_rules or {}

    def _apply_regime_rules(
        self, features_row: dict, long_score: float, short_score: float,
    ) -> tuple[float, float]:
        """Zero out or scale scores according to the configured rule for
        the bar's structure_regime. Missing regime key or missing
        structure_regime feature → no-op (allow everything)."""
        if not self.regime_rules:
            return long_score, short_score
        regime = features_row.get("structure_regime")
        if regime is None:
            return long_score, short_score
        rule = self.regime_rules.get(str(regime))
        if rule is None:
            return long_score, short_score
        # Accept pydantic model or dict
        allow_long = getattr(rule, "allow_long", None)
        if allow_long is None and isinstance(rule, dict):
            allow_long = rule.get("allow_long", True)
        allow_short = getattr(rule, "allow_short", None)
        if allow_short is None and isinstance(rule, dict):
            allow_short = rule.get("allow_short", True)
        mult = getattr(rule, "score_multiplier", None)
        if mult is None and isinstance(rule, dict):
            mult = rule.get("score_multiplier", 1.0)
        mult = 1.0 if mult is None else float(mult)
        long_out = long_score * mult if allow_long else 0.0
        short_out = short_score * mult if allow_short else 0.0
        return long_out, short_out

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

        if self.composer == "weighted_avg":
            total_w = sum(self.weights[n] for n in outputs) or 1.0
            long_sum = sum(outputs[n].long * self.weights[n] for n in outputs) / total_w
            short_sum = sum(outputs[n].short * self.weights[n] for n in outputs) / total_w
        elif self.composer == "max_score":
            # Take the maximum long/short across all enabled strategies. Any
            # strategy with an above-threshold conviction can trigger. This is
            # the right composer when strategies operate in disjoint regimes
            # (e.g. trend_following in trending, mean_reversion in ranging).
            long_sum = max((outputs[n].long for n in outputs), default=0.0)
            short_sum = max((outputs[n].short for n in outputs), default=0.0)
        else:
            raise NotImplementedError(f"unhandled composer {self.composer}")

        # Structure-regime gate — e.g. block longs in "range" where the
        # historical grid shows 36% WR / net loss.
        long_sum, short_sum = self._apply_regime_rules(
            features_row, long_sum, short_sum,
        )

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
