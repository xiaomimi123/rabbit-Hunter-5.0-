from dataclasses import dataclass
import pandas as pd
from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput
from rabbit_hunter.strategy_router.router import StrategyRouter, Intent


class StubLong(BaseStrategy):
    name = "stub_long"; version = "0.1"
    def score(self, fr, fh): return ScoreOutput(long=0.8, short=0.1)

class StubShort(BaseStrategy):
    name = "stub_short"; version = "0.1"
    def score(self, fr, fh): return ScoreOutput(long=0.1, short=0.7)

class StubWait(BaseStrategy):
    name = "stub_wait"; version = "0.1"
    def score(self, fr, fh): return ScoreOutput(long=0.2, short=0.2)


def _weights(mapping):
    return {name: {"weight": w} for name, w in mapping.items()}


def test_weighted_avg_long_wins():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_long": 1.0, "stub_wait": 1.0}),
        strategies=[StubLong(), StubWait()],
    )
    intent = r.route("BTC-USDT-SWAP", {"close": 100}, pd.DataFrame(), open_action_threshold=0.4)
    assert intent.action == "open_long"
    assert intent.symbol == "BTC-USDT-SWAP"
    assert 0.4 <= intent.conviction <= 1.0
    assert intent.contributing_strategies == {"stub_long": 0.8, "stub_wait": 0.2}


def test_weighted_avg_short_wins():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_short": 1.0}),
        strategies=[StubShort()],
    )
    intent = r.route("ETH-USDT-SWAP", {"close": 100}, pd.DataFrame(), open_action_threshold=0.5)
    assert intent.action == "open_short"


def test_below_threshold_returns_wait():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_wait": 1.0}),
        strategies=[StubWait()],
    )
    intent = r.route("BTC-USDT-SWAP", {"close": 100}, pd.DataFrame(), open_action_threshold=0.5)
    assert intent.action == "wait"


def test_features_snapshot_passed_through():
    r = StrategyRouter(
        composer="weighted_avg",
        strategy_weights=_weights({"stub_long": 1.0}),
        strategies=[StubLong()],
    )
    intent = r.route("BTC-USDT-SWAP", {"close": 100, "ema20": 105}, pd.DataFrame(), open_action_threshold=0.5)
    assert intent.features_snapshot["ema20"] == 105
