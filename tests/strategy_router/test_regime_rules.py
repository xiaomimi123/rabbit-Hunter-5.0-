"""Tests for structure-regime gates in the StrategyRouter.

The rules were derived from the structure × side performance grid:
range-longs were the only net-losing cell. The router must be able to
block a side per regime without touching any other regime's behavior.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rabbit_hunter.scoring_engine.base import BaseStrategy, ScoreOutput
from rabbit_hunter.strategy_router.router import StrategyRouter


class _Stub(BaseStrategy):
    name = "stub"
    version = "0"

    def __init__(self, long: float, short: float):
        self._long, self._short = long, short

    def score(self, features_row, features_history):
        return ScoreOutput(long=self._long, short=self._short,
                            components={}, metadata={})


def _router(long: float, short: float, rules: dict | None = None):
    return StrategyRouter(
        composer="max_score",
        strategy_weights={"stub": {"weight": 1.0}},
        strategies=[_Stub(long, short)],
        regime_rules=rules,
    )


def _route(r, regime: str | None):
    row = {"close": 100.0}
    if regime is not None:
        row["structure_regime"] = regime
    return r.route("BTC-USDT-SWAP", row, pd.DataFrame(),
                    open_action_threshold=0.5)


# ============================================================
# No rules → unchanged behavior
# ============================================================

def test_no_rules_long_passes():
    intent = _route(_router(0.8, 0.1), "range")
    assert intent.action == "open_long"


def test_missing_regime_key_passes():
    """A regime with no configured rule keeps default allow-all."""
    rules = {"range": {"allow_long": False}}
    intent = _route(_router(0.8, 0.1, rules), "uptrend")
    assert intent.action == "open_long"


def test_missing_structure_feature_passes():
    """A bar without structure_regime (e.g. warmup) is not gated."""
    rules = {"range": {"allow_long": False}}
    intent = _route(_router(0.8, 0.1, rules), None)
    assert intent.action == "open_long"


# ============================================================
# Blocking
# ============================================================

def test_range_long_blocked():
    rules = {"range": {"allow_long": False, "allow_short": True}}
    intent = _route(_router(0.8, 0.1, rules), "range")
    assert intent.action == "wait"    # long zeroed, short below threshold


def test_range_short_still_allowed():
    rules = {"range": {"allow_long": False, "allow_short": True}}
    intent = _route(_router(0.1, 0.8, rules), "range")
    assert intent.action == "open_short"


def test_block_both_sides():
    rules = {"range": {"allow_long": False, "allow_short": False}}
    intent = _route(_router(0.9, 0.9, rules), "range")
    assert intent.action == "wait"


def test_uptrend_unaffected_by_range_rule():
    rules = {"range": {"allow_long": False}}
    intent = _route(_router(0.8, 0.1, rules), "uptrend")
    assert intent.action == "open_long"


# ============================================================
# Score multiplier
# ============================================================

def test_score_multiplier_pushes_below_threshold():
    """0.6 long × 0.5 multiplier = 0.3 < 0.5 threshold → wait."""
    rules = {"range": {"score_multiplier": 0.5}}
    intent = _route(_router(0.6, 0.1, rules), "range")
    assert intent.action == "wait"


def test_score_multiplier_keeps_strong_signal():
    """1.0 long × 0.5 = 0.5 → still meets the 0.5 threshold."""
    rules = {"range": {"score_multiplier": 0.5}}
    intent = _route(_router(1.0, 0.1, rules), "range")
    assert intent.action == "open_long"
    assert intent.conviction == pytest.approx(0.5)


# ============================================================
# Pydantic RegimeRule objects work the same as dicts
# ============================================================

def test_pydantic_rule_objects():
    from rabbit_hunter.config.schema import RegimeRule
    rules = {"range": RegimeRule(allow_long=False, allow_short=True)}
    intent = _route(_router(0.8, 0.1, rules), "range")
    assert intent.action == "wait"
    intent = _route(_router(0.1, 0.8, rules), "range")
    assert intent.action == "open_short"
