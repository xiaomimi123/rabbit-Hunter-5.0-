import pandas as pd
from rabbit_hunter.scoring_engine.rules_hard import pass_hard_rules, HardRulesParams


def _row(**overrides):
    base = {
        "quote_volume_24h": 1_000_000_000.0,
        "atr_pct": 0.02,
        "atr_pct_baseline": 0.02,
    }
    base.update(overrides)
    return base


def test_pass_normal():
    ok, reasons = pass_hard_rules(_row(), HardRulesParams(
        min_quote_volume_24h=1_000_000.0,
        atr_pct_max_multiplier=5.0,
        atr_pct_baseline_window=500,
    ))
    assert ok and reasons == []


def test_reject_low_liquidity():
    ok, reasons = pass_hard_rules(
        _row(quote_volume_24h=100.0),
        HardRulesParams(min_quote_volume_24h=1_000.0, atr_pct_max_multiplier=5.0, atr_pct_baseline_window=500),
    )
    assert not ok and "low_liquidity" in reasons


def test_reject_extreme_volatility():
    ok, reasons = pass_hard_rules(
        _row(atr_pct=1.0, atr_pct_baseline=0.02),
        HardRulesParams(min_quote_volume_24h=1_000.0, atr_pct_max_multiplier=5.0, atr_pct_baseline_window=500),
    )
    assert not ok and "extreme_volatility" in reasons
