import math
from rabbit_hunter.config.schema import CircuitBreakerConfig
from rabbit_hunter.risk_engine.circuit_breaker import CircuitBreaker


def _cfg(enabled=True, multiplier=3.0):
    return CircuitBreakerConfig(
        enabled=enabled,
        atr_shock_multiplier=multiplier,
        emergency_close_on_shock=True,
    )


def test_disabled_never_trips():
    cb = CircuitBreaker(_cfg(enabled=False))
    result = cb.check({"atr_14": 1000.0, "atr_pct": 100.0, "atr_pct_baseline": 0.01})
    assert result.tripped is False


def test_normal_atr_does_not_trip():
    cb = CircuitBreaker(_cfg())
    result = cb.check({"atr_14": 200.0, "atr_pct": 0.01, "atr_pct_baseline": 0.01})
    assert result.tripped is False
    assert result.atr_ratio == 1.0


def test_shock_at_exact_multiplier_trips():
    cb = CircuitBreaker(_cfg(multiplier=3.0))
    # atr_pct is exactly 3× baseline
    result = cb.check({"atr_pct": 0.06, "atr_pct_baseline": 0.02})
    assert result.tripped is True
    assert result.atr_ratio == 3.0
    assert "atr_ratio=3.00>=3.0" in result.reason


def test_shock_above_multiplier_trips():
    cb = CircuitBreaker(_cfg(multiplier=3.0))
    result = cb.check({"atr_pct": 0.1, "atr_pct_baseline": 0.02})  # 5×
    assert result.tripped is True
    assert result.atr_ratio == 5.0


def test_missing_baseline_does_not_trip():
    """During warmup atr_pct_baseline is NaN — should not falsely trip."""
    cb = CircuitBreaker(_cfg())
    result = cb.check({"atr_pct": 0.05})  # no baseline
    assert result.tripped is False


def test_atr_pct_fallback_from_atr_and_close():
    """If atr_pct isn't present, compute from atr_14/close."""
    cb = CircuitBreaker(_cfg(multiplier=3.0))
    # atr=6000, close=100000 → atr_pct=0.06. baseline=0.02 → ratio=3.0 → trip.
    result = cb.check({"atr_14": 6000.0, "close": 100_000.0, "atr_pct_baseline": 0.02})
    assert result.tripped is True
    assert math.isclose(result.atr_ratio, 3.0, abs_tol=1e-9)


def test_zero_or_negative_baseline_does_not_crash():
    cb = CircuitBreaker(_cfg())
    result = cb.check({"atr_pct": 0.1, "atr_pct_baseline": 0.0})
    assert result.tripped is False
