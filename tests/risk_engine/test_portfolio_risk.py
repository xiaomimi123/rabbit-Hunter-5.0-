import math
import numpy as np
import pandas as pd
import pytest
from dataclasses import dataclass

from rabbit_hunter.config.schema import PortfolioRiskConfig
from rabbit_hunter.risk_engine.portfolio_risk import PortfolioRiskEngine
from rabbit_hunter.risk_engine.position_sizing import Order


def _default_cfg(**overrides):
    base = {
        "enabled": True,
        "correlation_window_bars": 720,
        "max_correlation_threshold": 0.7,
        "correlated_size_reduction": 0.5,
        "max_gross_leverage": 3.0,
    }
    base.update(overrides)
    return PortfolioRiskConfig(**base)


def _feats(prices):
    return pd.DataFrame({"timestamp": range(len(prices)), "close": prices})


def _mk_order(symbol="ETH-USDT-SWAP", side="long", entry=3000.0, size=0.01):
    return Order(
        symbol=symbol, side=side, entry_price=entry,
        stop_price=entry - 100.0 if side == "long" else entry + 100.0,
        take_profit_price=entry + 200.0 if side == "long" else entry - 200.0,
        size=size,
        leverage=(size * entry) / 10_000.0,
    )


@dataclass
class _Pos:
    symbol: str
    side: str
    entry_price: float
    size: float


def test_disabled_config_passes_through():
    engine = PortfolioRiskEngine(_default_cfg(enabled=False), {})
    order = _mk_order()
    result = engine.evaluate(candidate=order, open_positions={}, equity=10_000.0)
    assert result.accepted is True
    assert result.adjusted_order is order
    assert result.size_multiplier == 1.0
    assert result.reasons == []


def test_correlation_between_identical_series_is_one():
    prices = np.linspace(100, 200, 1000)
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {"A": _feats(prices), "B": _feats(prices)},
    )
    assert math.isclose(engine.correlation("A", "B"), 1.0, abs_tol=1e-9)


def test_correlation_between_uncorrelated_series_near_zero():
    rng = np.random.default_rng(42)
    prices_a = 100 + np.cumsum(rng.normal(0, 1, 1000))
    prices_b = 100 + np.cumsum(rng.normal(0, 1, 1000))
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {"A": _feats(prices_a), "B": _feats(prices_b)},
    )
    assert abs(engine.correlation("A", "B")) < 0.15


def test_high_correlation_triggers_size_reduction():
    prices = np.linspace(100, 200, 1000)
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {"BTC-USDT-SWAP": _feats(prices), "ETH-USDT-SWAP": _feats(prices)},
    )
    open_positions = {
        "BTC-USDT-SWAP": _Pos(
            symbol="BTC-USDT-SWAP", side="long",
            entry_price=60_000.0, size=0.01,  # notional $600
        ),
    }
    candidate = _mk_order(symbol="ETH-USDT-SWAP", entry=3_000.0, size=0.05)  # notional $150
    result = engine.evaluate(candidate, open_positions, equity=10_000.0)
    # Correlation 1.0 > 0.7 → multiplier 0.5
    assert result.accepted is True
    assert result.size_multiplier == pytest.approx(0.5)
    assert result.adjusted_order.size == pytest.approx(0.05 * 0.5)
    assert any("correlation:BTC-USDT-SWAP" in r for r in result.reasons)


def test_gross_leverage_cap_shrinks_order():
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {},  # no features → correlations all 0 → correlation gate inactive
    )
    # Existing notional 2.5× equity leaves 0.5× headroom for new order.
    open_positions = {
        "BTC-USDT-SWAP": _Pos(
            symbol="BTC-USDT-SWAP", side="long",
            entry_price=60_000.0, size=0.417,  # notional ≈ 25_000 = 2.5×
        ),
    }
    candidate = _mk_order(symbol="SOL-USDT-SWAP", entry=200.0, size=100.0)  # notional 20_000 = 2.0×
    # Total would be 4.5× — needs shrinking. Existing = 60000*0.417 = 25020.
    # Available: 30000 - 25020 = 4980 → new size = 4980/200 = 24.9.
    result = engine.evaluate(candidate, open_positions, equity=10_000.0)
    assert result.accepted is True
    assert result.adjusted_order.size == pytest.approx(24.9, rel=1e-3)
    assert any("gross_leverage_shrunk" in r for r in result.reasons)


def test_gross_leverage_full_rejects():
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {},  # no features → correlations all 0 → correlation gate inactive
    )
    # Existing at exactly 3× cap → no room
    open_positions = {
        "BTC-USDT-SWAP": _Pos(
            symbol="BTC-USDT-SWAP", side="long",
            entry_price=60_000.0, size=0.5,  # notional 30_000 = 3× on $10k
        ),
    }
    candidate = _mk_order(symbol="SOL-USDT-SWAP", entry=200.0, size=10.0)
    result = engine.evaluate(candidate, open_positions, equity=10_000.0)
    assert result.accepted is False
    assert result.adjusted_order is None
    assert any("gross_leverage_full" in r for r in result.reasons)


def test_no_correlated_position_passes_through():
    rng = np.random.default_rng(7)
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {
            "A": _feats(100 + np.cumsum(rng.normal(0, 1, 1000))),
            "B": _feats(100 + np.cumsum(rng.normal(0, 1, 1000))),
        },
    )
    open_positions = {
        "A": _Pos(symbol="A", side="long", entry_price=100.0, size=1.0),
    }
    candidate = _mk_order(symbol="B", entry=100.0, size=1.0)
    result = engine.evaluate(candidate, open_positions, equity=10_000.0)
    # Low correlation → no reduction. Small notional → no leverage cap.
    assert result.accepted is True
    assert result.size_multiplier == 1.0
    assert result.adjusted_order is candidate


def test_correlation_reduction_is_non_compounding_at_multi_symbol():
    """With 5 correlated open positions, size_mult must be exactly
    correlated_size_reduction (0.5), NOT 0.5**5 = 0.03125. Non-compounding
    is the design fix for 10-symbol backtests where multiplicative stacking
    collapses all orders to ~1/16 size."""
    prices = np.linspace(100, 200, 1000)
    engine = PortfolioRiskEngine(
        _default_cfg(),
        {sym: _feats(prices) for sym in ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"]},
    )
    open_positions = {
        sym: _Pos(symbol=sym, side="long", entry_price=100.0, size=1.0)
        for sym in ["BTC", "ETH", "SOL", "BNB", "XRP"]  # 5 correlated positions
    }
    candidate = _mk_order(symbol="DOGE", entry=100.0, size=1.0)
    result = engine.evaluate(candidate, open_positions, equity=10_000.0)
    assert result.accepted is True
    # The critical assertion — mult is 0.5, NOT 0.5**5 = 0.03125
    assert result.size_multiplier == pytest.approx(0.5)
