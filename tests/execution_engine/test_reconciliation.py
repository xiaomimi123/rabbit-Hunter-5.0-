"""Tests for reconcile_positions — every discrepancy kind, plus the OK path.

Reconciliation is the safety net that catches ledger↔exchange drift. If
any branch here silently reports OK when it shouldn't, live trading is
exposed. Every path is pinned.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from rabbit_hunter.execution_engine.reconciliation import (
    reconcile_positions, ReconcileReport, PositionDiscrepancy,
)


@dataclass
class _StubPosition:
    """Just the two fields the reconciler reads."""
    side: str
    size: float


# ============================================================
# OK path
# ============================================================

def test_ok_when_both_empty():
    r = reconcile_positions({}, {})
    assert r.ok is True
    assert r.discrepancies == []
    assert r.ledger_position_count == 0
    assert r.exchange_position_count == 0


def test_ok_when_ledger_and_exchange_match():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.01)},
        exchange_positions={"BTC-USDT-SWAP":
                            {"side": "long", "size": 0.01, "entry_price": 50_000.0}},
    )
    assert r.ok is True
    assert r.ledger_position_count == 1
    assert r.exchange_position_count == 1


def test_ok_when_sizes_within_tolerance():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.01)},
        exchange_positions={"BTC-USDT-SWAP":
                            {"side": "long", "size": 0.01001, "entry_price": 50_000.0}},
        size_tolerance_pct=0.01,  # 1%
    )
    assert r.ok is True


# ============================================================
# missing_on_exchange — ledger thinks we have a position, exchange doesn't
# ============================================================

def test_missing_on_exchange_flagged():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.01)},
        exchange_positions={},
    )
    assert r.ok is False
    assert len(r.discrepancies) == 1
    d = r.discrepancies[0]
    assert d.kind == "missing_on_exchange"
    assert d.symbol == "BTC-USDT-SWAP"
    assert "long" in d.detail
    assert d.exchange is None


# ============================================================
# missing_on_ledger — the risky direction (hidden exposure)
# ============================================================

def test_missing_on_ledger_flagged():
    r = reconcile_positions(
        ledger_positions={},
        exchange_positions={
            "BTC-USDT-SWAP":
            {"side": "short", "size": 0.02, "entry_price": 50_000.0},
        },
    )
    assert r.ok is False
    d = r.discrepancies[0]
    assert d.kind == "missing_on_ledger"
    assert d.symbol == "BTC-USDT-SWAP"
    assert "short" in d.detail
    assert d.ledger is None


# ============================================================
# side_mismatch — ledger thinks long, exchange has short (or vice versa)
# ============================================================

def test_side_mismatch_flagged():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.01)},
        exchange_positions={"BTC-USDT-SWAP":
                            {"side": "short", "size": 0.01, "entry_price": 50_000.0}},
    )
    assert r.ok is False
    d = r.discrepancies[0]
    assert d.kind == "side_mismatch"
    # Sizes matched but side didn't — no size_mismatch should follow
    assert len(r.discrepancies) == 1


# ============================================================
# size_mismatch — same side, different size beyond tolerance
# ============================================================

def test_size_mismatch_flagged_when_beyond_tolerance():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.01)},
        exchange_positions={"BTC-USDT-SWAP":
                            {"side": "long", "size": 0.015, "entry_price": 50_000.0}},
        size_tolerance_pct=0.01,  # 1%
    )
    assert r.ok is False
    d = r.discrepancies[0]
    assert d.kind == "size_mismatch"
    assert "0.015" in d.detail


def test_size_zero_on_ledger_flagged_if_exchange_nonzero():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.0)},
        exchange_positions={"BTC-USDT-SWAP":
                            {"side": "long", "size": 0.01, "entry_price": 50_000.0}},
    )
    assert r.ok is False
    assert r.discrepancies[0].kind == "size_mismatch"


# ============================================================
# Multiple symbols — every mismatch reported, not just the first
# ============================================================

def test_multiple_discrepancies_reported():
    r = reconcile_positions(
        ledger_positions={
            "BTC-USDT-SWAP": _StubPosition("long", 0.01),
            "ETH-USDT-SWAP": _StubPosition("short", 1.0),
            "SOL-USDT-SWAP": _StubPosition("long", 5.0),
        },
        exchange_positions={
            "BTC-USDT-SWAP":
                {"side": "long", "size": 0.01, "entry_price": 50_000.0},  # OK
            # ETH missing on exchange
            "SOL-USDT-SWAP":
                {"side": "short", "size": 5.0, "entry_price": 100.0},  # side mismatch
            "DOGE-USDT-SWAP":
                {"side": "long", "size": 10.0, "entry_price": 0.1},   # missing on ledger
        },
    )
    assert r.ok is False
    assert len(r.discrepancies) == 3
    kinds = {(d.symbol, d.kind) for d in r.discrepancies}
    assert ("ETH-USDT-SWAP", "missing_on_exchange") in kinds
    assert ("SOL-USDT-SWAP", "side_mismatch") in kinds
    assert ("DOGE-USDT-SWAP", "missing_on_ledger") in kinds


# ============================================================
# as_lines() — used by the reconcile CLI for logging
# ============================================================

def test_as_lines_ok_returns_one_line():
    r = reconcile_positions({}, {})
    lines = r.as_lines()
    assert len(lines) == 1
    assert "OK" in lines[0]


def test_as_lines_mismatch_lists_each_discrepancy():
    r = reconcile_positions(
        ledger_positions={"BTC-USDT-SWAP": _StubPosition("long", 0.01)},
        exchange_positions={},
    )
    lines = r.as_lines()
    assert "MISMATCH" in lines[0]
    assert any("BTC-USDT-SWAP" in ln for ln in lines[1:])
