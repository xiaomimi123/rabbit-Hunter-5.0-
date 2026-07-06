"""Position reconciliation — compare ledger vs exchange, one snapshot per call.

Drift between the internal Ledger and the exchange state is the single most
dangerous class of production bug. It can appear when:
  - A live order was placed but the ledger.record_entry() never happened
    (crash between the exchange call and the write).
  - A stop-loss fired on the exchange side but the ledger was closed by
    a competing signal (or vice versa).
  - Manual UI intervention (an operator closed a position from the phone).

This module produces a structured `ReconcileReport` that lists every
mismatch so an operator (or a watchdog) can act on it. It does NOT
auto-correct — silent auto-correction is exactly the pattern that produces
the "the executor thinks it's flat but there's a hidden 5x leveraged
position" catastrophe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PositionDiscrepancy:
    symbol: str
    kind: str            # "missing_on_exchange" | "missing_on_ledger"
                         # | "side_mismatch" | "size_mismatch"
    ledger: dict | None
    exchange: dict | None
    detail: str


@dataclass
class ReconcileReport:
    ok: bool
    ledger_position_count: int
    exchange_position_count: int
    discrepancies: list[PositionDiscrepancy] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        """Human-readable summary lines for logging."""
        head = (f"reconcile: ledger={self.ledger_position_count} "
                f"exchange={self.exchange_position_count} "
                f"status={'OK' if self.ok else 'MISMATCH'}")
        if self.ok:
            return [head]
        lines = [head, "discrepancies:"]
        for d in self.discrepancies:
            lines.append(f"  - {d.symbol} [{d.kind}] {d.detail}")
        return lines


def reconcile_positions(
    ledger_positions: dict[str, Any],
    exchange_positions: dict[str, dict],
    size_tolerance_pct: float = 0.001,
) -> ReconcileReport:
    """Compare a dict of ledger `Position` objects with the flat exchange
    position dicts returned by LiveExecutor.fetch_exchange_positions().

    A discrepancy is reported for every symbol where:
      - Ledger has a position but exchange doesn't (or vice versa)
      - Both have positions but sides differ
      - Both have positions on the same side but sizes differ by more
        than `size_tolerance_pct` (default 0.1%, to absorb ccxt rounding)
    """
    discrepancies: list[PositionDiscrepancy] = []

    ledger_syms = set(ledger_positions.keys())
    exchange_syms = set(exchange_positions.keys())

    for sym in sorted(ledger_syms - exchange_syms):
        pos = ledger_positions[sym]
        discrepancies.append(PositionDiscrepancy(
            symbol=sym, kind="missing_on_exchange",
            ledger={"side": getattr(pos, "side", None),
                    "size": float(getattr(pos, "size", 0.0))},
            exchange=None,
            detail=f"ledger says {getattr(pos, 'side', '?')} "
                   f"{float(getattr(pos, 'size', 0.0)):.6f} but exchange is flat",
        ))

    for sym in sorted(exchange_syms - ledger_syms):
        ex = exchange_positions[sym]
        discrepancies.append(PositionDiscrepancy(
            symbol=sym, kind="missing_on_ledger",
            ledger=None, exchange=ex,
            detail=f"exchange has {ex['side']} {ex['size']:.6f} "
                   f"but ledger is flat",
        ))

    for sym in sorted(ledger_syms & exchange_syms):
        pos = ledger_positions[sym]
        ex = exchange_positions[sym]
        ledger_side = getattr(pos, "side", None)
        ledger_size = float(getattr(pos, "size", 0.0))
        if ledger_side != ex["side"]:
            discrepancies.append(PositionDiscrepancy(
                symbol=sym, kind="side_mismatch",
                ledger={"side": ledger_side, "size": ledger_size},
                exchange=ex,
                detail=f"ledger={ledger_side} exchange={ex['side']}",
            ))
            continue
        # Sizes match within tolerance?
        # Zero-guard: if ledger_size == 0 (shouldn't happen but be safe),
        # treat any nonzero exchange size as a mismatch.
        if ledger_size == 0:
            if ex["size"] != 0:
                discrepancies.append(PositionDiscrepancy(
                    symbol=sym, kind="size_mismatch",
                    ledger={"side": ledger_side, "size": 0.0},
                    exchange=ex,
                    detail=f"ledger=0 but exchange={ex['size']:.6f}",
                ))
            continue
        rel_diff = abs(ex["size"] - ledger_size) / ledger_size
        if rel_diff > size_tolerance_pct:
            discrepancies.append(PositionDiscrepancy(
                symbol=sym, kind="size_mismatch",
                ledger={"side": ledger_side, "size": ledger_size},
                exchange=ex,
                detail=(f"ledger={ledger_size:.6f} exchange={ex['size']:.6f} "
                        f"rel_diff={rel_diff*100:.3f}%"),
            ))

    return ReconcileReport(
        ok=len(discrepancies) == 0,
        ledger_position_count=len(ledger_syms),
        exchange_position_count=len(exchange_syms),
        discrepancies=discrepancies,
    )
