"""Phase 3 · § 3.3 — Portfolio-level risk gates.

Two functions:
  1. Inter-symbol correlation risk:   scale down (or reject) an incoming
     order if a highly-correlated symbol is already open.
  2. Gross-leverage cap:              hard-reject any order that would
     push |sum(notional)| / equity past the configured cap.

Both gates run AFTER the single-trade `RiskEngine.size()` sizing, so this
module receives an already-sized candidate `Order` and returns either an
adjusted order or None (rejected).

Correlation is computed once at engine __init__ over the full historical
window per (symbol, symbol) pair using log returns of close price. For
Phase 3 backtest this is a static matrix. A future Phase 3+ could roll
the window forward per bar (cost: O(N × pairs × window) per bar).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rabbit_hunter.config.schema import PortfolioRiskConfig
from rabbit_hunter.risk_engine.position_sizing import Order


@dataclass(frozen=True)
class PortfolioRiskResult:
    """Outcome of running the portfolio gates against a candidate order."""
    accepted: bool
    adjusted_order: Order | None
    size_multiplier: float           # 1.0 = no change; 0.0 = rejected
    reasons: list[str]                # e.g. ["correlation:BTC-USDT-SWAP=0.87",
                                      #        "gross_leverage=3.2>3.0"]


class PortfolioRiskEngine:
    """Multi-symbol risk gate. Constructed once with the full features
    dataframe per symbol; consulted every candidate order."""

    def __init__(
        self,
        cfg: PortfolioRiskConfig,
        features_by_symbol: dict[str, pd.DataFrame],
    ):
        self.cfg = cfg
        # Correlation matrix computed once over the full window. Symbols
        # not in the map get correlation 0 with everything.
        self._corr = self._compute_correlation_matrix(features_by_symbol)

    @staticmethod
    def _compute_correlation_matrix(
        features_by_symbol: dict[str, pd.DataFrame],
    ) -> dict[tuple[str, str], float]:
        """Pearson correlation of log returns across the full feature window.
        Simple, static, and directional-side-agnostic — we're asking "do
        these two symbols move together", not "is the current trade a
        contrarian bet"."""
        returns: dict[str, np.ndarray] = {}
        for sym, df in features_by_symbol.items():
            if len(df) < 2 or "close" not in df.columns:
                continue
            close = df["close"].to_numpy(dtype=float)
            logret = np.diff(np.log(np.clip(close, 1e-12, None)))
            returns[sym] = logret

        out: dict[tuple[str, str], float] = {}
        symbols = list(returns)
        for i, a in enumerate(symbols):
            for b in symbols[i:]:
                if a == b:
                    out[(a, b)] = 1.0
                    continue
                arr_a, arr_b = returns[a], returns[b]
                n = min(len(arr_a), len(arr_b))
                if n < 30:  # not enough overlap to trust
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(arr_a[-n:], arr_b[-n:])[0, 1])
                    if not np.isfinite(corr):
                        corr = 0.0
                out[(a, b)] = corr
                out[(b, a)] = corr
        return out

    def correlation(self, sym_a: str, sym_b: str) -> float:
        return self._corr.get((sym_a, sym_b), 0.0)

    def evaluate(
        self,
        candidate: Order,
        open_positions: dict,     # symbol -> Position (from Ledger)
        equity: float,
    ) -> PortfolioRiskResult:
        """Apply both gates. Returns (accepted, adjusted_order_or_None, mult, reasons)."""
        if not self.cfg.enabled:
            return PortfolioRiskResult(True, candidate, 1.0, [])

        reasons: list[str] = []
        size_mult = 1.0

        # --- Gate 1: correlation with any open position ---
        for sym, _pos in open_positions.items():
            if sym == candidate.symbol:
                continue
            rho = abs(self.correlation(candidate.symbol, sym))
            if rho > self.cfg.max_correlation_threshold:
                size_mult *= self.cfg.correlated_size_reduction
                reasons.append(f"correlation:{sym}={rho:.2f}")

        # If reduction pushed size to 0, treat as reject
        if size_mult <= 1e-9:
            return PortfolioRiskResult(
                accepted=False, adjusted_order=None,
                size_multiplier=0.0,
                reasons=reasons + ["size_zeroed_by_correlation"],
            )

        # --- Gate 2: gross leverage cap ---
        # Sum existing notional + candidate's adjusted notional.
        existing_notional = 0.0
        for pos in open_positions.values():
            # entry_price × size = notional at entry (mark-to-market drift ignored)
            existing_notional += pos.entry_price * pos.size

        candidate_size = candidate.size * size_mult
        candidate_notional = candidate.entry_price * candidate_size
        total_notional = existing_notional + candidate_notional
        gross_lev = total_notional / equity if equity > 0 else float("inf")

        if gross_lev > self.cfg.max_gross_leverage:
            # Try shrinking the candidate to fit under the cap.
            available_notional = self.cfg.max_gross_leverage * equity - existing_notional
            if available_notional <= 0:
                return PortfolioRiskResult(
                    accepted=False, adjusted_order=None,
                    size_multiplier=0.0,
                    reasons=reasons + [f"gross_leverage_full={gross_lev:.2f}>{self.cfg.max_gross_leverage:.1f}"],
                )
            reduction = available_notional / candidate_notional
            size_mult *= reduction
            candidate_size = candidate.size * size_mult
            reasons.append(f"gross_leverage_shrunk={gross_lev:.2f}>{self.cfg.max_gross_leverage:.1f}")

        # Nothing to change → fast path
        if size_mult >= 1.0 - 1e-9:
            return PortfolioRiskResult(True, candidate, 1.0, reasons)

        # Build a new Order with the reduced size + recomputed leverage
        new_size = candidate.size * size_mult
        new_notional = new_size * candidate.entry_price
        new_leverage = new_notional / equity if equity > 0 else 0.0
        adjusted = Order(
            symbol=candidate.symbol,
            side=candidate.side,
            entry_price=candidate.entry_price,
            stop_price=candidate.stop_price,
            take_profit_price=candidate.take_profit_price,
            size=new_size,
            leverage=new_leverage,
        )
        return PortfolioRiskResult(True, adjusted, size_mult, reasons)
