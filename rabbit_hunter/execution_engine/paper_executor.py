"""PaperExecutor (v0.1.0) — same simulation as BacktestExecutor, semantically
"paper trade" for live shadow runs.

For a shadow run, we detect a K-line has closed at real time; the "next bar"
that would be the entry (in backtest terms) is now `the current bar`. The
same fee + slippage + funding model applies — we're not calling any exchange
API, only recording the intent and computing what a simulated fill would have
been.

Design decision: PaperExecutor inherits from BacktestExecutor because their
behavior is IDENTICAL. Keeping a separate class name makes call sites
self-documenting ("this is not real trading") and lets future variants of
paper trading (e.g. simulated exchange latency) be added without touching
backtest code.
"""
from __future__ import annotations

from .backtest_executor import BacktestExecutor


class PaperExecutor(BacktestExecutor):
    """Same fill model as BacktestExecutor; called from shadow (live) code path."""

    pass
