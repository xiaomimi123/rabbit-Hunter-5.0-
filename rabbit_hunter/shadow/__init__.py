"""Phase 4 · Shadow Mode — live paper trading using the exact same
Feature Engine / Scoring / Router / Risk code as backtest.

Not a real trading module. It simulates fills without touching any
exchange API, records the intent, and writes snapshots we can later
diff against real market activity to validate the model:
  - Did the simulated fill price match what the market actually printed?
  - Did our slippage assumption (0.1 × ATR) match reality?
  - Did our funding settlement math match Binance's actual charge?

Architecture spec § 4.1: shadow mode is what unlocks small-money real
trading. Never send real orders until shadow has run cleanly for weeks.
"""
from .runner import ShadowRunner, ShadowConfig

__all__ = ["ShadowRunner", "ShadowConfig"]
