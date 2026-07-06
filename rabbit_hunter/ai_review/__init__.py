"""Phase 4 · § 3.5 — AI Review Agent utilities.

Loads report artifacts and constructs a well-crafted prompt for external
LLMs (Claude, GPT, etc.) to review the backtest and suggest parameter
tuning. Strictly follows the architecture spec's rule: AI Review NEVER
directly modifies the strategy — it only produces suggestions that a
human must vet, backtest, shadow-mode validate, and approve.
"""
from .agent import build_review_prompt, load_report_bundle

__all__ = ["build_review_prompt", "load_report_bundle"]
