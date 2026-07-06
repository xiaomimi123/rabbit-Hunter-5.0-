from __future__ import annotations
from .base import BaseExecutor, Fill


class PaperExecutor(BaseExecutor):
    def submit(self, order, next_bar, atr: float) -> Fill:
        raise NotImplementedError("PaperExecutor is a Phase 1b+ deliverable")
