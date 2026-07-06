from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: Literal["long", "short"]
    fill_price: float
    size: float
    timestamp: int
    fees: float
    slippage: float
    reason: str = "entry"


class BaseExecutor(ABC):
    @abstractmethod
    def submit(self, order, next_bar: dict, atr: float) -> Fill: ...
