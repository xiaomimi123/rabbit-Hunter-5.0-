from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class ScoreOutput:
    long: float
    short: float
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStrategy(ABC):
    name: str = ""
    version: str = "0.0.0"

    @abstractmethod
    def score(self, features_row: dict, features_history: pd.DataFrame) -> ScoreOutput: ...
