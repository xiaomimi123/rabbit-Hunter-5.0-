"""Where each web route looks for data on disk.

Centralized so tests can override with tmp_path and production can
point at container-mounted volumes without changing route code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(".")

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def shadow(self) -> Path:
        return self.root / "shadows"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def baselines(self) -> Path:
        return self.root / "baselines"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def config_default(self) -> Path:
        return self.configs / "default.yaml"

    @property
    def config_history(self) -> Path:
        return self.configs / ".history"

    @property
    def ml_config(self) -> Path:
        return self.configs / "strategies" / "ml_scoring.yaml"
