from pathlib import Path
import yaml
from .schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
