from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    bin_size_sec: float | None = None
    smoothing_window_bins: int = 3
    chat_offset_seconds: float = 0.0
    pre_roll_sec: float = 15.0
    post_roll_sec: float = 25.0
    peak_quantile: float = 0.9


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_preset(path: Path) -> dict[str, float]:
    payload = load_yaml(path)
    return payload.get("weights", {})
