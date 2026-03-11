from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BIN_SIZE = 5.0
DEFAULT_SMOOTHING = 3
DEFAULT_PRE_ROLL = 15.0
DEFAULT_POST_ROLL = 25.0


@dataclass
class AppConfig:
    bin_size_sec: float = DEFAULT_BIN_SIZE
    smoothing_window_bins: int = DEFAULT_SMOOTHING
    chat_offset_seconds: float = 0.0
    pre_roll_sec: float = DEFAULT_PRE_ROLL
    post_roll_sec: float = DEFAULT_POST_ROLL
    peak_quantile: float = 0.9


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_preset(path: Path) -> dict[str, float]:
    data = load_yaml(path)
    return data.get("weights", {})
