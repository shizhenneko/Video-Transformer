from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


REQUIRED_SECTIONS = {
    "system",
    "proxy",
    "downloader",
    "validator",
    "image_generator",
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")

    missing = REQUIRED_SECTIONS.difference(data.keys())
    if missing:
        raise ValueError(f"Config missing sections: {', '.join(sorted(missing))}")

    return data