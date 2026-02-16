from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


REQUIRED_SECTIONS = {
    "system",
    "proxy",
    "downloader",
    "validator",
    "image_generator",
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    _load_dotenv(DEFAULT_DOTENV_PATH)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a mapping")

    missing = REQUIRED_SECTIONS.difference(data.keys())
    if missing:
        raise ValueError(f"Config missing sections: {', '.join(sorted(missing))}")

    api_keys = data.get("api_keys")
    if not isinstance(api_keys, dict):
        api_keys = {}
        data["api_keys"] = api_keys

    env_map = {
        "VT_GEMINI_API_KEY": "gemini",
        "VT_KIMI_API_KEY": "kimi",
        "VT_NANO_BANANA_API_KEY": "nano_banana",
    }
    for env_name, key_name in env_map.items():
        value = os.environ.get(env_name)
        if value:
            api_keys[key_name] = value

    return data
