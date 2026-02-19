from __future__ import annotations

import os
from pathlib import Path
from typing import cast

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


def _ensure_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _coerce_str(value: object, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _apply_system_defaults(system: dict[str, object]) -> None:
    note_profile = _coerce_str(system.get("note_profile"), "default").lower()
    if note_profile not in {"default", "pdf"}:
        note_profile = "default"
    system["note_profile"] = note_profile

    quality_gates = _ensure_dict(system.get("quality_gates"))
    if "enabled" not in quality_gates:
        quality_gates["enabled"] = False
    if "max_extra_llm_calls" not in quality_gates:
        quality_gates["max_extra_llm_calls"] = 1
    system["quality_gates"] = quality_gates

    pdf_math = _ensure_dict(system.get("pdf_math"))
    if "enable_display_math" not in pdf_math:
        pdf_math["enable_display_math"] = note_profile == "pdf"
    system["pdf_math"] = pdf_math

    pdf_diagrams = _ensure_dict(system.get("pdf_diagrams"))
    if "enable_tikz" not in pdf_diagrams:
        pdf_diagrams["enable_tikz"] = False
    system["pdf_diagrams"] = pdf_diagrams

    render = _ensure_dict(system.get("render"))
    if "include_concept_index" not in render:
        render["include_concept_index"] = note_profile != "pdf"
    system["render"] = render

    pdf_typesetting = _ensure_dict(system.get("pdf_typesetting"))
    if "engine" not in pdf_typesetting:
        pdf_typesetting["engine"] = "xelatex"
    if "mainfont" not in pdf_typesetting:
        pdf_typesetting["mainfont"] = "TeX Gyre Termes"
    if "monofont" not in pdf_typesetting:
        pdf_typesetting["monofont"] = "DejaVu Sans Mono"
    if "header_tex_path" not in pdf_typesetting:
        pdf_typesetting["header_tex_path"] = None
    system["pdf_typesetting"] = pdf_typesetting


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    _load_dotenv(DEFAULT_DOTENV_PATH)
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw_data = cast(object, yaml.safe_load(config_path.read_text(encoding="utf-8")))
    if not isinstance(raw_data, dict):
        raise ValueError("Config must be a mapping")
    data = cast(dict[str, object], raw_data)

    missing = REQUIRED_SECTIONS.difference(data.keys())
    if missing:
        raise ValueError(f"Config missing sections: {', '.join(sorted(missing))}")

    system = data.get("system")
    if not isinstance(system, dict):
        raise ValueError("Config system section must be a mapping")
    system = cast(dict[str, object], system)
    _apply_system_defaults(system)

    api_keys = data.get("api_keys")
    if not isinstance(api_keys, dict):
        api_keys = {}
        data["api_keys"] = api_keys
    api_keys = cast(dict[str, object], api_keys)

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
