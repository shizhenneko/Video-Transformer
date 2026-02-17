from pathlib import Path
import uuid

import pytest
import yaml

from utils.config import load_config


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    tmp_root = root / "data" / "output" / "logs" / "pytest_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = tmp_root / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def test_load_config_success():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {"max_api_calls": 10},
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = temp_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    loaded = load_config(config_path)
    assert loaded["system"]["max_api_calls"] == 10


def test_load_config_missing_sections():
    temp_dir = _make_temp_dir()
    config_data = {"system": {"max_api_calls": 10}}
    config_path = temp_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_config(config_path)