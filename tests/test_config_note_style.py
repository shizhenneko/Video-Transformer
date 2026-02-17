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


def _create_config_file(temp_dir: Path, config_data: dict) -> Path:
    config_path = temp_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path


def test_default_note_style_values():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {
            "max_api_calls": 10,
            "self_check_mode": "static",
            "note_style": "core_appendix",
            "question_scope": "chapter_only",
            "answer_placement": "inline_after_questions",
            "code_placement": "appendix_only",
        },
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = _create_config_file(temp_dir, config_data)

    loaded = load_config(config_path)

    assert loaded["system"]["note_style"] == "core_appendix"
    assert loaded["system"]["question_scope"] == "chapter_only"
    assert loaded["system"]["answer_placement"] == "inline_after_questions"
    assert loaded["system"]["code_placement"] == "appendix_only"


def test_legacy_mode_config():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {
            "max_api_calls": 10,
            "self_check_mode": "static",
            "note_style": "full_detail",
            "question_scope": "per_section",
            "answer_placement": "end_of_chapter",
            "code_placement": "inline",
        },
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = _create_config_file(temp_dir, config_data)

    loaded = load_config(config_path)

    assert loaded["system"]["note_style"] == "full_detail"
    assert loaded["system"]["question_scope"] == "per_section"
    assert loaded["system"]["answer_placement"] == "end_of_chapter"
    assert loaded["system"]["code_placement"] == "inline"


def test_self_check_mode_preserved():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {
            "max_api_calls": 10,
            "self_check_mode": "static",
            "note_style": "core_appendix",
            "question_scope": "chapter_only",
            "answer_placement": "inline_after_questions",
            "code_placement": "appendix_only",
        },
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = _create_config_file(temp_dir, config_data)

    loaded = load_config(config_path)

    assert loaded["system"]["self_check_mode"] == "static"


def test_config_loading_with_missing_note_style_flags():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {
            "max_api_calls": 10,
            "self_check_mode": "static",
        },
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = _create_config_file(temp_dir, config_data)

    loaded = load_config(config_path)

    assert "self_check_mode" in loaded["system"]
    assert loaded["system"]["self_check_mode"] == "static"


def test_mixed_mode_config():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {
            "max_api_calls": 10,
            "self_check_mode": "static",
            "note_style": "core_appendix",
            "question_scope": "per_section",
            "answer_placement": "inline_after_questions",
            "code_placement": "inline",
        },
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = _create_config_file(temp_dir, config_data)

    loaded = load_config(config_path)

    assert loaded["system"]["note_style"] == "core_appendix"
    assert loaded["system"]["question_scope"] == "per_section"
    assert loaded["system"]["answer_placement"] == "inline_after_questions"
    assert loaded["system"]["code_placement"] == "inline"


def test_all_flags_independent():
    temp_dir = _make_temp_dir()
    config_data = {
        "system": {
            "max_api_calls": 10,
            "self_check_mode": "default",
            "note_style": "full_detail",
            "question_scope": "chapter_only",
            "answer_placement": "end_of_chapter",
            "code_placement": "appendix_only",
        },
        "proxy": {"base_url": "http://localhost:8000"},
        "downloader": {"retry_times": 3},
        "validator": {"threshold": 75},
        "image_generator": {"style": "paper"},
    }
    config_path = _create_config_file(temp_dir, config_data)

    loaded = load_config(config_path)

    assert loaded["system"]["self_check_mode"] == "default"
    assert loaded["system"]["note_style"] == "full_detail"
    assert loaded["system"]["question_scope"] == "chapter_only"
    assert loaded["system"]["answer_placement"] == "end_of_chapter"
    assert loaded["system"]["code_placement"] == "appendix_only"


def test_production_config_has_correct_defaults():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

    loaded = load_config(config_path)

    assert loaded["system"]["self_check_mode"] == "static"
    assert loaded["system"]["note_style"] == "core_appendix"
    assert loaded["system"]["question_scope"] == "chapter_only"
    assert loaded["system"]["answer_placement"] == "inline_after_questions"
    assert loaded["system"]["code_placement"] == "appendix_only"
