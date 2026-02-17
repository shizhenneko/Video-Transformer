import json
from pathlib import Path

from analyzer.content_analyzer import ContentAnalyzer


def test_json_sanitizer_stray_g_prefix():
    text = '{\nG          "connections": []\n}'
    cleaned, count = ContentAnalyzer._strip_stray_token_prefixes(text)
    assert count == 1
    assert json.loads(cleaned)["connections"] == []


def test_json_sanitizer_stray_digit_prefix():
    text = '{\n1          "self_check": {"ok": true}\n}'
    cleaned, count = ContentAnalyzer._strip_stray_token_prefixes(text)
    assert count == 1
    assert json.loads(cleaned)["self_check"]["ok"] is True


def test_json_sanitizer_preserves_valid_json():
    text = '{\n  "key": "value",\n  "num": 1\n}'
    cleaned, count = ContentAnalyzer._strip_stray_token_prefixes(text)
    assert cleaned == text
    assert count == 0


def test_json_sanitizer_preserves_string_content():
    text = '{\n  "value": "G something",\n  "other": "1 value"\n}'
    cleaned, count = ContentAnalyzer._strip_stray_token_prefixes(text)
    assert cleaned == text
    assert count == 0
    assert json.loads(cleaned)["value"] == "G something"


def test_json_sanitizer_real_fixture():
    fixture_path = Path("data/output/logs/failed_json_1771209246.txt")
    assert fixture_path.exists()
    text = fixture_path.read_text(encoding="utf-8")
    cleaned, count = ContentAnalyzer._strip_stray_token_prefixes(text)
    assert count > 0
    parsed = ContentAnalyzer._try_repair_json(cleaned)
    assert parsed is not None
    assert "title" in parsed
