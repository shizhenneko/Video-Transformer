from __future__ import annotations

from pathlib import Path
import re
from typing import cast

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_NOTE_PATH = ROOT_DIR / "data/output/documents/BV19TKHzUEVs_p2_knowledge_note.md"
COMPRESSED_NOTE_PATH = (
    ROOT_DIR / "data/output/documents/BV19TKHzUEVs_p2_knowledge_note_compressed.md"
)

TOPIC_PATTERN: re.Pattern[str] = re.compile(r"^#### (\d+)\. (.+)$", re.MULTILINE)
LINE_BUDGET = 300
MIND_MAP_REFERENCE = "../blueprints/BV19TKHzUEVs_p2_mind_map.png"
FORBIDDEN_MARKERS = [
    "**🧩 挑战",
    "**💡 原理拆解",
    "**🌰 自包含示例",
    "**⚠️ 常见误区",
    "**🔗 关联知识",
]


@pytest.fixture(scope="session")
def extracted_topics() -> list[str]:
    if not SOURCE_NOTE_PATH.exists():
        pytest.fail(f"Source note missing: {SOURCE_NOTE_PATH}")

    source_text = SOURCE_NOTE_PATH.read_text(encoding="utf-8")
    matches = cast(list[tuple[str, str]], TOPIC_PATTERN.findall(source_text))
    topics = [topic.strip() for _, topic in matches]

    if len(topics) != 54:
        pytest.fail(f"Expected 54 topics, found {len(topics)}")

    return topics


@pytest.fixture(scope="session")
def compressed_note_text() -> str:
    if not COMPRESSED_NOTE_PATH.exists():
        pytest.fail(f"Compressed note missing: {COMPRESSED_NOTE_PATH}")

    return COMPRESSED_NOTE_PATH.read_text(encoding="utf-8")


def _extract_coverage_section(text: str) -> str:
    match = re.search(r"^## 📌 覆盖清单 \(Coverage Index\)\s*$", text, re.MULTILINE)
    if not match:
        return ""

    start = match.end()
    next_heading = re.search(r"^##\s", text[start:], re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end]


def test_compressed_file_exists():
    assert COMPRESSED_NOTE_PATH.exists(), (
        f"Missing compressed note: {COMPRESSED_NOTE_PATH}"
    )


def test_compressed_file_line_count_within_budget(
    compressed_note_text: str,
):
    line_count = len(compressed_note_text.splitlines())
    assert line_count <= LINE_BUDGET, (
        f"Line count {line_count} exceeds budget {LINE_BUDGET}"
    )


def test_compressed_file_has_mind_map_reference(compressed_note_text: str):
    assert MIND_MAP_REFERENCE in compressed_note_text, (
        "Mind map reference missing from compressed note"
    )


def test_compressed_file_has_coverage_index_with_54_entries(
    compressed_note_text: str, extracted_topics: list[str]
):
    coverage_section = _extract_coverage_section(compressed_note_text)
    assert coverage_section, "Coverage Index section missing"

    missing_topics = [
        topic for topic in extracted_topics if topic not in coverage_section
    ]
    assert not missing_topics, "Coverage Index missing topics: " + ", ".join(
        missing_topics
    )


def test_compressed_file_contains_all_topics(
    compressed_note_text: str, extracted_topics: list[str]
):
    missing_topics = [
        topic for topic in extracted_topics if topic not in compressed_note_text
    ]
    assert not missing_topics, "Compressed note missing topics: " + ", ".join(
        missing_topics
    )


def test_compressed_file_no_per_section_template_markers(
    compressed_note_text: str,
):
    violations: dict[str, int] = {
        marker: compressed_note_text.count(marker)
        for marker in FORBIDDEN_MARKERS
        if marker in compressed_note_text
    }
    assert not violations, f"Forbidden markers present: {violations}"
