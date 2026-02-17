from __future__ import annotations

import re
from typing import Protocol


APPENDIX_HEADING = "## 📎 附录 (Appendix)"
REQUIRED_HEADINGS_DEFAULT = [
    "## 📌 覆盖清单 (Coverage Index)",
    APPENDIX_HEADING,
]
FORBIDDEN_PATTERNS_DEFAULT = [
    "**🧩 挑战",
    "**✅ 自测（做完再看答案）**",
]


def validate_markdown_structure(markdown: str, mode: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    normalized_mode = (mode or "").strip().lower()

    if normalized_mode == "default":
        for heading in REQUIRED_HEADINGS_DEFAULT:
            if heading not in markdown:
                errors.append(f"缺少必需标题: {heading}")

        for pattern in FORBIDDEN_PATTERNS_DEFAULT:
            if pattern in markdown:
                errors.append(f"禁用内容命中: {pattern}")

        appendix_index = markdown.find(APPENDIX_HEADING)
        fence_index = markdown.find("```")
        if appendix_index != -1 and fence_index != -1 and fence_index < appendix_index:
            errors.append("代码围栏出现在附录之前")

    return len(errors) == 0, errors


def detect_stub_output(markdown: str) -> bool:
    if not markdown or not markdown.strip():
        return True

    normalized = " ".join(markdown.lower().split())
    if normalized in {"final report", "final report.", "final"}:
        return True
    if normalized.startswith("final report") and len(normalized) <= 30:
        return True

    lines = markdown.splitlines()
    if not _has_substantive_content(lines):
        return True

    if _has_empty_section(lines):
        return True

    return False


class KnowledgeDocumentLike(Protocol):
    def to_markdown(
        self,
        image_paths: list[str] | None = None,
        self_check_mode: str = "static",
    ) -> str: ...


def validate_knowledge_document(
    doc: KnowledgeDocumentLike,
    mode: str,
) -> tuple[bool, list[str]]:
    markdown = doc.to_markdown(self_check_mode=mode)
    errors: list[str] = []

    if detect_stub_output(markdown):
        errors.append("检测到疑似占位/空内容输出")

    _, structure_errors = validate_markdown_structure(markdown, mode)
    errors.extend(structure_errors)

    return len(errors) == 0, errors


def _has_substantive_content(lines: list[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("```"):
            continue
        if stripped in {"---", "***"}:
            continue

        cleaned = re.sub(r"^(\s*>+\s*)", "", stripped)
        cleaned = re.sub(r"^(\s*[-*+]\s+)", "", cleaned)
        cleaned = re.sub(r"^(\s*\d+\.\s+)", "", cleaned)
        if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", cleaned):
            return True

    return False


def _has_empty_section(lines: list[str]) -> bool:
    for idx, line in enumerate(lines):
        if not re.match(r"^#{1,6}\s+", line.strip()):
            continue

        for next_line in lines[idx + 1 :]:
            next_stripped = next_line.strip()
            if not next_stripped:
                continue
            if re.match(r"^#{1,6}\s+", next_stripped):
                return True
            break

    return False
