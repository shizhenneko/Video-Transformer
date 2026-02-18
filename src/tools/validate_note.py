"""Validator for lecture-style Markdown notes."""

import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


@dataclass
class Violation:
    rule: str
    line: int
    message: str


@dataclass
class ValidationResult:
    is_valid: bool
    violations: List[Violation] = field(default_factory=list)

    def get_summary(self) -> str:
        if self.is_valid:
            return "✓ Validation passed"

        lines = [f"✗ Validation failed with {len(self.violations)} violation(s):\n"]
        for v in self.violations:
            lines.append(f"  Line {v.line}: [{v.rule}] {v.message}")
        return "\n".join(lines)

    def get_violations_by_rule(self) -> Dict[str, List[Violation]]:
        by_rule = defaultdict(list)
        for v in self.violations:
            by_rule[v.rule].append(v)
        return dict(by_rule)


REQUIRED_SECTIONS = [
    "## 学习目标",
    "## 先修知识与快速回顾",
    "## 学习路线图（本讲你会走到哪里）",
    "## 🔍 讲义正文",
    "## 📌 覆盖清单 (Coverage Index)",
    "## 📎 附录 (Appendix)",
]

REQUIRED_SECTIONS_TEXTBOOK = [
    "## 核心概念图谱",
    "## 主题详解",
    "## 实战与代码",
    "## FAQ / 避坑指南",
    "## 📎 附录 (Appendix)",
]

TEXTBOOK_SECTION_MARKERS = REQUIRED_SECTIONS_TEXTBOOK[:-1]
LEGACY_SECTION_MARKERS = REQUIRED_SECTIONS[:-2]

REQUIRED_CHAPTER_SUBSECTIONS = [
    "#### 动机：为什么要学这个？",
    "#### 直觉：用一句话抓住本质",
    "#### 推导/机制：用纯文本公式讲清楚",
    "#### 工程实践：怎么用、怎么调、怎么排查",
    "#### 示例：输入→步骤→输出",
    "#### 常见误区：错在哪里/怎么改",
    "#### 本章练习",
    "#### 本章参考答案",
]

FORBIDDEN_PATTERNS = {
    "latex_dollar": r"\$[^$]+\$",
    "latex_paren_inline": r"\\\([^)]+\\\)",
    "latex_bracket_display": r"\\\[[^\]]+\\\]",
    "html_details": r"<details>",
    "html_summary": r"<summary>",
    "placeholder_missing": r"未在源笔记中显式给出",
    "placeholder_tbd": r"\bTBD\b",
    "placeholder_todo": r"\bTODO\b",
    "placeholder_pending": r"待补充",
    "dict_repr_input": r"\{'input':",
    "dict_repr_steps": r"\{'steps':",
    "dict_repr_output": r"\{'output':",
}

TIMESTAMP_PATTERNS = [
    r"\b\d{1,2}:\d{2}\b",
    r"\(\d{1,2}:\d{2}[–—-]\d{1,2}:\d{2}\)",
]


def check_title(content: str) -> List[Violation]:
    violations = []
    lines = content.split("\n")

    if not lines:
        violations.append(
            Violation(rule="title_missing", line=1, message="Document is empty")
        )
        return violations

    first_line = lines[0].strip()
    if not first_line.startswith("# "):
        violations.append(
            Violation(
                rule="title_missing",
                line=1,
                message="First line must be a level-1 heading (# ...)",
            )
        )

    return violations


def detect_lecture_format(content: str) -> str:
    if any(section in content for section in TEXTBOOK_SECTION_MARKERS):
        return "textbook"
    if any(section in content for section in LEGACY_SECTION_MARKERS):
        return "legacy"
    return "legacy"


def check_required_sections(
    content: str, required_sections: List[str] = REQUIRED_SECTIONS
) -> List[Violation]:
    violations = []

    for section in required_sections:
        if section not in content:
            violations.append(
                Violation(
                    rule="required_section_missing",
                    line=0,
                    message=f"Missing required section: {section}",
                )
            )

    return violations


def check_chapter_structure(content: str) -> List[Violation]:
    violations = []
    lines = content.split("\n")

    chapter_pattern = r"^###\s+第\d+章："
    chapters = []

    for i, line in enumerate(lines, 1):
        if re.match(chapter_pattern, line.strip()):
            chapters.append((i, line.strip()))

    if not chapters:
        violations.append(
            Violation(
                rule="no_chapters",
                line=0,
                message="No chapters found (expected at least 1 chapter with format '### 第X章：...')",
            )
        )
        return violations

    for chapter_line, chapter_title in chapters:
        chapter_end = len(lines)
        for next_chapter_line, _ in chapters:
            if next_chapter_line > chapter_line:
                chapter_end = next_chapter_line - 1
                break

        chapter_content = "\n".join(lines[chapter_line - 1 : chapter_end])

        for subsection in REQUIRED_CHAPTER_SUBSECTIONS:
            if subsection not in chapter_content:
                violations.append(
                    Violation(
                        rule="chapter_missing_subsection",
                        line=chapter_line,
                        message=f"Chapter '{chapter_title}' missing required subsection: {subsection}",
                    )
                )

        exercise_section_match = re.search(
            r"####\s+本章练习\s*\n(.*?)(?=####|$)", chapter_content, re.DOTALL
        )
        if exercise_section_match:
            exercise_content = exercise_section_match.group(1)
            exercise_count = len(re.findall(r"^\d+\.", exercise_content, re.MULTILINE))

            if exercise_count < 3:
                violations.append(
                    Violation(
                        rule="chapter_insufficient_exercises",
                        line=chapter_line,
                        message=f"Chapter '{chapter_title}' has {exercise_count} exercises (minimum 3 required)",
                    )
                )

    return violations


def check_forbidden_patterns(content: str) -> List[Violation]:
    violations = []
    lines = content.split("\n")

    for pattern_name, pattern in FORBIDDEN_PATTERNS.items():
        for i, line in enumerate(lines, 1):
            matches = re.finditer(pattern, line)
            for match in matches:
                violations.append(
                    Violation(
                        rule=pattern_name,
                        line=i,
                        message=f"Forbidden pattern '{pattern_name}' found: {match.group()[:50]}",
                    )
                )

    return violations


def check_timestamps_in_main_text(content: str) -> List[Violation]:
    violations = []
    lines = content.split("\n")

    appendix_line = None
    for i, line in enumerate(lines, 1):
        if "## 📎 附录 (Appendix)" in line:
            appendix_line = i
            break

    if appendix_line is None:
        appendix_line = len(lines) + 1

    for pattern in TIMESTAMP_PATTERNS:
        for i, line in enumerate(lines, 1):
            if i >= appendix_line:
                break

            matches = re.finditer(pattern, line)
            for match in matches:
                violations.append(
                    Violation(
                        rule="timestamp_in_main_text",
                        line=i,
                        message=f"Timestamp found in main text (should only appear in appendix): {match.group()}",
                    )
                )

    return violations


def validate_note(content: str) -> ValidationResult:
    all_violations = []
    lecture_format = detect_lecture_format(content)

    all_violations.extend(check_title(content))
    if lecture_format == "textbook":
        all_violations.extend(
            check_required_sections(content, REQUIRED_SECTIONS_TEXTBOOK)
        )
    else:
        all_violations.extend(check_required_sections(content))
        all_violations.extend(check_chapter_structure(content))
    all_violations.extend(check_forbidden_patterns(content))
    all_violations.extend(check_timestamps_in_main_text(content))

    all_violations.sort(key=lambda v: (v.line, v.rule))

    return ValidationResult(
        is_valid=len(all_violations) == 0, violations=all_violations
    )


def validate_file(file_path: Path) -> ValidationResult:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return validate_note(content)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            violations=[
                Violation(rule="file_error", line=0, message=f"Error reading file: {e}")
            ],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate lecture-style Markdown notes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.tools.validate_note note.md
  python -m src.tools.validate_note --glob "data/output/documents/*.md"
        """,
    )
    parser.add_argument("file", nargs="?", help="Markdown file to validate")
    parser.add_argument("--glob", help="Glob pattern for batch validation")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )

    args = parser.parse_args()

    if not args.file and not args.glob:
        parser.print_help()
        sys.exit(1)

    files_to_validate = []

    if args.glob:
        from glob import glob

        files_to_validate = [Path(f) for f in glob(args.glob, recursive=True)]
        if not files_to_validate:
            print(f"No files found matching pattern: {args.glob}")
            sys.exit(1)
    elif args.file:
        files_to_validate = [Path(args.file)]

    total_files = len(files_to_validate)
    valid_files = 0
    total_violations = 0

    for file_path in files_to_validate:
        if not file_path.exists():
            print(f"✗ {file_path}: File not found")
            continue

        result = validate_file(file_path)

        if result.is_valid:
            valid_files += 1
            if args.verbose or total_files == 1:
                print(f"✓ {file_path}: Valid")
        else:
            total_violations += len(result.violations)
            print(f"\n✗ {file_path}:")
            if args.verbose or total_files == 1:
                print(result.get_summary())
            else:
                print(f"  {len(result.violations)} violation(s) found")

    if total_files > 1:
        print(f"\n{'=' * 60}")
        print(f"Summary: {valid_files}/{total_files} files valid")
        if total_violations > 0:
            print(f"Total violations: {total_violations}")

    sys.exit(0 if valid_files == total_files else 1)


if __name__ == "__main__":
    main()
