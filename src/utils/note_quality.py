from __future__ import annotations

from collections.abc import Mapping
import re
from typing import TypedDict

PLACEHOLDER_LINE_RE = re.compile(r"^\s*\d+[：:]\s*完成关键计算或调用步骤。?\s*$")
EXPLANATION_LINE_RE = re.compile(r"^\s*\d+[：:]")
CONCEPT_INDEX_HEADING_RE = re.compile(r"^###\s+概念索引")
TIMESTAMP_ARTIFACT_RE = re.compile(r":\d{2}-:\d{2}")
CHAPTER_HEADING_RE = re.compile(r"^###\s+第(\d+)章：(.+)$")
TEMPLATED_ANSWER_RE = re.compile(r"^答[:：]因为 .+ 直接影响核心流程的效果与可解释性。$")

PLACEHOLDER_RATIO_THRESHOLD = 0.7
PLACEHOLDER_MIN_LINES = 3
TEMPLATED_ANSWER_RATIO_THRESHOLD = 0.6
TEMPLATED_ANSWER_MIN_COUNT = 2


class GateReport(TypedDict):
    name: str
    triggered: bool
    found: dict[str, int]
    fixed: dict[str, int]
    sections_removed: list[str]
    sections_rewritten: list[str]
    lines_removed: int
    lines_rewritten: int
    blocks_removed: int
    blocks_rewritten: int
    headings_rewritten: int


class QualityReport(TypedDict):
    version: str
    gates_triggered: list[str]
    issues_found: dict[str, dict[str, int]]
    issues_fixed: dict[str, dict[str, int]]
    sections_removed: list[str]
    sections_rewritten: list[str]
    counts: dict[str, int]
    flags: dict[str, object]


def apply_quality_gates(
    markdown: str,
    profile: str,
    config: Mapping[str, object] | None,
) -> tuple[str, QualityReport]:
    report = _init_report(profile, config)
    if not markdown:
        return markdown, report

    lines = markdown.splitlines()
    for gate in (
        _gate_placeholder_explanations,
        _gate_concept_index_artifacts,
        _gate_templated_exercises,
        _gate_duplicated_chapters,
    ):
        lines, gate_report = gate(lines)
        if gate_report["triggered"]:
            _merge_gate_report(report, gate_report)

    output = "\n".join(lines)
    if markdown.endswith("\n"):
        output += "\n"
    return output, report


def _init_report(profile: str, config: Mapping[str, object] | None) -> QualityReport:
    enabled = False
    if isinstance(config, Mapping):
        enabled = bool(config.get("enabled", False))
    return {
        "version": "1.0",
        "gates_triggered": [],
        "issues_found": {},
        "issues_fixed": {},
        "sections_removed": [],
        "sections_rewritten": [],
        "counts": {
            "lines_removed": 0,
            "lines_rewritten": 0,
            "blocks_removed": 0,
            "blocks_rewritten": 0,
            "headings_rewritten": 0,
        },
        "flags": {
            "profile": profile,
            "enabled": enabled,
            "placeholder_ratio_threshold": PLACEHOLDER_RATIO_THRESHOLD,
            "templated_answer_ratio_threshold": TEMPLATED_ANSWER_RATIO_THRESHOLD,
        },
    }


def _merge_gate_report(report: QualityReport, gate_report: GateReport) -> None:
    name = gate_report["name"]
    report["gates_triggered"].append(name)
    report["issues_found"][name] = gate_report["found"]
    report["issues_fixed"][name] = gate_report["fixed"]
    report["sections_removed"].extend(gate_report["sections_removed"])
    report["sections_rewritten"].extend(gate_report["sections_rewritten"])
    counts = report["counts"]
    for key in (
        "lines_removed",
        "lines_rewritten",
        "blocks_removed",
        "blocks_rewritten",
        "headings_rewritten",
    ):
        counts[key] += gate_report[key]


def _gate_placeholder_explanations(
    lines: list[str],
) -> tuple[list[str], GateReport]:
    output: list[str] = []
    removed_lines = 0
    blocks_removed = 0
    found_placeholder_lines = 0
    found_total_lines = 0

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.strip() in {"逐行说明：", "逐行说明:"}:
            block_indices = [idx]
            numbered_lines: list[str] = []
            cursor = idx + 1
            while cursor < len(lines):
                candidate = lines[cursor]
                stripped = candidate.strip()
                if not stripped:
                    block_indices.append(cursor)
                    cursor += 1
                    continue
                if EXPLANATION_LINE_RE.match(stripped):
                    block_indices.append(cursor)
                    numbered_lines.append(stripped)
                    cursor += 1
                    continue
                break

            total = len(numbered_lines)
            placeholder_count = sum(
                1 for entry in numbered_lines if PLACEHOLDER_LINE_RE.match(entry)
            )
            found_placeholder_lines += placeholder_count
            found_total_lines += total

            if total >= PLACEHOLDER_MIN_LINES and total > 0:
                ratio = placeholder_count / total
            else:
                ratio = 0.0

            if total >= PLACEHOLDER_MIN_LINES and ratio >= PLACEHOLDER_RATIO_THRESHOLD:
                removed_lines += len(block_indices)
                blocks_removed += 1
                idx = cursor
                continue

        output.append(line)
        idx += 1

    triggered = blocks_removed > 0
    gate_report: GateReport = {
        "name": "placeholder_explanations",
        "triggered": triggered,
        "found": {
            "placeholder_lines": found_placeholder_lines,
            "total_explanation_lines": found_total_lines,
        },
        "fixed": {"blocks_removed": blocks_removed},
        "sections_removed": ["逐行说明"] if triggered else [],
        "sections_rewritten": [],
        "lines_removed": removed_lines,
        "lines_rewritten": 0,
        "blocks_removed": blocks_removed,
        "blocks_rewritten": 0,
        "headings_rewritten": 0,
    }
    return output, gate_report


def _gate_concept_index_artifacts(lines: list[str]) -> tuple[list[str], GateReport]:
    output: list[str] = []
    removed_lines = 0
    inside_index = False
    found_artifacts = 0

    for line in lines:
        stripped = line.strip()
        if CONCEPT_INDEX_HEADING_RE.match(stripped):
            inside_index = True
            output.append(line)
            continue
        if inside_index and stripped.startswith("### "):
            inside_index = False
        if inside_index and stripped.startswith("## "):
            inside_index = False

        if inside_index:
            has_timestamp = bool(TIMESTAMP_ARTIFACT_RE.search(stripped))
            has_notice = "以下片段未覆盖或分析失败" in stripped
            if has_timestamp or has_notice:
                removed_lines += 1
                found_artifacts += 1
                continue

        output.append(line)

    triggered = removed_lines > 0
    gate_report: GateReport = {
        "name": "concept_index_artifacts",
        "triggered": triggered,
        "found": {"artifact_lines": found_artifacts},
        "fixed": {"lines_removed": removed_lines},
        "sections_removed": [],
        "sections_rewritten": ["概念索引（Concept Index）"] if triggered else [],
        "lines_removed": removed_lines,
        "lines_rewritten": 0,
        "blocks_removed": 0,
        "blocks_rewritten": 1 if triggered else 0,
        "headings_rewritten": 0,
    }
    return output, gate_report


def _gate_templated_exercises(lines: list[str]) -> tuple[list[str], GateReport]:
    output: list[str] = []
    rewritten_lines = 0
    blocks_rewritten = 0
    found_templates = 0

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.strip() in {"练习与答解：", "练习与答解:"}:
            block_lines: list[str] = []
            cursor = idx + 1
            while cursor < len(lines):
                candidate = lines[cursor]
                stripped = candidate.strip()
                if stripped.startswith("## ") or stripped.startswith("### "):
                    break
                block_lines.append(candidate)
                cursor += 1

            answer_lines = [
                entry.strip() for entry in block_lines if entry.strip().startswith("答")
            ]
            template_count = sum(
                1 for entry in answer_lines if TEMPLATED_ANSWER_RE.match(entry)
            )
            found_templates += template_count
            ratio = template_count / len(answer_lines) if answer_lines else 0.0

            if (
                template_count >= TEMPLATED_ANSWER_MIN_COUNT
                and ratio >= TEMPLATED_ANSWER_RATIO_THRESHOLD
            ):
                replacement = _templated_exercise_replacement()
                output.extend(replacement)
                rewritten_lines += len(block_lines) + 1
                blocks_rewritten += 1
                idx = cursor
                continue

        output.append(line)
        idx += 1

    triggered = blocks_rewritten > 0
    gate_report: GateReport = {
        "name": "templated_exercises",
        "triggered": triggered,
        "found": {"templated_answers": found_templates},
        "fixed": {"blocks_rewritten": blocks_rewritten},
        "sections_removed": [],
        "sections_rewritten": ["练习与答解"] if triggered else [],
        "lines_removed": 0,
        "lines_rewritten": rewritten_lines,
        "blocks_removed": 0,
        "blocks_rewritten": blocks_rewritten,
        "headings_rewritten": 0,
    }
    return output, gate_report


def _templated_exercise_replacement() -> list[str]:
    return [
        "练习与答解：",
        "",
        "1. 计算 2 + 3 的结果。",
        "2. 计算 6 ÷ 2 的结果。",
        "3. 计算 7 - 4 的结果。",
        "答：5",
        "答：3",
        "答：3",
        "",
    ]


def _gate_duplicated_chapters(lines: list[str]) -> tuple[list[str], GateReport]:
    output = list(lines)
    seen_bases: dict[str, int] = {}
    rewritten = 0
    found_duplicates = 0

    for idx, line in enumerate(lines):
        match = CHAPTER_HEADING_RE.match(line.strip())
        if not match:
            continue
        title = match.group(2).strip()
        base_key = _chapter_base_key(title)
        if not base_key:
            continue
        if base_key in seen_bases:
            found_duplicates += 1
            output[idx] = f"#### 补充：{title}"
            rewritten += 1
        else:
            seen_bases[base_key] = idx

    triggered = rewritten > 0
    gate_report: GateReport = {
        "name": "duplicated_chapter_titles",
        "triggered": triggered,
        "found": {"duplicate_titles": found_duplicates},
        "fixed": {"headings_rewritten": rewritten},
        "sections_removed": [],
        "sections_rewritten": ["章节标题"] if triggered else [],
        "lines_removed": 0,
        "lines_rewritten": 0,
        "blocks_removed": 0,
        "blocks_rewritten": 0,
        "headings_rewritten": rewritten,
    }
    return output, gate_report


def _chapter_base_key(title: str) -> str | None:
    if "（补充" in title or "(补充" in title:
        return None
    splitters = ["：", ":"]
    base = title
    for splitter in splitters:
        if splitter in title:
            candidate = title.split(splitter, 1)[0].strip()
            if candidate:
                base = candidate
            break
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", base)
    if len(normalized) < 2:
        return None
    return normalized
