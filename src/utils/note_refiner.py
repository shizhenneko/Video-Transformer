from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Callable, Mapping
from typing import cast

from .note_refiner_contract import (
    DEFAULT_BUDGET_SPEC,
    DEFAULT_CODE_BUDGET_POLICY,
    DEFAULT_COVERAGE_POLICY,
    DEFAULT_MAPPING_RULES,
    DEFAULT_OUTPUT_STRUCTURE,
    HEADING_COVERAGE_INDEX,
    MAPPING_TABLE_HEADER,
    MAPPING_TABLE_SEPARATOR,
    SOURCE_GLOSSARY_HEADING,
    SOURCE_KEY_TAKEAWAYS_HEADING,
    BudgetSpec,
    CodeBudgetPolicy,
    CoveragePolicy,
    KeyTakeawayMappingRules,
    OutputStructure,
    budget_for_duration,
    count_budget_lines,
    format_budget_warning,
    normalize_takeaway,
    normalize_topic_title,
)

TOPIC_HEADING_RE = re.compile(r"^#### (\d+)\. (.+)$")
SECTION_HEADING_RE = re.compile(r"^##\s+")
SUBSECTION_HEADING_RE = re.compile(r"^###\s+")
CODE_FENCE_RE = re.compile(r"^```")

EXPLANATION_MARKERS = (
    "**💡 原理解析**",
    "**💡 原理拆解**",
)
EXAMPLE_MARKERS = (
    "**🌰 举个栗子**",
    "**🌰 示例**",
    "**🌰 自包含示例",
)
PITFALL_MARKER = "**⚠️ 常见误区**"

LECTURE_SECTION_MARKERS = (
    "## 学习目标",
    "## 先修知识与快速回顾",
    "## 学习路线图（本讲你会走到哪里）",
    "## 🔍 讲义正文",
    "## 讲义正文",
    "## 核心概念图谱",
    "## 主题详解",
    "## 实战与代码",
    "## FAQ / 避坑指南",
    "## 📎 附录 (Appendix)",
    "## 📎 附录",
)

TEXTBOOK_LECTURE_MARKERS = (
    "## 核心概念图谱",
    "## 主题详解",
    "## 实战与代码",
    "## FAQ / 避坑指南",
)

TEXTBOOK_APPENDIX_HEADINGS = (
    "## 📎 附录 (Appendix)",
    "## 📎 附录",
)

APPENDIX_HEADINGS = (
    "## 📎 附录 (Appendix)",
    "## 📎 附录",
)

TIMESTAMP_RE = re.compile(
    r"\s+\(\d{1,2}:\d{2}(?::\d{2})?(?:[–-]\d{1,2}:\d{2}(?::\d{2})?)?\)\s*$"
)


@dataclass(frozen=True)
class TopicRecord:
    number: int | None
    title: str
    normalized: str
    key_points: list[str]
    pitfalls: list[str]
    expansions: list[str]


@dataclass(frozen=True)
class RawTopicBlock:
    number: int | None
    title: str
    lines: list[str]


def refine_note(
    markdown: str, *, duration_seconds: float | None, config: Mapping[str, object]
) -> str:
    budget_spec = _coerce_budget_spec(config.get("budget_spec"))
    coverage_policy = _coerce_coverage_policy(config.get("coverage_policy"))
    code_policy = _coerce_code_budget_policy(config.get("code_budget_policy"))
    mapping_rules = _coerce_mapping_rules(config.get("mapping_rules"))
    _ = _coerce_output_structure(config.get("output_structure"))

    lines = markdown.splitlines()
    if _is_textbook_lecture_note(lines):
        return _refine_textbook_lecture_note(
            markdown,
            duration_seconds=duration_seconds,
            budget_spec=budget_spec,
        )
    if _is_lecture_note(lines):
        return _refine_lecture_note(
            markdown,
            budget_spec=budget_spec,
            mapping_rules=mapping_rules,
        )

    key_takeaways = _parse_key_takeaways(lines)
    glossary_terms = _parse_glossary_terms(lines)
    raw_topics = _parse_topic_blocks(lines)
    if not raw_topics:
        raw_topics = _fallback_topics_from_coverage(lines)

    topics = _build_topics(raw_topics)
    topics = _dedupe_topics(topics)

    mapping_results, unmapped_takeaways = _map_key_takeaways_to_topics(
        key_takeaways,
        topics,
        glossary_terms,
        mapping_rules,
    )

    mistakes_lines = _build_mistakes_section(topics)
    key_points_lines = _build_key_points_section(key_takeaways, topics)
    mapping_lines = _build_mapping_section(mapping_results, mapping_rules)
    unmapped_lines = _build_unmapped_section(unmapped_takeaways, mapping_rules)
    coverage_lines = _build_coverage_section(topics)

    expanded_heading = DEFAULT_OUTPUT_STRUCTURE.required_headings[2]
    coverage_heading = HEADING_COVERAGE_INDEX

    expanded_lines = _build_expanded_section(
        topics,
        mapping_results,
        duration_seconds,
        budget_spec,
        prefix_lines=mistakes_lines + key_points_lines + mapping_lines + unmapped_lines,
        suffix_lines=[coverage_heading, ""] + coverage_lines,
    )

    output_lines = [
        *mistakes_lines,
        *key_points_lines,
        *mapping_lines,
        *unmapped_lines,
        expanded_heading,
        "",
        *expanded_lines,
        coverage_heading,
        "",
        *coverage_lines,
    ]

    output = "\n".join(output_lines).rstrip() + "\n"

    if duration_seconds is not None and coverage_policy.warn_on_budget_exceed:
        budget = budget_for_duration(duration_seconds, budget_spec)
        actual_lines = count_budget_lines(
            output, exclude_code_from_budget=code_policy.exclude_code_from_budget
        )
        if actual_lines > budget.max_lines:
            warning = format_budget_warning(actual_lines, budget.target_lines)
            output = output.rstrip() + "\n" + warning + "\n"

    return output


def _is_lecture_note(lines: list[str]) -> bool:
    first_non_empty = next((line for line in lines if line.strip()), "")
    if not first_non_empty.startswith("# "):
        return False
    text = "\n".join(lines)
    return any(marker in text for marker in LECTURE_SECTION_MARKERS)


def _is_textbook_lecture_note(lines: list[str]) -> bool:
    first_non_empty = next((line for line in lines if line.strip()), "")
    if not first_non_empty.startswith("# "):
        return False
    text = "\n".join(lines)
    return all(marker in text for marker in TEXTBOOK_LECTURE_MARKERS)


@dataclass(frozen=True)
class SectionBlock:
    heading: str
    lines: list[str]


def _split_sections(lines: list[str]) -> tuple[list[str], list[SectionBlock]]:
    preamble: list[str] = []
    sections: list[SectionBlock] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_heading, current_lines
        if current_heading is not None:
            sections.append(SectionBlock(heading=current_heading, lines=current_lines))
        current_heading = None
        current_lines = []

    for line in lines:
        if line.startswith("## "):
            flush()
            current_heading = line
            continue
        if current_heading is None:
            preamble.append(line)
        else:
            current_lines.append(line)

    flush()
    return preamble, sections


def _join_sections(preamble: list[str], sections: list[SectionBlock]) -> list[str]:
    combined: list[str] = []
    combined.extend(preamble)
    for section in sections:
        combined.append(section.heading)
        combined.extend(section.lines)
    return combined


def _update_sections(
    lines: list[str],
    updaters: dict[str, Callable[[list[str]], list[str]]],
) -> list[str]:
    preamble, sections = _split_sections(lines)
    updated_sections: list[SectionBlock] = []
    for section in sections:
        updater = updaters.get(section.heading)
        if updater is None:
            updated_sections.append(section)
        else:
            updated_sections.append(
                SectionBlock(heading=section.heading, lines=updater(section.lines))
            )
    return _join_sections(preamble, updated_sections)


def _refine_textbook_lecture_note(
    markdown: str,
    *,
    duration_seconds: float | None,
    budget_spec: BudgetSpec,
) -> str:
    lines = markdown.splitlines()
    if duration_seconds is None:
        return markdown.rstrip() + "\n"

    budget = budget_for_duration(duration_seconds, budget_spec)
    trimmed = _trim_textbook_lecture_lines(lines, budget.target_lines)
    return "\n".join(trimmed).rstrip() + "\n"


def _trim_textbook_lecture_lines(lines: list[str], target_lines: int) -> list[str]:
    if len(lines) <= target_lines:
        return lines

    trimmed = list(lines)

    reducers = [
        _trim_textbook_examples,
        _trim_textbook_long_bullets,
        _trim_textbook_faq_items,
        _trim_textbook_appendix_code,
    ]

    for reducer in reducers:
        if len(trimmed) <= target_lines:
            break
        trimmed = reducer(trimmed)

    return trimmed


def _trim_textbook_examples(lines: list[str]) -> list[str]:
    def trim_section(section_lines: list[str]) -> list[str]:
        output: list[str] = []
        example_count = 0
        skipping = False
        for line in section_lines:
            if line.startswith("### 示例 "):
                example_count += 1
                if example_count > 1:
                    skipping = True
                    continue
                skipping = False
            if skipping:
                continue
            output.append(line)
        return output

    return _update_sections(lines, {"## 实战与代码": trim_section})


def _trim_textbook_long_bullets(lines: list[str]) -> list[str]:
    def trim_concept_map(section_lines: list[str]) -> list[str]:
        output: list[str] = []
        subtopic_count = 0
        for line in section_lines:
            if line.startswith("- "):
                subtopic_count = 0
                output.append(line)
                continue
            if line.startswith("  - "):
                if subtopic_count < 3:
                    output.append(line)
                    subtopic_count += 1
                continue
            output.append(line)
        return output

    def trim_chapter_bullets(section_lines: list[str]) -> list[str]:
        output: list[str] = []
        in_bullets = False
        bullet_count = 0
        for line in section_lines:
            stripped = line.strip()
            if stripped == "内容串讲：":
                in_bullets = True
                bullet_count = 0
                output.append(line)
                continue
            if in_bullets and line.lstrip().startswith("- "):
                bullet_count += 1
                if bullet_count <= 5:
                    output.append(line)
                continue
            if in_bullets and stripped and not line.lstrip().startswith("- "):
                in_bullets = False
            output.append(line)
        return output

    return _update_sections(
        lines,
        {
            "## 核心概念图谱": trim_concept_map,
            "## 主题详解": trim_chapter_bullets,
        },
    )


def _trim_textbook_faq_items(lines: list[str]) -> list[str]:
    def trim_section(section_lines: list[str]) -> list[str]:
        output: list[str] = []
        in_pitfalls = False
        pitfall_count = 0
        in_exercises = False
        question_count = 0
        answer_count = 0

        for line in section_lines:
            stripped = line.strip()
            if stripped == "常见坑：":
                in_pitfalls = True
                pitfall_count = 0
                output.append(line)
                continue
            if stripped == "练习与答解：":
                in_pitfalls = False
                in_exercises = True
                question_count = 0
                answer_count = 0
                output.append(line)
                continue

            if in_pitfalls and line.lstrip().startswith("- "):
                pitfall_count += 1
                if pitfall_count <= 4:
                    output.append(line)
                continue
            if in_pitfalls and stripped and not line.lstrip().startswith("- "):
                in_pitfalls = False

            if in_exercises:
                if re.match(r"^\d+\.\s+", stripped):
                    question_count += 1
                    if question_count <= 3:
                        output.append(line)
                    continue
                if re.match(r"^(答|答解)[:：]", stripped):
                    answer_count += 1
                    if answer_count <= 3:
                        output.append(line)
                    continue

            output.append(line)

        return output

    return _update_sections(lines, {"## FAQ / 避坑指南": trim_section})


def _trim_textbook_appendix_code(lines: list[str]) -> list[str]:
    def trim_section(section_lines: list[str]) -> list[str]:
        output: list[str] = []
        in_code = False
        code_blocks = 0
        allowed_blocks = 1
        in_code_area = False
        for line in section_lines:
            stripped = line.strip()
            if stripped == "### 代码与伪代码":
                in_code_area = True
                output.append(line)
                continue
            if (
                in_code_area
                and stripped.startswith("### ")
                and stripped != "### 代码与伪代码"
            ):
                in_code_area = False
            if in_code_area and CODE_FENCE_RE.match(stripped):
                if not in_code:
                    code_blocks += 1
                    in_code = True
                    if code_blocks > allowed_blocks:
                        continue
                else:
                    in_code = False
                    if code_blocks > allowed_blocks:
                        continue
            if in_code_area and code_blocks > allowed_blocks:
                if in_code:
                    continue
            output.append(line)
        return output

    return _update_sections(
        lines, {heading: trim_section for heading in TEXTBOOK_APPENDIX_HEADINGS}
    )


def _refine_lecture_note(
    markdown: str,
    *,
    budget_spec: BudgetSpec,
    mapping_rules: KeyTakeawayMappingRules,
) -> str:
    lines = markdown.splitlines()
    key_takeaways = _parse_key_takeaways(lines)
    glossary_terms = _parse_glossary_terms(lines)
    raw_topics = _parse_topic_blocks(lines)
    if not raw_topics:
        raw_topics = _fallback_topics_from_coverage(lines)

    topics = _build_topics(raw_topics)
    topics = _dedupe_topics(topics)

    mapping_results, unmapped_takeaways = _map_key_takeaways_to_topics(
        key_takeaways,
        topics,
        glossary_terms,
        mapping_rules,
    )

    mistakes_lines = _build_mistakes_section(topics)
    key_points_lines = _build_key_points_section(key_takeaways, topics)
    mapping_lines = _build_mapping_section(mapping_results, mapping_rules)
    unmapped_lines = _build_unmapped_section(unmapped_takeaways, mapping_rules)
    coverage_lines = _build_coverage_section(topics)

    expanded_heading = DEFAULT_OUTPUT_STRUCTURE.required_headings[2]
    coverage_heading = HEADING_COVERAGE_INDEX

    expanded_lines = _build_expanded_section(
        topics,
        mapping_results,
        None,
        budget_spec,
        prefix_lines=mistakes_lines + key_points_lines + mapping_lines + unmapped_lines,
        suffix_lines=[coverage_heading, ""] + coverage_lines,
        topic_heading_prefix="#### ",
    )

    output_lines = [
        *mistakes_lines,
        *key_points_lines,
        *mapping_lines,
        *unmapped_lines,
        expanded_heading,
        "",
        *expanded_lines,
        coverage_heading,
        "",
        *coverage_lines,
    ]

    output_lines = _dedupe_lines_in_sections(output_lines)
    appendix_lines = _to_appendix_sections(output_lines)
    refiner_headings = {
        _convert_heading_level(DEFAULT_OUTPUT_STRUCTURE.required_headings[0], 3),
        _convert_heading_level(DEFAULT_OUTPUT_STRUCTURE.required_headings[1], 3),
        _convert_heading_level(DEFAULT_OUTPUT_STRUCTURE.required_headings[2], 3),
        _convert_heading_level(HEADING_COVERAGE_INDEX, 3),
        _convert_heading_level(mapping_rules.mapping_section_heading, 3),
        _convert_heading_level(mapping_rules.unmapped_section_heading, 3),
    }

    return _append_refiner_appendix(markdown, appendix_lines, refiner_headings)


def _append_refiner_appendix(
    markdown: str, appendix_lines: list[str], refiner_headings: set[str]
) -> str:
    lines = markdown.splitlines()
    appendix_start = _find_appendix_heading(lines)
    if appendix_start is None:
        heading = APPENDIX_HEADINGS[0]
        combined = _ensure_trailing_blank(lines) + [heading, ""] + appendix_lines
        return "\n".join(combined).rstrip() + "\n"

    appendix_end = _find_next_section(lines, appendix_start + 1)
    preserved = _strip_refiner_appendix_blocks(
        lines[appendix_start + 1 : appendix_end], refiner_headings
    )
    preserved = _ensure_trailing_blank(preserved)
    new_appendix = preserved + appendix_lines
    combined = lines[: appendix_start + 1] + new_appendix + lines[appendix_end:]
    return "\n".join(combined).rstrip() + "\n"


def _find_appendix_heading(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines):
        if line.strip() in APPENDIX_HEADINGS:
            return idx
    return None


def _find_next_section(lines: list[str], start: int) -> int:
    for idx in range(start, len(lines)):
        if lines[idx].startswith("## "):
            return idx
    return len(lines)


def _strip_refiner_appendix_blocks(
    lines: list[str], refiner_headings: set[str]
) -> list[str]:
    stripped: list[str] = []
    skipping = False
    for line in lines:
        line_stripped = line.strip()
        if skipping:
            if line_stripped.startswith("### ") or line_stripped.startswith("## "):
                skipping = False
            else:
                continue
        if line_stripped in refiner_headings:
            skipping = True
            continue
        stripped.append(line)
    return stripped


def _convert_heading_level(heading: str, level: int) -> str:
    text = heading.lstrip("#").strip()
    return f"{'#' * level} {text}"


def _dedupe_lines_in_sections(lines: list[str]) -> list[str]:
    """Deduplicate lines within sections by normalizing whitespace."""
    result: list[str] = []
    current_section_lines: list[str] = []
    current_section_heading: str | None = None

    def flush_section() -> None:
        nonlocal current_section_lines, current_section_heading
        if current_section_heading:
            result.append(current_section_heading)
        if current_section_lines:
            seen: set[str] = set()
            for line in current_section_lines:
                normalized = " ".join(line.split())
                if normalized and normalized not in seen:
                    result.append(line)
                    seen.add(normalized)
                elif not normalized:
                    result.append(line)
        current_section_lines = []
        current_section_heading = None

    for line in lines:
        if line.startswith("### "):
            flush_section()
            current_section_heading = line
        else:
            current_section_lines.append(line)

    flush_section()
    return result


def _to_appendix_sections(lines: list[str]) -> list[str]:
    converted: list[str] = []
    for line in lines:
        if line.startswith("## "):
            converted.append(_convert_heading_level(line, 3))
            continue
        if line.startswith("### "):
            converted.append(_convert_heading_level(line, 4))
            continue
        converted.append(line)
    return _dedupe_appendix_lines(converted)


def _dedupe_appendix_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if line.startswith("### ") or line.startswith("#### "):
            deduped.append(line)
            seen.clear()
            continue
        normalized = " ".join(line.strip().split())
        if not normalized:
            if deduped and deduped[-1] == "":
                continue
            deduped.append("")
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _ensure_trailing_blank(lines: list[str]) -> list[str]:
    if not lines:
        return []
    if lines[-1].strip():
        return lines + [""]
    return lines


def _coerce_budget_spec(value: object) -> BudgetSpec:
    if isinstance(value, BudgetSpec):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return BudgetSpec(
            lines_per_hour=_coerce_int(
                mapping, "lines_per_hour", 0, DEFAULT_BUDGET_SPEC.lines_per_hour
            ),
            min_cap=_coerce_int(mapping, "min_cap", 0, DEFAULT_BUDGET_SPEC.min_cap),
            max_cap=_coerce_int(mapping, "max_cap", 0, DEFAULT_BUDGET_SPEC.max_cap),
            tolerance_ratio=_coerce_float(
                mapping, "tolerance_ratio", 0.0, DEFAULT_BUDGET_SPEC.tolerance_ratio
            ),
        )
    return DEFAULT_BUDGET_SPEC


def _coerce_coverage_policy(value: object) -> CoveragePolicy:
    if isinstance(value, CoveragePolicy):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return CoveragePolicy(
            require_all_topics=_coerce_bool(
                mapping,
                "require_all_topics",
                DEFAULT_COVERAGE_POLICY.require_all_topics,
            ),
            budget_is_soft_constraint=_coerce_bool(
                mapping,
                "budget_is_soft_constraint",
                DEFAULT_COVERAGE_POLICY.budget_is_soft_constraint,
            ),
            warn_on_budget_exceed=_coerce_bool(
                mapping,
                "warn_on_budget_exceed",
                DEFAULT_COVERAGE_POLICY.warn_on_budget_exceed,
            ),
            budget_warning_template=_coerce_str(
                mapping,
                "budget_warning_template",
                DEFAULT_COVERAGE_POLICY.budget_warning_template,
            ),
        )
    return DEFAULT_COVERAGE_POLICY


def _coerce_code_budget_policy(value: object) -> CodeBudgetPolicy:
    if isinstance(value, CodeBudgetPolicy):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return CodeBudgetPolicy(
            exclude_code_from_budget=_coerce_bool(
                mapping,
                "exclude_code_from_budget",
                DEFAULT_CODE_BUDGET_POLICY.exclude_code_from_budget,
            )
        )
    return DEFAULT_CODE_BUDGET_POLICY


def _coerce_mapping_rules(value: object) -> KeyTakeawayMappingRules:
    if isinstance(value, KeyTakeawayMappingRules):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        match_order = _coerce_tuple(
            mapping, "match_order", DEFAULT_MAPPING_RULES.match_order
        )
        mapping_output_formats = _coerce_tuple(
            mapping,
            "mapping_output_formats",
            DEFAULT_MAPPING_RULES.mapping_output_formats,
        )
        return KeyTakeawayMappingRules(
            match_order=match_order,
            mapping_output_formats=mapping_output_formats,
            mapping_section_heading=_coerce_str(
                mapping,
                "mapping_section_heading",
                DEFAULT_MAPPING_RULES.mapping_section_heading,
            ),
            unmapped_section_heading=_coerce_str(
                mapping,
                "unmapped_section_heading",
                DEFAULT_MAPPING_RULES.unmapped_section_heading,
            ),
        )
    return DEFAULT_MAPPING_RULES


def _coerce_output_structure(value: object) -> OutputStructure:
    if isinstance(value, OutputStructure):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        required_headings = _coerce_tuple(
            mapping, "required_headings", DEFAULT_OUTPUT_STRUCTURE.required_headings
        )
        return OutputStructure(
            required_headings=required_headings,
            coverage_heading=_coerce_str(
                mapping, "coverage_heading", DEFAULT_OUTPUT_STRUCTURE.coverage_heading
            ),
            mapping_heading=_coerce_str(
                mapping, "mapping_heading", DEFAULT_OUTPUT_STRUCTURE.mapping_heading
            ),
            unmapped_heading=_coerce_str(
                mapping, "unmapped_heading", DEFAULT_OUTPUT_STRUCTURE.unmapped_heading
            ),
        )
    return DEFAULT_OUTPUT_STRUCTURE


def _coerce_int(
    mapping: Mapping[str, object], key: str, minimum: int, default: int
) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(value, minimum)
    if isinstance(value, float):
        return max(int(value), minimum)
    return default


def _coerce_float(
    mapping: Mapping[str, object], key: str, minimum: float, default: float
) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return max(float(value), minimum)
    return default


def _coerce_bool(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return default


def _coerce_str(mapping: Mapping[str, object], key: str, default: str) -> str:
    value = mapping.get(key, default)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _coerce_tuple(
    mapping: Mapping[str, object], key: str, default: tuple[str, ...]
) -> tuple[str, ...]:
    value = mapping.get(key, default)
    if isinstance(value, list):
        raw_list = cast(list[object], value)
        items = [item for item in raw_list if isinstance(item, str)]
        if len(items) == len(raw_list):
            return tuple(items)
    if isinstance(value, tuple):
        raw_tuple = cast(tuple[object, ...], value)
        items = [item for item in raw_tuple if isinstance(item, str)]
        if len(items) == len(raw_tuple):
            return tuple(items)
    return default


def _parse_key_takeaways(lines: list[str]) -> list[str]:
    takeaways = _parse_key_takeaways_section(lines, SOURCE_KEY_TAKEAWAYS_HEADING)
    if takeaways:
        return takeaways
    return _parse_key_takeaways_section(lines, "## 学习目标")


def _parse_key_takeaways_section(lines: list[str], heading: str) -> list[str]:
    takeaways: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == heading:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section:
            item = _parse_bullet_line(stripped)
            if item:
                takeaways.append(item)
    return takeaways


def _parse_glossary_terms(lines: list[str]) -> list[str]:
    terms: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == SOURCE_GLOSSARY_HEADING:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section:
            match = re.match(r"^-\s*\*\*(.+?)\*\*\s*:\s*(.+)$", stripped)
            if match:
                terms.append(match.group(1).strip())
                continue
            match = re.match(r"^-\s*([^:]+)\s*:\s*(.+)$", stripped)
            if match:
                terms.append(match.group(1).strip())
    return terms


def _parse_topic_blocks(lines: list[str]) -> list[RawTopicBlock]:
    blocks: list[RawTopicBlock] = []
    current: RawTopicBlock | None = None
    in_code = False

    def flush() -> None:
        nonlocal current
        if current:
            blocks.append(current)
            current = None

    for line in lines:
        stripped = line.rstrip("\n")
        if CODE_FENCE_RE.match(stripped.strip()):
            in_code = not in_code
        if not in_code:
            heading = TOPIC_HEADING_RE.match(stripped)
            if heading:
                flush()
                number = int(heading.group(1))
                title = _strip_timestamp(heading.group(2).strip())
                current = RawTopicBlock(number=number, title=title, lines=[])
                continue
            if current and (
                SECTION_HEADING_RE.match(stripped)
                or SUBSECTION_HEADING_RE.match(stripped)
                or stripped.startswith("# ")
            ):
                flush()
                continue

        if current:
            current.lines.append(stripped)

    flush()
    return blocks


def _fallback_topics_from_coverage(lines: list[str]) -> list[RawTopicBlock]:
    topics: list[RawTopicBlock] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == HEADING_COVERAGE_INDEX:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if not in_section:
            continue
        if not stripped:
            continue
        match = re.match(r"^(\d+)\.\s*(.+?)(?:\s+—.*)?$", stripped)
        if match:
            number = int(match.group(1))
            title = _strip_timestamp(match.group(2).strip())
            topics.append(RawTopicBlock(number=number, title=title, lines=[]))
            continue
        match = re.match(r"^-\s*(.+)$", stripped)
        if match:
            title = _strip_timestamp(match.group(1).strip())
            topics.append(RawTopicBlock(number=None, title=title, lines=[]))
    return topics


def _build_topics(raw_topics: list[RawTopicBlock]) -> list[TopicRecord]:
    topics: list[TopicRecord] = []
    for raw in raw_topics:
        key_points, pitfalls, expansions = _extract_topic_material(raw.lines)
        topics.append(
            TopicRecord(
                number=raw.number,
                title=raw.title,
                normalized=normalize_topic_title(raw.title),
                key_points=key_points,
                pitfalls=pitfalls,
                expansions=expansions,
            )
        )
    return topics


def _dedupe_topics(topics: list[TopicRecord]) -> list[TopicRecord]:
    deduped: list[TopicRecord] = []
    seen: dict[str, TopicRecord] = {}
    for topic in topics:
        if not topic.normalized:
            deduped.append(topic)
            continue
        if topic.normalized in seen:
            existing = seen[topic.normalized]
            merged = _merge_topic_records(existing, topic)
            seen[topic.normalized] = merged
            for idx, item in enumerate(deduped):
                if item.normalized == topic.normalized:
                    deduped[idx] = merged
                    break
            continue
        seen[topic.normalized] = topic
        deduped.append(topic)
    return deduped


def _merge_topic_records(a: TopicRecord, b: TopicRecord) -> TopicRecord:
    number = a.number if a.number is not None else b.number
    title = a.title or b.title
    key_points = a.key_points or b.key_points
    pitfalls = a.pitfalls or b.pitfalls
    expansions = a.expansions or b.expansions
    return TopicRecord(
        number=number,
        title=title,
        normalized=a.normalized or b.normalized,
        key_points=key_points,
        pitfalls=pitfalls,
        expansions=expansions,
    )


def _extract_topic_material(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    explanation_lines: list[str] = []
    example_lines: list[str] = []
    pitfall_lines: list[str] = []
    mode: str | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(stripped.startswith(marker) for marker in EXPLANATION_MARKERS):
            mode = "explanation"
            continue
        if any(stripped.startswith(marker) for marker in EXAMPLE_MARKERS):
            mode = "example"
            continue
        if stripped.startswith(PITFALL_MARKER):
            mode = "pitfall"
            continue
        if stripped.startswith("**"):
            mode = None
            continue
        if stripped.startswith("#### ") or stripped.startswith("### "):
            mode = None
            continue

        cleaned = _clean_block_line(stripped)
        if not cleaned:
            continue
        if mode == "explanation":
            explanation_lines.append(cleaned)
        elif mode == "example":
            example_lines.append(cleaned)
        elif mode == "pitfall":
            pitfall_lines.append(cleaned)

    key_points = _extract_key_points(explanation_lines, example_lines)
    expansions = _extract_expansions(explanation_lines, example_lines)
    pitfalls = pitfall_lines[:1]
    return key_points, pitfalls, expansions


def _extract_key_points(
    explanation_lines: list[str], example_lines: list[str]
) -> list[str]:
    candidates = _extract_bullet_lines(explanation_lines) or _extract_bullet_lines(
        example_lines
    )
    if candidates:
        return candidates[:2]

    text = " ".join(explanation_lines or example_lines).strip()
    sentences = _split_sentences(text)
    if sentences:
        return sentences[:2]
    return []


def _extract_expansions(
    explanation_lines: list[str], example_lines: list[str]
) -> list[str]:
    sentences = _split_sentences(" ".join(explanation_lines))
    example_sentences = _split_sentences(" ".join(example_lines))
    expansions: list[str] = []
    for sentence in sentences + example_sentences:
        if sentence and sentence not in expansions:
            expansions.append(sentence)
        if len(expansions) >= 4:
            break
    return expansions


def _build_mistakes_section(topics: list[TopicRecord]) -> list[str]:
    heading = DEFAULT_OUTPUT_STRUCTURE.required_headings[0]
    lines = [heading, ""]
    pitfalls: list[str] = []
    for topic in topics:
        for pitfall in topic.pitfalls:
            if pitfall not in pitfalls:
                pitfalls.append(pitfall)
    for pitfall in pitfalls:
        lines.append(f"- {pitfall}")
    if pitfalls:
        lines.append("")
    return lines


def _build_key_points_section(
    key_takeaways: list[str], topics: list[TopicRecord]
) -> list[str]:
    heading = DEFAULT_OUTPUT_STRUCTURE.required_headings[1]
    lines = [heading, ""]
    for takeaway in key_takeaways:
        lines.append(f"- {takeaway}")
    top_topics = topics[:8]
    for topic in top_topics:
        if not topic.key_points:
            continue
        key_point = "；".join(topic.key_points[:2])
        lines.append(f"- {topic.title}：{key_point}")
    lines.append("")
    return lines


def _map_key_takeaways_to_topics(
    takeaways: list[str],
    topics: list[TopicRecord],
    glossary_terms: list[str],
    rules: KeyTakeawayMappingRules,
) -> tuple[list[tuple[str, list[TopicRecord], str]], list[str]]:
    normalized_topics = [
        (topic, normalize_topic_title(topic.title)) for topic in topics
    ]
    normalized_terms = [normalize_takeaway(term) for term in glossary_terms]
    mapped_topics: set[str] = set()
    results: list[tuple[str, list[TopicRecord], str]] = []
    unmapped: list[str] = []

    for takeaway in takeaways:
        normalized_takeaway = normalize_takeaway(takeaway)
        matched_topic: TopicRecord | None = None
        strategy_used = ""

        for strategy in rules.match_order:
            if strategy == "topic_substring" and normalized_takeaway:
                for topic, normalized in normalized_topics:
                    if normalized and normalized in normalized_takeaway:
                        matched_topic = topic
                        strategy_used = strategy
                        break
            elif strategy == "glossary_term" and normalized_terms:
                for term in normalized_terms:
                    if term and term in normalized_takeaway:
                        for topic, normalized in normalized_topics:
                            if term in normalized:
                                matched_topic = topic
                                strategy_used = strategy
                                break
                    if matched_topic:
                        break
            elif strategy == "earliest_unmatched":
                for topic, normalized in normalized_topics:
                    if normalized and normalized not in mapped_topics:
                        matched_topic = topic
                        strategy_used = strategy
                        break

            if matched_topic:
                break

        if matched_topic:
            mapped_topics.add(matched_topic.normalized)
            results.append((takeaway, [matched_topic], strategy_used))
        else:
            unmapped.append(takeaway)

    return results, unmapped


def _build_mapping_section(
    mapping_results: list[tuple[str, list[TopicRecord], str]],
    rules: KeyTakeawayMappingRules,
) -> list[str]:
    lines = [rules.mapping_section_heading, ""]
    lines.append(MAPPING_TABLE_HEADER)
    lines.append(MAPPING_TABLE_SEPARATOR)
    for takeaway, topics, strategy in mapping_results:
        topic_names = ", ".join(_format_topic_label(topic) for topic in topics)
        lines.append(f"| {takeaway} | {topic_names} | {strategy} |")
    lines.append("")
    return lines


def _build_unmapped_section(
    unmapped_takeaways: list[str], rules: KeyTakeawayMappingRules
) -> list[str]:
    lines = [rules.unmapped_section_heading, ""]
    if not unmapped_takeaways:
        lines.append("- （无未映射结论）")
    else:
        for takeaway in unmapped_takeaways:
            lines.append(f"- {takeaway}")
    lines.append("")
    return lines


def _build_expanded_section(
    topics: list[TopicRecord],
    mapping_results: list[tuple[str, list[TopicRecord], str]],
    duration_seconds: float | None,
    budget_spec: BudgetSpec,
    *,
    prefix_lines: list[str],
    suffix_lines: list[str],
    topic_heading_prefix: str = "### ",
) -> list[str]:
    mapped_topics: list[TopicRecord] = []
    for _, mapped, _ in mapping_results:
        for topic in mapped:
            if topic not in mapped_topics:
                mapped_topics.append(topic)

    ordered_topics = mapped_topics + [
        topic for topic in topics if topic not in mapped_topics
    ]

    if duration_seconds is None:
        expanded_lines: list[str] = []
        for topic in ordered_topics:
            expanded_lines.extend(
                _build_topic_expansion_block(topic, heading_prefix=topic_heading_prefix)
            )
        return expanded_lines

    budget = budget_for_duration(duration_seconds, budget_spec)
    base_line_count = len(prefix_lines) + 2 + len(suffix_lines)
    available = max(budget.target_lines - base_line_count, 0)
    expanded_lines = []

    for topic in ordered_topics:
        block = _build_topic_expansion_block(topic, heading_prefix=topic_heading_prefix)
        if available <= 0:
            break
        if len(block) <= available:
            expanded_lines.extend(block)
            available -= len(block)
        else:
            break

    return expanded_lines


def _build_topic_expansion_block(
    topic: TopicRecord, *, heading_prefix: str = "### "
) -> list[str]:
    lines: list[str] = []
    heading = _format_topic_label(topic)
    lines.append(f"{heading_prefix}{heading}")
    if topic.key_points:
        key_points = _join_points(topic.key_points, 2)
        if key_points:
            lines.append(f"- 关键要点：{key_points}")

    if topic.pitfalls:
        lines.append(f"- 常见误区：{topic.pitfalls[0]}")

    for expansion in topic.expansions[:4]:
        lines.append(f"- 展开：{expansion}")
    lines.append("")
    return lines


def _build_coverage_section(topics: list[TopicRecord]) -> list[str]:
    return [f"- [x] {_format_topic_label(topic)}" for topic in topics]


def _format_topic_label(topic: TopicRecord) -> str:
    if topic.number is not None:
        return f"{topic.number}. {topic.title}"
    return topic.title


def _extract_bullet_lines(lines: list[str]) -> list[str]:
    bullets: list[str] = []
    for line in lines:
        item = _parse_bullet_line(line)
        if item:
            bullets.append(item)
    return bullets


def _parse_bullet_line(line: str) -> str | None:
    match = re.match(r"^[-*]\s+(.+)$", line)
    if match:
        return match.group(1).strip()
    match = re.match(r"^\d+\.\s+(.+)$", line)
    if match:
        return match.group(1).strip()
    return None


def _clean_block_line(line: str) -> str:
    cleaned = line.strip()
    if cleaned.startswith(">"):
        cleaned = cleaned.lstrip(">").strip()
    if cleaned.startswith("-"):
        cleaned = cleaned.lstrip("-").strip()
    return cleaned


def _split_sentences(text: str) -> list[str]:
    text = " ".join(text.strip().split())
    if not text:
        return []
    sentences: list[str] = []
    buffer: list[str] = []
    for ch in text:
        buffer.append(ch)
        if ch in "。！？.!?":
            sentences.append("".join(buffer).strip())
            buffer = []
    if buffer:
        sentences.append("".join(buffer).strip())
    return [sentence for sentence in sentences if sentence]


def _strip_timestamp(title: str) -> str:
    return TIMESTAMP_RE.sub("", title).strip()


def _join_points(points: list[str], max_points: int) -> str:
    trimmed = points[:max_points]
    return "；".join(trimmed) if trimmed else ""
