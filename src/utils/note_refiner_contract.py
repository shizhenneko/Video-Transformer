from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
import re
from collections.abc import Iterable

HEADING_MISTAKES = "## ⚠️ 易错点总结"
HEADING_KEY_POINTS = "## ⭐ 知识重点"
HEADING_EXPANDED = "## 📚 重点展开"
HEADING_COVERAGE_INDEX = "## 📌 覆盖清单 (Coverage Index)"

REQUIRED_OUTPUT_HEADINGS: tuple[str, ...] = (
    HEADING_MISTAKES,
    HEADING_KEY_POINTS,
    HEADING_EXPANDED,
    HEADING_COVERAGE_INDEX,
)

SOURCE_KEY_TAKEAWAYS_HEADING = "## 📝 关键结论 (Key Takeaways)"
SOURCE_GLOSSARY_HEADING = "## 📖 关键术语表 (Glossary)"

MAPPING_SECTION_HEADING = "## 🔗 关键结论映射 (Key Takeaway Mapping)"
UNMAPPED_TAKEAWAYS_HEADING = "## Unmapped Takeaways"
MAPPING_TABLE_HEADER = "| Takeaway | Topics | Match Strategy |"
MAPPING_TABLE_SEPARATOR = "| --- | --- | --- |"
MAPPING_JSON_FENCE = "```json"

BUDGET_WARNING_TEMPLATE = "<!-- BUDGET_EXCEEDED: actual={actual}, target={target} -->"


@dataclass(frozen=True)
class BudgetSpec:
    lines_per_hour: int = 400
    min_cap: int = 220
    max_cap: int = 900
    tolerance_ratio: float = 0.10

    def target_lines(self, duration_seconds: float) -> int:
        raw_target = ceil(duration_seconds / 3600 * self.lines_per_hour)
        return max(self.min_cap, min(self.max_cap, raw_target))

    def tolerance_range(self, target_lines: int) -> tuple[int, int]:
        lower = ceil(target_lines * (1 - self.tolerance_ratio))
        upper = floor(target_lines * (1 + self.tolerance_ratio))
        return lower, upper


@dataclass(frozen=True)
class BudgetResult:
    target_lines: int
    min_lines: int
    max_lines: int


def budget_for_duration(
    duration_seconds: float, spec: BudgetSpec | None = None
) -> BudgetResult:
    spec = spec or BudgetSpec()
    target = spec.target_lines(duration_seconds)
    min_lines, max_lines = spec.tolerance_range(target)
    return BudgetResult(target_lines=target, min_lines=min_lines, max_lines=max_lines)


def format_budget_warning(actual_lines: int, target_lines: int) -> str:
    return BUDGET_WARNING_TEMPLATE.format(actual=actual_lines, target=target_lines)


@dataclass(frozen=True)
class CoveragePolicy:
    require_all_topics: bool = True
    budget_is_soft_constraint: bool = True
    warn_on_budget_exceed: bool = True
    budget_warning_template: str = BUDGET_WARNING_TEMPLATE


@dataclass(frozen=True)
class CodeBudgetPolicy:
    exclude_code_from_budget: bool = False


CODE_FENCE_RE = re.compile(r"^```")


def count_budget_lines(text: str, exclude_code_from_budget: bool = False) -> int:
    lines = text.splitlines()
    if not exclude_code_from_budget:
        return len(lines)

    count = 0
    in_code_block = False
    for line in lines:
        if CODE_FENCE_RE.match(line.strip()):
            in_code_block = not in_code_block
            continue
        if not in_code_block:
            count += 1
    return count


@dataclass(frozen=True)
class KeyTakeawayMappingRules:
    match_order: tuple[str, ...] = (
        "topic_substring",
        "glossary_term",
        "earliest_unmatched",
    )
    mapping_output_formats: tuple[str, ...] = ("markdown_table", "json_fence")
    mapping_section_heading: str = MAPPING_SECTION_HEADING
    unmapped_section_heading: str = UNMAPPED_TAKEAWAYS_HEADING


_NORMALIZE_RE = re.compile(
    r"[\s\-—_·`~!@#$%^&*()=+\[\]{};:'\",.<>/?\\|，。！？：；（）【】《》“”‘’、]",
    re.UNICODE,
)


def normalize_topic_title(title: str) -> str:
    return _NORMALIZE_RE.sub("", title.strip().lower())


def normalize_takeaway(text: str) -> str:
    return _NORMALIZE_RE.sub("", text.strip().lower())


def build_coverage_index_lines(topics: Iterable[str]) -> list[str]:
    return [f"- {topic}" for topic in topics]


@dataclass(frozen=True)
class OutputStructure:
    required_headings: tuple[str, ...] = REQUIRED_OUTPUT_HEADINGS
    coverage_heading: str = HEADING_COVERAGE_INDEX
    mapping_heading: str = MAPPING_SECTION_HEADING
    unmapped_heading: str = UNMAPPED_TAKEAWAYS_HEADING


DEFAULT_BUDGET_SPEC = BudgetSpec()
DEFAULT_COVERAGE_POLICY = CoveragePolicy()
DEFAULT_CODE_BUDGET_POLICY = CodeBudgetPolicy()
DEFAULT_MAPPING_RULES = KeyTakeawayMappingRules()
DEFAULT_OUTPUT_STRUCTURE = OutputStructure()
