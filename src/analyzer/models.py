"""
数据模型定义模块

定义视频分析结果的数据结构，用于结构化存储和传递分析数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import re
import warnings

from .validators import detect_stub_output, validate_markdown_structure


@dataclass
class VisualSchemaItem:
    """单张知识蓝图的 Visual Schema"""

    type: str
    """图片类型: overview / detail_flow / comparison"""

    description: str
    """图片描述（中文）"""

    schema: str
    """Visual Schema Markdown 字符串"""


@dataclass
class KnowledgeDocument:
    """精英知识笔记数据结构"""

    title: str
    """笔记标题"""

    one_sentence_summary: str
    """一句话核心总结"""

    key_takeaways: list[str]
    """关键结论/行动建议列表"""

    deep_dive: list[dict[str, Any]]
    """
    深度解析列表（分章节），每项包含:
    - chapter_title: 章节标题
    - chapter_summary: 章节概述
    - sections: 知识点列表，每个知识点包含:
        - topic: 知识点主题
        - explanation: 原理解析
        - example: 具体示例
        - code: 代码演示 (可选)
        - connections: 与其他知识点的关联说明列表
    """

    glossary: dict[str, str]
    """关键术语表：{术语: 通俗定义}"""

    visual_schemas: list[VisualSchemaItem] = field(default_factory=list)
    """知识蓝图 Visual Schema 列表（1-2 张）"""

    def to_markdown(
        self,
        image_paths: list[str] | None = None,
        self_check_mode: str = "static",
    ) -> str:
        """
        将知识笔记转换为 Markdown 格式

        Args:
             image_paths: 知识蓝图图片的相对路径列表(可选)
             self_check_mode: 自测题渲染模式(static/interactive/questions_only)

        Returns:
            格式化的 Markdown 文档字符串
        """
        self_check_mode = self._normalize_self_check_mode(self_check_mode)

        if self_check_mode == "default":
            return self._to_markdown_default(image_paths)

        if self_check_mode == "lecture":
            return self._to_markdown_lecture(image_paths)

        lines = [
            f"# {self.title}",
            "",
            "> 🎯 **一句话核心**",
            f"> {self.one_sentence_summary}",
            "",
            "## 📝 关键结论 (Key Takeaways)",
            "",
        ]

        # 添加关键结论
        for point in self.key_takeaways:
            lines.append(f"- {point}")
        lines.append("")

        # 添加知识蓝图图片（支持多张）
        if image_paths:
            lines.extend(
                [
                    "## 🖼️ 核心图解 (Visual Architecture)",
                    "",
                ]
            )
            for idx, img_path in enumerate(image_paths):
                desc = ""
                if idx < len(self.visual_schemas):
                    desc = self.visual_schemas[idx].description
                label = desc if desc else f"知识蓝图 {idx + 1}"
                lines.append(f"**{label}**")
                lines.append("")
                lines.append(f"![{label}]({img_path})")
                lines.append("")

        # 添加深度解析（分章节）
        lines.extend(
            [
                "## 🔍 深度解析 (Deep Dive)",
                "",
            ]
        )

        chapter_num = 0
        global_section_num = 0
        legacy_answers: list[str] = []

        for chapter in self.deep_dive:
            chapter_num += 1
            chapter_title = chapter.get("chapter_title", f"第{chapter_num}章")
            chapter_summary = chapter.get("chapter_summary", "")
            sections = chapter.get("sections", [])

            # 如果是旧格式（扁平 deep_dive，无 chapter_title），兼容处理
            if "topic" in chapter and "chapter_title" not in chapter:
                global_section_num += 1
                answers = self._render_section(
                    lines, global_section_num, chapter, self_check_mode
                )
                if self_check_mode == "static" and answers:
                    legacy_answers.extend(answers)
                continue

            lines.append(f"### 第{chapter_num}章：{chapter_title}")
            lines.append("")
            if chapter_summary:
                lines.append(f"> {chapter_summary}")
                lines.append("")

            chapter_answers: list[str] = []
            for section in sections:
                global_section_num += 1
                answers = self._render_section(
                    lines, global_section_num, section, self_check_mode
                )
                if self_check_mode == "static" and answers:
                    chapter_answers.extend(answers)

            if self_check_mode == "static" and chapter_answers:
                lines.append("#### 📌 本章自测答案")
                lines.append("")
                lines.extend(chapter_answers)

        if self_check_mode == "static" and legacy_answers:
            lines.append("### 📌 自测答案")
            lines.append("")
            lines.extend(legacy_answers)

        # 添加关键术语表
        if self.glossary:
            lines.extend(
                [
                    "## 📖 关键术语表 (Glossary)",
                    "",
                ]
            )
            for term, definition in self.glossary.items():
                lines.append(f"- **{term}**: {definition}")
            lines.append("")

        return "\n".join(lines)

    def _to_markdown_default(self, image_paths: list[str] | None) -> str:
        lines = [
            f"# {self.title}",
            "",
            "> 🎯 **一句话核心**",
            f"> {self.one_sentence_summary}",
            "",
            "## 📝 关键结论 (Key Takeaways)",
            "",
        ]

        for point in self.key_takeaways:
            lines.append(f"- {point}")
        lines.append("")

        if image_paths:
            lines.extend(
                [
                    "## 🖼️ 核心图解 (Visual Architecture)",
                    "",
                ]
            )
            for idx, img_path in enumerate(image_paths):
                desc = ""
                if idx < len(self.visual_schemas):
                    desc = self.visual_schemas[idx].description
                label = desc if desc else f"知识蓝图 {idx + 1}"
                lines.append(f"**{label}**")
                lines.append("")
                lines.append(f"![{label}]({img_path})")
                lines.append("")

        lines.extend(
            [
                "## 🔍 深度解析 (Deep Dive)",
                "",
            ]
        )

        chapter_num = 0
        global_section_num = 0
        appendix_lines: list[str] = []
        coverage_lines: list[str] = []

        chapters: list[dict[str, Any]] = []
        if any("chapter_title" in item for item in self.deep_dive):
            chapters = self.deep_dive
        else:
            chapters = [
                {
                    "chapter_title": "核心要点",
                    "chapter_summary": "",
                    "sections": self.deep_dive,
                }
            ]

        for chapter in chapters:
            chapter_num += 1
            chapter_title = chapter.get("chapter_title", f"第{chapter_num}章")
            chapter_summary = chapter.get("chapter_summary", "")
            sections = chapter.get("sections", [])

            lines.append(f"### 第{chapter_num}章：{chapter_title}")
            lines.append("")
            if chapter_summary:
                lines.append(f"> {chapter_summary}")
                lines.append("")

            coverage_lines.append(f"- 第{chapter_num}章：{chapter_title}")

            chapter_questions: list[dict[str, str]] = []

            appendix_lines.append(f"### 第{chapter_num}章：{chapter_title}")
            appendix_lines.append("")
            if chapter_summary:
                appendix_lines.append(f"> {chapter_summary}")
                appendix_lines.append("")

            for section in sections:
                global_section_num += 1
                self._render_section_compact(lines, global_section_num, section)
                appendix_lines.extend(
                    self._render_section_appendix(global_section_num, section)
                )

                topic = section.get("topic", "未知主题")
                coverage_lines.append(f"- {topic}")

                raw_self_check = section.get("self_check", [])
                if isinstance(raw_self_check, list):
                    for item in raw_self_check:
                        if isinstance(item, dict) and "q" in item and "a" in item:
                            chapter_questions.append(item)

            if chapter_questions:
                lines.append(f"### 📋 第{chapter_num}章自测")
                lines.append("")
                for idx, qa in enumerate(chapter_questions, 1):
                    label = f"Q{chapter_num}.{idx}"
                    question_text = str(qa["q"]).strip()
                    answer_text = str(qa["a"]).strip()
                    lines.append(f"- {label}：{question_text}")
                    lines.append(f"- {label} -> 答案：{answer_text}")
                lines.append("")

        lines.extend(
            [
                "## 📌 覆盖清单 (Coverage Index)",
                "",
            ]
        )
        lines.extend(coverage_lines)
        lines.append("")

        lines.extend(
            [
                "## 📎 附录 (Appendix)",
                "",
            ]
        )
        if appendix_lines:
            lines.extend(appendix_lines)
        else:
            lines.append("- （无附录内容）")
        lines.append("")

        if self.glossary:
            lines.extend(
                [
                    "## 📖 关键术语表 (Glossary)",
                    "",
                ]
            )
            for term, definition in self.glossary.items():
                lines.append(f"- **{term}**: {definition}")
            lines.append("")

        return "\n".join(lines)

    def _to_markdown_lecture(self, image_paths: list[str] | None) -> str:
        def cleaned(value: Any) -> str:
            return self._sanitize_lecture_text(value)

        def cleaned_main(value: Any) -> str:
            sanitized = cleaned(value)
            return sanitized.replace("```", "").strip()

        def normalize_code_block(value: Any) -> str:
            sanitized = cleaned(_normalize_field_value(value))
            if not sanitized:
                return ""
            return "\n".join(
                [line for line in sanitized.splitlines() if line.strip() != "```"]
            ).strip()

        def split_sentences(text: str) -> list[str]:
            if not text:
                return []
            parts = re.split(r"[。！？!?]", text)
            return [part.strip() for part in parts if part.strip()]

        def collect_unique_sentences(text: str, seen: set[str]) -> list[str]:
            sentences: list[str] = []
            for sentence in split_sentences(text):
                if sentence in seen:
                    continue
                seen.add(sentence)
                sentences.append(sentence)
            return sentences

        def collect_topics(sections: list[dict[str, Any]]) -> list[str]:
            topics: list[str] = []
            for section in sections:
                topic = cleaned_main(section.get("topic", ""))
                if topic:
                    topics.append(topic)
            return topics

        lines: list[str] = [f"# {self.title}", ""]
        chapters = self._normalize_chapters(self.deep_dive)

        lines.extend(["## 核心概念图谱", ""])
        thesis = cleaned_main(self.one_sentence_summary)
        if not thesis and self.key_takeaways:
            thesis = cleaned_main(self.key_takeaways[0])
        if thesis:
            lines.append(thesis)
        else:
            lines.append("本讲围绕核心概念与实践脉络展开。")
        lines.append("")

        if chapters:
            for chapter_num, chapter in enumerate(chapters, 1):
                chapter_title = cleaned_main(
                    chapter.get("chapter_title", f"第{chapter_num}章")
                )
                if not chapter_title:
                    chapter_title = f"第{chapter_num}章"
                lines.append(f"- 第{chapter_num}章：{chapter_title}")
                section_topics = collect_topics(chapter.get("sections", []))
                if section_topics:
                    for topic in section_topics:
                        lines.append(f"  - {topic}")
                else:
                    chapter_summary = cleaned_main(chapter.get("chapter_summary", ""))
                    if chapter_summary:
                        lines.append(f"  - {chapter_summary}")
        else:
            lines.append("- 本讲围绕关键主题逐步展开")
        lines.append("")

        lines.extend(["## 主题详解", ""])
        concept_index: list[str] = []
        appendix_code_blocks: list[tuple[str, str]] = []

        for chapter_num, chapter in enumerate(chapters, 1):
            chapter_title = cleaned_main(
                chapter.get("chapter_title", f"第{chapter_num}章")
            )
            if not chapter_title:
                chapter_title = f"第{chapter_num}章"
            chapter_summary = cleaned_main(chapter.get("chapter_summary", ""))
            sections = chapter.get("sections", [])

            lines.append(f"### 第{chapter_num}章：{chapter_title}")
            lines.append("")

            topics = collect_topics(sections)
            if chapter_summary:
                lines.append(chapter_summary)
            elif topics:
                lines.append(f"本章围绕 {'、'.join(topics)} 展开。")
            else:
                lines.append(f"本章梳理 {chapter_title} 的关键问题与应用场景。")
            lines.append("")

            if chapter_title:
                concept_index.append(chapter_title)
            concept_index.extend(topics)

            stitched_sentences: list[str] = []
            seen_sentences: set[str] = set()
            for section in sections:
                explanation = cleaned_main(
                    _normalize_field_value(section.get("explanation", ""))
                )
                example = cleaned_main(
                    _normalize_field_value(section.get("example", ""))
                )
                stitched_sentences.extend(
                    collect_unique_sentences(explanation, seen_sentences)
                )
                stitched_sentences.extend(
                    collect_unique_sentences(example, seen_sentences)
                )

                code = normalize_code_block(section.get("code", ""))
                if code:
                    label = cleaned_main(section.get("topic", "")) or chapter_title
                    appendix_code_blocks.append((label, code))

            if stitched_sentences:
                lines.append("内容串讲：")
                lines.append("")
                for sentence in stitched_sentences[:8]:
                    lines.append(f"- {sentence}")
                lines.append("")
            else:
                lines.append("内容串讲：")
                lines.append("")
                lines.append(f"- 本章聚焦 {chapter_title} 的核心逻辑与落地路径。")
                lines.append("")

        if not chapters:
            lines.append("本讲内容以关键概念串联，暂无章节拆分。")
            lines.append("")

        lines.extend(["## 实战与代码", ""])
        if appendix_code_blocks:
            for idx, (label, code) in enumerate(appendix_code_blocks[:2], 1):
                lines.append(f"### 示例 {idx}：{label}")
                lines.append("")
                code_lines = [line for line in code.splitlines() if line.strip()]
                if not code_lines:
                    lines.append("本示例仅给出思路，代码略。")
                    lines.append("")
                    continue

                lines.append("代码：")
                lines.append("")
                for line_num, line in enumerate(code_lines, 1):
                    lines.append(f"{line_num}. {line}")
                lines.append("")

                lines.append("逐行说明：")
                lines.append("")
                for line_num, line in enumerate(code_lines, 1):
                    lowered = line.lower()
                    if "fit" in lowered or "train" in lowered:
                        explanation = "执行训练或拟合步骤。"
                    elif "predict" in lowered:
                        explanation = "输出预测结果供后续评估。"
                    elif "print" in lowered or "log" in lowered:
                        explanation = "打印或记录关键结果。"
                    elif "load" in lowered or "read" in lowered:
                        explanation = "加载必要的数据或模型。"
                    else:
                        explanation = "完成关键计算或调用步骤。"
                    lines.append(f"{line_num}：{explanation}")
                lines.append("")
        else:
            lines.append("本讲无可复用代码片段")
            lines.append("")

        lines.extend(["## FAQ / 避坑指南", ""])
        pitfalls: list[str] = []
        for chapter in chapters:
            for section in chapter.get("sections", []):
                for mistake in self._coerce_list(section.get("common_mistakes", [])):
                    cleaned_mistake = cleaned_main(mistake)
                    if cleaned_mistake:
                        pitfalls.append(cleaned_mistake)
        deduped_pitfalls: list[str] = []
        seen_pitfalls = set()
        for pitfall in pitfalls:
            if pitfall not in seen_pitfalls:
                seen_pitfalls.add(pitfall)
                deduped_pitfalls.append(pitfall)
        if not deduped_pitfalls:
            deduped_pitfalls = [
                "只看训练效果，忽略验证集表现。",
                "关键假设未检验，导致结论偏差。",
            ]

        lines.append("常见坑：")
        lines.append("")
        for pitfall in deduped_pitfalls[:6]:
            lines.append(f"- {pitfall}")
        lines.append("")

        exercises: list[tuple[str, str]] = []
        for chapter in chapters:
            raw_questions = chapter.get("chapter_self_check", [])
            if isinstance(raw_questions, list):
                for item in raw_questions:
                    if not isinstance(item, dict):
                        continue
                    question = cleaned_main(item.get("q", ""))
                    answer = cleaned_main(item.get("a", ""))
                    if question and answer:
                        exercises.append((question, answer))

        if len(exercises) < 2:
            topic_pool: list[str] = []
            for chapter in chapters:
                topic_pool.extend(collect_topics(chapter.get("sections", [])))
            for topic in topic_pool[:4]:
                question = f"为什么 {topic} 在本讲中是关键环节？"
                answer = f"因为 {topic} 直接影响核心流程的效果与可解释性。"
                exercises.append((cleaned_main(question), cleaned_main(answer)))

        deduped_exercises: list[tuple[str, str]] = []
        seen_questions = set()
        for question, answer in exercises:
            if question in seen_questions:
                continue
            seen_questions.add(question)
            deduped_exercises.append((question, answer))

        selected_exercises = deduped_exercises[:4]
        if len(selected_exercises) < 2:
            fallback_question = "结合本讲内容，说明一个关键概念的应用场景。"
            fallback_answer = "可用于解决与核心概念相关的实际建模或决策问题。"
            selected_exercises.append((fallback_question, fallback_answer))
        selected_exercises = selected_exercises[:4]

        lines.append("练习与答解：")
        lines.append("")
        for idx, (question, _) in enumerate(selected_exercises, 1):
            lines.append(f"{idx}. {question}")
        for _, (_, answer) in enumerate(selected_exercises, 1):
            lines.append(f"答：{answer}")
        lines.append("")

        lines.extend(["## 📎 附录 (Appendix)", ""])
        lines.append("### 图解（知识蓝图）")
        lines.append("")
        if image_paths:
            for idx, img_path in enumerate(image_paths):
                desc = ""
                if idx < len(self.visual_schemas):
                    desc = cleaned(self.visual_schemas[idx].description)
                label = desc if desc else f"知识蓝图 {idx + 1}"
                lines.append(f"**{label}**")
                lines.append("")
                lines.append(f"![{label}]({img_path})")
                lines.append("")
        elif self.visual_schemas:
            for schema in self.visual_schemas:
                description = cleaned(schema.description)
                if description:
                    lines.append(f"- {description}")
                schema_text = cleaned(schema.schema)
                if schema_text:
                    lines.append("```")
                    lines.append(schema_text)
                    lines.append("```")
                    lines.append("")
        else:
            lines.append("- 暂无图解内容")
            lines.append("")

        lines.append("### 术语表（Glossary）")
        lines.append("")
        if self.glossary:
            for term, definition in self.glossary.items():
                cleaned_term = cleaned(term)
                cleaned_def = cleaned(definition)
                if cleaned_term and cleaned_def:
                    lines.append(f"- **{cleaned_term}**：{cleaned_def}")
        else:
            lines.append("- 暂无术语补充")
        lines.append("")

        lines.append("### 概念索引（Concept Index）")
        lines.append("")
        index_items: list[str] = []
        if self.key_takeaways:
            index_items.extend([cleaned(takeaway) for takeaway in self.key_takeaways])
        index_items.extend(concept_index)
        if self.glossary:
            index_items.extend([cleaned(key) for key in self.glossary])
        seen = set()
        for item in index_items:
            if item and item not in seen:
                seen.add(item)
                lines.append(f"- {item}")
        if not seen:
            lines.append("- 暂无概念索引")
        lines.append("")

        lines.append("### 代码与伪代码")
        lines.append("")
        if appendix_code_blocks:
            for label, code in appendix_code_blocks:
                if label:
                    lines.append(f"**{label}**")
                    lines.append("")
                lines.append("```python")
                lines.append(code)
                lines.append("```")
                lines.append("")
        else:
            lines.append("- 本讲无可复用代码片段")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _coerce_list(val: Any) -> list[Any]:
        """将值强制转换为列表（容错 Gemini 偶发类型偏差）"""
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val.strip():
            return [line.strip() for line in val.split("\n") if line.strip()]
        return []

    @staticmethod
    def _format_timestamp_for_display(section: dict[str, Any]) -> str:
        """
        从 section 提取并格式化时间戳用于显示

        Args:
            section: 包含时间戳信息的 section 字典

        Returns:
            格式化的时间戳字符串，如 "(00:12:34–00:13:10)" 或 "(00:12:34)"，
            如果没有时间戳则返回空字符串
        """
        import re

        def parse_time_value(value: Any) -> float | None:
            """解析时间值为秒数"""
            if value is None:
                return None
            if isinstance(value, (int, float)):
                num = float(value)
                if num > 1000:
                    return num / 1000.0
                return num
            raw = str(value).strip()
            if not raw:
                return None
            try:
                num = float(raw)
                if num > 1000:
                    return num / 1000.0
                return num
            except ValueError:
                pass
            if ":" in raw:
                parts = raw.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                elif len(parts) == 2:
                    hours = "0"
                    minutes, seconds = parts
                else:
                    return None
                try:
                    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                except ValueError:
                    return None
            return None

        def parse_time_range(value: Any) -> tuple[float | None, float | None]:
            """解析时间范围"""
            if isinstance(value, dict):
                start = parse_time_value(
                    value.get("start") or value.get("start_time") or value.get("begin")
                )
                end = parse_time_value(
                    value.get("end") or value.get("end_time") or value.get("finish")
                )
                return start, end
            if isinstance(value, str):
                matches = re.findall(r"\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2}", value)
                if not matches:
                    return parse_time_value(value), None
                if len(matches) == 1:
                    return parse_time_value(matches[0]), None
                start = parse_time_value(matches[0])
                end = parse_time_value(matches[1])
                return start, end
            start = parse_time_value(value)
            return start, None

        def format_seconds(seconds: float) -> str:
            """将秒数格式化为 HH:MM:SS"""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        # 尝试从多个可能的键提取时间戳
        start_time: float | None = None
        end_time: float | None = None

        # 优先检查 timestamp/time_range/timecode/time 字段
        for key in ("timestamp", "time_range", "timecode", "time"):
            if key in section:
                start_time, end_time = parse_time_range(section.get(key))
                if start_time is not None or end_time is not None:
                    break

        # 如果没找到，检查 start_time/end_time 字段
        if start_time is None and end_time is None:
            start_time = parse_time_value(
                section.get("start_time")
                or section.get("start")
                or section.get("begin")
            )
            end_time = parse_time_value(
                section.get("end_time") or section.get("end") or section.get("finish")
            )

        # 格式化输出
        if start_time is not None and end_time is not None:
            return f"({format_seconds(start_time)}–{format_seconds(end_time)})"
        elif start_time is not None:
            return f"({format_seconds(start_time)})"
        elif end_time is not None:
            return f"({format_seconds(end_time)})"

        return ""

    @staticmethod
    def _render_section(
        lines: list[str],
        num: int,
        section: dict[str, Any],
        self_check_mode: str,
    ) -> list[str]:
        """渲染单个知识点（v2 主动学习格式优先，v1 兜底）"""
        topic = section.get("topic", "未知主题")
        explanation = section.get("explanation", "")
        example = section.get("example", "")
        code = section.get("code", "")
        connections = section.get("connections", [])
        answer_lines: list[str] = []

        # 新字段（v2）
        challenge = KnowledgeDocument._coerce_list(section.get("challenge", []))
        common_mistakes = KnowledgeDocument._coerce_list(
            section.get("common_mistakes", [])
        )
        raw_self_check = section.get("self_check", [])
        self_check: list[dict[str, str]] = []
        if isinstance(raw_self_check, list):
            for item in raw_self_check:
                if isinstance(item, dict) and "q" in item and "a" in item:
                    self_check.append(item)

        use_v2 = bool(challenge or self_check or common_mistakes)

        timestamp_str = KnowledgeDocument._format_timestamp_for_display(section)
        if timestamp_str:
            lines.append(f"#### {num}. {topic} {timestamp_str}")
        else:
            lines.append(f"#### {num}. {topic}")

        if use_v2:
            # === v2: 主动学习格式 ===
            if challenge:
                lines.append("")
                lines.append("**🧩 挑战（先想 20 秒再往下看）**：")
                for c in challenge:
                    lines.append(f"- {c}")
                lines.append("")

            if code:
                lines.append("**💻 代码先行**：")
                lines.append("```python")
                lines.append(f"{code}")
                lines.append("```")
                lines.append("")

            if explanation:
                lines.append("**💡 原理拆解**：")
                lines.append(f"{explanation}")
                lines.append("")

            if example:
                lines.append("**🌰 自包含示例（输入 → 过程 → 输出）**：")
                lines.append(f"> {example}")
                lines.append("")

            if common_mistakes:
                lines.append("**⚠️ 常见误区**：")
                for m in common_mistakes:
                    lines.append(f"- {m}")
                lines.append("")

            if self_check:
                lines.append("**✅ 自测（做完再看答案）**：")

                question_lines: list[str] = []
                include_answers = self_check_mode in {"static", "interactive"}

                for idx, qa in enumerate(self_check, 1):
                    label = f"Q{num}.{idx}"
                    question_text = str(qa["q"]).strip()
                    question_lines.append(f"- {label}：{question_text}")

                    if include_answers:
                        answer_lines.append(f"- {label}（{topic}）：{question_text}")
                        answer_lines.append(f"  答案：{qa['a']}")
                        answer_lines.append("")

                lines.extend(question_lines)
                lines.append("")

                if self_check_mode == "interactive" and answer_lines:
                    lines.append("<details>")
                    lines.append("<summary>点击展开答案</summary>")
                    lines.append("")
                    lines.extend(answer_lines)
                    lines.append("</details>")
                    lines.append("")

            if connections:
                lines.append("**🔗 关联知识**：")
                for conn in connections:
                    lines.append(f"- {conn}")
                lines.append("")
        else:
            # === v1: 旧格式兜底（向后兼容） ===
            lines.append("**💡 原理解析**：")
            lines.append(f"{explanation}")
            lines.append("")
            if example:
                lines.append("**🌰 举个栗子**：")
                lines.append(f"> {example}")
                lines.append("")
            if code:
                lines.append("**💻 代码演示**：")
                lines.append("```python")
                lines.append(f"{code}")
                lines.append("```")
                lines.append("")
            if connections:
                lines.append("**🔗 关联知识**：")
                for conn in connections:
                    lines.append(f"- {conn}")
                lines.append("")

        if self_check_mode == "static":
            return answer_lines

        return []

    @staticmethod
    def _render_section_compact(
        lines: list[str],
        num: int,
        section: dict[str, Any],
    ) -> None:
        topic = section.get("topic", "未知主题")
        explanation = section.get("explanation", "")
        example = section.get("example", "")

        timestamp_str = KnowledgeDocument._format_timestamp_for_display(section)
        if timestamp_str:
            lines.append(f"#### {num}. {topic} {timestamp_str}")
        else:
            lines.append(f"#### {num}. {topic}")
        lines.append("")
        if explanation:
            lines.append("**💡 原理解析**：")
            lines.append(f"{explanation}")
            lines.append("")
        if example:
            lines.append("**🌰 示例**：")
            lines.append(f"> {example}")
            lines.append("")

    @staticmethod
    def _render_section_appendix(
        num: int,
        section: dict[str, Any],
    ) -> list[str]:
        lines: list[str] = []
        topic = section.get("topic", "未知主题")
        explanation = section.get("explanation", "")
        example = section.get("example", "")
        code = section.get("code", "")
        connections = section.get("connections", [])
        common_mistakes = KnowledgeDocument._coerce_list(
            section.get("common_mistakes", [])
        )

        timestamp_str = KnowledgeDocument._format_timestamp_for_display(section)
        if timestamp_str:
            lines.append(f"#### {num}. {topic} {timestamp_str}")
        else:
            lines.append(f"#### {num}. {topic}")
        lines.append("")

        if explanation:
            lines.append("**💡 原理拆解**：")
            lines.append(f"{explanation}")
            lines.append("")
        if example:
            lines.append("**🌰 自包含示例**：")
            lines.append(f"> {example}")
            lines.append("")
        if code:
            lines.append("**💻 完整代码**：")
            lines.append("```python")
            lines.append(f"{code}")
            lines.append("```")
            lines.append("")
        if common_mistakes:
            lines.append("**⚠️ 常见误区**：")
            for mistake in common_mistakes:
                lines.append(f"- {mistake}")
            lines.append("")
        if connections:
            lines.append("**🔗 关联知识**：")
            for conn in connections:
                lines.append(f"- {conn}")
            lines.append("")

        return lines

    @staticmethod
    def _normalize_self_check_mode(mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized in {
            "static",
            "interactive",
            "questions_only",
            "default",
            "lecture",
        }:
            return normalized
        return "static"

    @staticmethod
    def _normalize_chapters(deep_dive: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if any("chapter_title" in item for item in deep_dive):
            return deep_dive
        return [
            {
                "chapter_title": "核心要点",
                "chapter_summary": "",
                "sections": deep_dive,
            }
        ]

    @staticmethod
    def _sanitize_lecture_text(text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        patterns = [
            r"\$[^$]+\$",
            r"\\\([^)]+\\\)",
            r"\\\[[^\]]+\\\]",
            r"<details>",
            r"</details>",
            r"<summary>",
            r"</summary>",
            r"\b\d{1,2}:\d{2}\b",
            r"\(\d{1,2}:\d{2}[–—-]\d{1,2}:\d{2}\)",
        ]

        cleaned_lines: list[str] = []
        for line in text.splitlines():
            cleaned = line
            for pattern in patterns:
                cleaned = re.sub(pattern, "", cleaned)
            cleaned = " ".join(cleaned.split())
            if cleaned:
                cleaned_lines.append(cleaned)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def _render_chapter_exercises(
        chapter: dict[str, Any],
        chapter_title: str,
        topics: list[str],
    ) -> tuple[list[str], list[str]]:
        raw_questions = chapter.get("chapter_self_check", [])
        questions: list[str] = []
        answers: list[str] = []

        if isinstance(raw_questions, list):
            for item in raw_questions:
                if isinstance(item, dict) and "q" in item and "a" in item:
                    q = KnowledgeDocument._sanitize_lecture_text(item.get("q", ""))
                    a = KnowledgeDocument._sanitize_lecture_text(item.get("a", ""))
                    if q and a:
                        questions.append(q)
                        answers.append(a)

        fallback_pairs = KnowledgeDocument._generate_fallback_exercises(
            chapter_title, topics
        )
        while len(questions) < 3 and fallback_pairs:
            q, a = fallback_pairs.pop(0)
            questions.append(q)
            answers.append(a)

        if not questions:
            questions.append(f"用一句话概括 {chapter_title} 的核心主题。")
            answers.append(f"核心主题是：{chapter_title}。")

        question_lines = [f"{idx}. {q}" for idx, q in enumerate(questions, 1)]
        answer_lines = [f"{idx}. {a}" for idx, a in enumerate(answers, 1)]

        return question_lines, answer_lines

    @staticmethod
    def _generate_fallback_exercises(
        chapter_title: str,
        topics: list[str],
    ) -> list[tuple[str, str]]:
        topic_hint = "、".join(topics[:2]) if topics else chapter_title
        return [
            (
                "用一句话概括本章核心主题。",
                f"本章核心主题是：{chapter_title}。",
            ),
            (
                "列出本章涉及的两个关键概念。",
                f"关键概念包括：{topic_hint}。",
            ),
            (
                "给出一个本章的应用场景。",
                f"可用于与 {topic_hint} 相关的实际建模与评估任务。",
            ),
            (
                "说明一个常见误区并给出改进方向。",
                "常见误区是只看训练集指标，改进方向是加入验证集监控。",
            ),
        ]


def _format_bulleted_item(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    formatted = [f"- {lines[0]}"]
    for line in lines[1:]:
        formatted.append(f"  {line}")
    return formatted


def _format_ordered_list(items: list[str]) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        item_lines = [line for line in item.splitlines() if line.strip()]
        if not item_lines:
            continue
        lines.append(f"{index}. {item_lines[0]}")
        for line in item_lines[1:]:
            lines.append(f"   {line}")
    return lines


def _normalize_steps_value(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        steps: list[str] = []
        for item in value:
            normalized = _normalize_field_value(item)
            if normalized:
                steps.append(normalized)
        return steps
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    normalized = _normalize_field_value(value)
    return [normalized] if normalized else []


def _normalize_field_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        if not value:
            return ""
        io_keys = {"input", "steps", "output"}
        has_io_keys = any(key in value for key in io_keys)
        if has_io_keys:
            parts: list[str] = []
            input_value = _normalize_field_value(value.get("input"))
            if input_value:
                parts.append(f"输入：{input_value}")
            steps_value = _normalize_steps_value(value.get("steps"))
            if steps_value:
                parts.append("步骤：")
                parts.extend(_format_ordered_list(steps_value))
            output_value = _normalize_field_value(value.get("output"))
            if output_value:
                parts.append(f"输出：{output_value}")

            extra_keys = sorted(key for key in value.keys() if key not in io_keys)
            if extra_keys:
                parts.append("其他：")
                for key in extra_keys:
                    normalized = _normalize_field_value(value.get(key))
                    label = f"{key}: {normalized}" if normalized else f"{key}:"
                    parts.extend(_format_bulleted_item(label))
            return "\n".join(parts).strip()

        lines: list[str] = []
        for key in sorted(value.keys()):
            normalized = _normalize_field_value(value.get(key))
            label = f"{key}: {normalized}" if normalized else f"{key}:"
            lines.extend(_format_bulleted_item(label))
        return "\n".join(lines).strip()

    if isinstance(value, list):
        lines: list[str] = []
        for item in value:
            normalized = _normalize_field_value(item)
            if not normalized:
                continue
            item_lines = [line for line in normalized.splitlines() if line.strip()]
            if not item_lines:
                continue
            if all(line.startswith("- ") for line in item_lines):
                lines.extend(item_lines)
            else:
                lines.extend(_format_bulleted_item("\n".join(item_lines)))
        return "\n".join(lines).strip()

    return str(value).strip()


def _normalize_list_field(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            normalized = _normalize_field_value(item)
            if not normalized:
                continue
            lines = [line.strip() for line in normalized.splitlines() if line.strip()]
            if lines and all(line.startswith("- ") for line in lines):
                items.extend([line[2:].strip() for line in lines])
            else:
                items.append("\n".join(lines))
        return items
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    normalized = _normalize_field_value(value)
    return [normalized] if normalized else []


def _normalize_section_fields(section: dict[str, object]) -> dict[str, object]:
    normalized = dict(section)
    normalized["explanation"] = _normalize_field_value(section.get("explanation"))
    normalized["example"] = _normalize_field_value(section.get("example"))
    normalized["code"] = _normalize_field_value(section.get("code"))
    normalized["common_mistakes"] = _normalize_list_field(
        section.get("common_mistakes")
    )
    normalized["connections"] = _normalize_list_field(section.get("connections"))
    return normalized


def _normalize_deep_dive(deep_dive: object) -> list[dict[str, object]]:
    if not isinstance(deep_dive, list):
        return []
    normalized: list[dict[str, object]] = []
    for item in deep_dive:
        if not isinstance(item, dict):
            continue
        if "sections" in item and isinstance(item.get("sections"), list):
            chapter = dict(item)
            chapter_sections: list[dict[str, object]] = []
            for section in item.get("sections", []):
                if isinstance(section, dict):
                    chapter_sections.append(_normalize_section_fields(section))
            chapter["sections"] = chapter_sections
            normalized.append(chapter)
        else:
            normalized.append(_normalize_section_fields(item))
    return normalized


@dataclass
class AnalysisResult:
    """视频分析结果的完整数据结构"""

    video_path: str | Path
    """视频文件路径"""

    knowledge_doc: KnowledgeDocument
    """知识笔记对象"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """元数据（视频标题、时长等）"""

    @property
    def title(self) -> str:
        """获取文档标题"""
        return self.knowledge_doc.title

    @property
    def glossary(self) -> dict[str, str]:
        """获取术语表"""
        return self.knowledge_doc.glossary

    def to_markdown(
        self,
        image_paths: list[str] | None = None,
        self_check_mode: str = "static",
    ) -> str:
        """
        生成完整的 Markdown 文档

        Args:
            image_paths: 知识蓝图图片的相对路径列表(可选)
            self_check_mode: 自测题渲染模式(static/interactive/questions_only)

        Returns:
            包含知识笔记和知识蓝图结构的完整 Markdown 文档
        """
        markdown = self.knowledge_doc.to_markdown(
            image_paths=image_paths,
            self_check_mode=self_check_mode,
        )

        normalized_mode = (self_check_mode or "").strip().lower()
        if normalized_mode not in {
            "static",
            "interactive",
            "questions_only",
            "default",
        }:
            normalized_mode = "static"
        errors: list[str] = []

        if detect_stub_output(markdown):
            errors.append("检测到疑似占位/空内容输出")

        _, structure_errors = validate_markdown_structure(markdown, normalized_mode)
        errors.extend(structure_errors)

        if errors:
            message = "Markdown 校验失败: " + "; ".join(errors)
            if normalized_mode == "default":
                raise ValueError(message)
            warnings.warn(f"Markdown 校验警告(legacy 模式): {message}")

        return markdown

    @classmethod
    def from_api_response(
        cls,
        video_path: str | Path,
        response_data: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> AnalysisResult:
        """
        从 API 响应数据构建 AnalysisResult 对象

        Args:
            video_path: 视频文件路径
            response_data: API 返回的 JSON 数据
            metadata: 可选的元数据

        Returns:
            AnalysisResult 对象

        Raises:
            ValueError: 如果响应数据格式不正确
        """
        # 核心字段：缺失则无法构建有意义的文档，必须严格校验
        critical_fields = {
            "title",
            "one_sentence_summary",
            "key_takeaways",
            "deep_dive",
        }
        # 可选字段
        optional_defaults: dict[str, Any] = {
            "glossary": {},
        }

        missing_critical = critical_fields - response_data.keys()
        if missing_critical:
            raise ValueError(
                f"API 响应缺少必需字段: {', '.join(sorted(missing_critical))}"
            )

        # 解析 visual_schemas（支持新格式数组和旧格式单字符串）
        visual_schemas: list[VisualSchemaItem] = []
        raw_schemas = response_data.get("visual_schemas", [])
        if isinstance(raw_schemas, list) and len(raw_schemas) > 0:
            for item in raw_schemas:
                if isinstance(item, dict):
                    visual_schemas.append(
                        VisualSchemaItem(
                            type=item.get("type", "overview"),
                            description=item.get("description", ""),
                            schema=item.get("schema", ""),
                        )
                    )
                elif isinstance(item, str):
                    visual_schemas.append(
                        VisualSchemaItem(
                            type="overview",
                            description="",
                            schema=item,
                        )
                    )
        else:
            # 兼容旧格式: visual_schema 单字符串
            old_schema = response_data.get("visual_schema", "")
            if old_schema:
                visual_schemas.append(
                    VisualSchemaItem(
                        type="overview",
                        description="总览知识导图",
                        schema=old_schema,
                    )
                )

        knowledge_doc = KnowledgeDocument(
            title=response_data["title"],
            one_sentence_summary=response_data["one_sentence_summary"],
            key_takeaways=response_data["key_takeaways"],
            deep_dive=_normalize_deep_dive(response_data["deep_dive"]),
            glossary=response_data.get("glossary", optional_defaults["glossary"]),
            visual_schemas=visual_schemas,
        )

        return cls(
            video_path=video_path,
            knowledge_doc=knowledge_doc,
            metadata=metadata or {},
        )
