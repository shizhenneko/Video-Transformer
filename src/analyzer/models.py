"""
数据模型定义模块

定义视频分析结果的数据结构，用于结构化存储和传递分析数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    @staticmethod
    def _coerce_list(val: Any) -> list[Any]:
        """将值强制转换为列表（容错 Gemini 偶发类型偏差）"""
        if isinstance(val, list):
            return val
        if isinstance(val, str) and val.strip():
            return [line.strip() for line in val.split("\n") if line.strip()]
        return []

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
    def _normalize_self_check_mode(mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized in {"static", "interactive", "questions_only"}:
            return normalized
        return "static"


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
        return self.knowledge_doc.to_markdown(
            image_paths=image_paths,
            self_check_mode=self_check_mode,
        )

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
            deep_dive=response_data["deep_dive"],
            glossary=response_data.get("glossary", optional_defaults["glossary"]),
            visual_schemas=visual_schemas,
        )

        return cls(
            video_path=video_path,
            knowledge_doc=knowledge_doc,
            metadata=metadata or {},
        )
