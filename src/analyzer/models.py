"""
数据模型定义模块

定义视频分析结果的数据结构，用于结构化存储和传递分析数据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class KnowledgeDocument:
    """精英知识笔记数据结构"""

    title: str
    """笔记标题"""

    one_sentence_summary: str
    """一句话核心总结"""

    key_takeaways: list[str]
    """关键结论/行动建议列表"""

    deep_dive: list[dict[str, str]]
    """
    深度解析列表，每项包含:
    - topic: 知识点主题
    - explanation: 原理解析
    - example: 具体示例
    - code: 代码演示 (可选)
    """

    glossary: dict[str, str]
    """关键术语表：{术语: 通俗定义}"""

    visual_schema: str
    """知识蓝图视觉架构描述 (Visual Schema)"""

    def to_markdown(self, image_path: str | None = None) -> str:
        """
        将知识笔记转换为 Markdown 格式

        Args:
             image_path: 知识蓝图图片的相对路径(可选)

        Returns:
            格式化的 Markdown 文档字符串
        """
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

        # 添加知识蓝图部分 (Visual Schema) -> 核心图解
        # 如果有图片路径，则直接展示图片；否则不展示此部分（或根据需求保留文本，但 User 要求隐藏文本）
        if image_path:
            lines.extend(
                [
                    "## 🖼️ 核心图解 (Visual Architecture)",
                    "",
                    f"![Core Architecture]({image_path})",
                    "",
                ]
            )

        # 添加深度解析
        lines.extend(
            [
                "## 🔍 深度解析 (Deep Dive)",
                "",
            ]
        )

        for idx, item in enumerate(self.deep_dive, 1):
            topic = item.get("topic", "未知主题")
            explanation = item.get("explanation", "")
            example = item.get("example", "")
            code = item.get("code", "")

            lines.append(f"### {idx}. {topic}")
            lines.append(f"**💡 原理解析**：")
            lines.append(f"{explanation}")
            lines.append("")
            if example:
                lines.append(f"**🌰 举个栗子**：")
                lines.append(f"> {example}")
            if code:
                lines.append("")
                lines.append(f"**💻 代码演示**：")
                lines.append(f"```python")  # 默认为 python，后续可根据内容自动识别或设为通用
                lines.append(f"{code}")
                lines.append(f"```")
            lines.append("")

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

    def to_markdown(self, image_path: str | None = None) -> str:
        """
        生成完整的 Markdown 文档

        Args:
            image_path: 知识蓝图图片的相对路径(可选)

        Returns:
            包含知识笔记和知识蓝图结构的完整 Markdown 文档
        """
        # KnowledgeDocument.to_markdown 已经包含了所有内容
        return self.knowledge_doc.to_markdown(image_path=image_path)

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

        # mind_map_structure 在新版中已被废弃，这里忽略它，visual_schema 初始化为空
        # 在后续步骤中由 gemini_visual_schema 填充
        
        knowledge_doc = KnowledgeDocument(
            title=response_data["title"],
            one_sentence_summary=response_data["one_sentence_summary"],
            key_takeaways=response_data["key_takeaways"],
            deep_dive=response_data["deep_dive"],
            glossary=response_data.get("glossary", optional_defaults["glossary"]),
            visual_schema=response_data.get("visual_schema", ""),  # 优先从 step 1 获取
        )

        return cls(
            video_path=video_path,
            knowledge_doc=knowledge_doc,
            metadata=metadata or {},
        )

