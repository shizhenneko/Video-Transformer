"""
知识文档结构契约测试 (TDD RED Phase)

验证 Core+Appendix 输出结构契约：
- 默认模式：紧凑核心内容 + 完整附录
- 遗留模式：保持向后兼容性
"""

import pytest
from analyzer.models import KnowledgeDocument


@pytest.fixture
def minimal_knowledge_doc():
    """最小化知识文档 fixture（用于测试结构契约）"""
    return KnowledgeDocument(
        title="测试标题：深度学习基础",
        one_sentence_summary="这是一句话总结，用于测试文档结构。",
        key_takeaways=[
            "关键结论1：深度学习需要大量数据",
            "关键结论2：反向传播是核心算法",
        ],
        deep_dive=[
            {
                "chapter_title": "神经网络基础",
                "chapter_summary": "本章介绍神经网络的基本概念和结构。",
                "sections": [
                    {
                        "topic": "感知机模型",
                        "explanation": "感知机是最简单的神经网络单元，由输入层和输出层组成。",
                        "example": "输入：[0.5, 0.3] → 权重：[0.2, 0.8] → 输出：0.5*0.2 + 0.3*0.8 = 0.34",
                        "code": "def perceptron(x, w):\n    return sum(xi * wi for xi, wi in zip(x, w))",
                        "challenge": [
                            "如果输入是负数，感知机会如何处理？",
                            "为什么需要激活函数？",
                        ],
                        "self_check": [
                            {"q": "感知机有几层？", "a": "两层：输入层和输出层"},
                            {"q": "权重的作用是什么？", "a": "控制输入特征的重要性"},
                        ],
                    },
                    {
                        "topic": "激活函数",
                        "explanation": "激活函数引入非线性，使神经网络能够学习复杂模式。",
                        "example": "ReLU(x) = max(0, x)，当 x=-2 时输出 0，当 x=3 时输出 3。",
                        "code": "def relu(x):\n    return max(0, x)",
                        "challenge": ["为什么线性激活函数无法解决复杂问题？"],
                        "self_check": [
                            {"q": "ReLU 的全称是什么？", "a": "Rectified Linear Unit"},
                        ],
                    },
                ],
            },
            {
                "chapter_title": "反向传播算法",
                "chapter_summary": "本章讲解如何通过反向传播训练神经网络。",
                "sections": [
                    {
                        "topic": "梯度下降",
                        "explanation": "梯度下降通过计算损失函数的梯度来更新权重。",
                        "example": "权重更新：w_new = w_old - learning_rate * gradient",
                        "code": "w = w - 0.01 * gradient",
                        "self_check": [
                            {
                                "q": "学习率过大会怎样？",
                                "a": "可能导致训练不稳定或发散",
                            },
                        ],
                    },
                ],
            },
        ],
        glossary={
            "感知机": "最简单的神经网络单元，只有输入层和输出层",
            "激活函数": "引入非线性的数学函数，如 ReLU、Sigmoid",
            "梯度下降": "通过计算梯度优化模型参数的算法",
        },
    )


class TestDefaultModeStructure:
    """测试默认模式（Core+Appendix）的结构契约"""

    def test_required_headings_exist_and_ordered(self, minimal_knowledge_doc):
        """测试必需的标题存在且顺序正确"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        # 必需的标题（按顺序）
        required_headings = [
            "# 测试标题：深度学习基础",
            "## 📝 关键结论 (Key Takeaways)",
            "## 🔍 深度解析 (Deep Dive)",
            "## 📌 覆盖清单 (Coverage Index)",
            "## 📎 附录 (Appendix)",
            "## 📖 关键术语表 (Glossary)",
        ]

        for heading in required_headings:
            assert heading in markdown, f"缺少必需标题: {heading}"

        # 验证标题顺序
        positions = [markdown.find(h) for h in required_headings]
        assert positions == sorted(positions), "标题顺序不正确"

    def test_no_challenge_blocks_in_default_mode(self, minimal_knowledge_doc):
        """测试默认模式下没有挑战块"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        assert "**🧩 挑战" not in markdown, "默认模式不应包含挑战块"
        assert "**🧩" not in markdown, "默认模式不应包含任何挑战标记"

    def test_no_per_section_self_check_in_default_mode(self, minimal_knowledge_doc):
        """测试默认模式下没有每节自测块"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        # 不应该有每节的自测块
        assert "**✅ 自测（做完再看答案）**" not in markdown, (
            "默认模式不应包含每节自测块"
        )

    def test_chapter_level_self_check_exists(self, minimal_knowledge_doc):
        """测试章节级自测存在（新格式：### 📋 第1章自测）"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        assert "### 📋 第1章自测" in markdown, (
            "默认模式应包含章节级自测标题（格式：### 📋 第1章自测），"
            "而非当前的 #### 📌 本章自测答案"
        )

    def test_self_check_answers_immediately_after_questions(
        self, minimal_knowledge_doc
    ):
        """测试自测答案紧跟在问题后面"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        # 查找自测问题和答案的位置
        lines = markdown.split("\n")

        # 应该能找到问题和答案在相邻区域
        found_question = False
        found_answer = False
        max_gap = 10  # 问题和答案之间最多间隔10行

        for i, line in enumerate(lines):
            if "Q1.1" in line or "Q1." in line:
                found_question = True
                # 在接下来的几行内应该能找到答案
                for j in range(i, min(i + max_gap, len(lines))):
                    if "答案" in lines[j] or "A1.1" in lines[j]:
                        found_answer = True
                        break
                break

        assert found_question, "应包含自测问题"
        assert found_answer, "答案应紧跟在问题后面"

    def test_answers_do_not_repeat_question_stems(self, minimal_knowledge_doc):
        """测试答案不重复问题题干"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        lines = markdown.split("\n")

        # 查找答案行
        for line in lines:
            if "答案：" in line or line.strip().startswith("A"):
                # 答案不应该包含完整的问题题干
                # 例如：不应该是 "答案：感知机有几层？两层"
                # 而应该是 "答案：两层：输入层和输出层"

                # 检查答案中是否包含问号（表示重复了问题）
                answer_part = line.split("答案：")[-1] if "答案：" in line else line
                assert "？" not in answer_part, f"答案不应重复问题题干: {line}"

    def test_no_code_fences_in_main_content(self, minimal_knowledge_doc):
        """测试主内容区没有代码围栏（只在附录中）"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        # 分割文档：找到附录的位置
        appendix_start = markdown.find("## 📎 附录 (Appendix)")

        if appendix_start == -1:
            pytest.fail("未找到附录部分")

        main_content = markdown[:appendix_start]
        appendix_content = markdown[appendix_start:]

        # 主内容不应包含代码围栏
        assert "```" not in main_content, "主内容区不应包含代码围栏"

        # 附录应该包含代码
        assert "```" in appendix_content, "附录应包含代码示例"

    def test_coverage_index_generated_from_deep_dive(self, minimal_knowledge_doc):
        """测试覆盖清单从 deep_dive 生成"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        # 覆盖清单应该包含所有章节和主题
        assert "## 📌 覆盖清单 (Coverage Index)" in markdown

        # 应该列出所有主题
        assert "感知机模型" in markdown
        assert "激活函数" in markdown
        assert "梯度下降" in markdown

        # 覆盖清单应该在深度解析之后、附录之前
        deep_dive_pos = markdown.find("## 🔍 深度解析 (Deep Dive)")
        coverage_pos = markdown.find("## 📌 覆盖清单 (Coverage Index)")
        appendix_pos = markdown.find("## 📎 附录 (Appendix)")

        assert deep_dive_pos < coverage_pos < appendix_pos, "覆盖清单位置不正确"


class TestLegacyModeCompatibility:
    """测试遗留模式的向后兼容性"""

    def test_legacy_mode_has_per_section_challenges(self, minimal_knowledge_doc):
        """测试遗留模式保留每节挑战块"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="static")

        # 遗留模式应该包含挑战块
        assert "**🧩 挑战" in markdown, "遗留模式应包含挑战块"

    def test_legacy_mode_has_per_section_self_check(self, minimal_knowledge_doc):
        """测试遗留模式保留每节自测块"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="static")

        # 遗留模式应该包含每节自测
        assert "**✅ 自测" in markdown, "遗留模式应包含每节自测块"

    def test_legacy_mode_has_code_in_main_content(self, minimal_knowledge_doc):
        """测试遗留模式在主内容中包含代码"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="static")

        # 遗留模式应该在主内容中包含代码
        deep_dive_start = markdown.find("## 🔍 深度解析 (Deep Dive)")
        glossary_start = markdown.find("## 📖 关键术语表 (Glossary)")

        if deep_dive_start == -1 or glossary_start == -1:
            pytest.fail("未找到深度解析或术语表部分")

        main_content = markdown[deep_dive_start:glossary_start]

        assert "```" in main_content, "遗留模式应在主内容中包含代码围栏"

    def test_legacy_mode_no_coverage_index(self, minimal_knowledge_doc):
        """测试遗留模式没有覆盖清单"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="static")

        # 遗留模式不应该有覆盖清单
        assert "## 📌 覆盖清单 (Coverage Index)" not in markdown, (
            "遗留模式不应包含覆盖清单"
        )

    def test_legacy_mode_no_appendix(self, minimal_knowledge_doc):
        """测试遗留模式没有附录"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="static")

        # 遗留模式不应该有附录
        assert "## 📎 附录 (Appendix)" not in markdown, "遗留模式不应包含附录"


class TestAnswerFormat:
    """测试答案格式规范"""

    def test_answer_format_concise(self, minimal_knowledge_doc):
        """测试答案格式简洁（不重复问题）"""
        markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")

        lines = markdown.split("\n")

        for i, line in enumerate(lines):
            if "答案：" in line:
                # 答案应该直接给出，不重复问题
                # 正确格式: "答案：两层：输入层和输出层"
                # 错误格式: "答案：感知机有几层？答案是两层"

                answer_text = line.split("答案：", 1)[-1].strip()

                # 答案不应该以问号开头或包含问号
                assert not answer_text.startswith("感知机"), (
                    f"答案重复了问题主语: {line}"
                )
                assert "？" not in answer_text, (
                    f"答案包含问号（可能重复了问题）: {line}"
                )


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_glossary_handled(self):
        """测试空术语表的处理"""
        doc = KnowledgeDocument(
            title="测试",
            one_sentence_summary="测试",
            key_takeaways=["测试"],
            deep_dive=[
                {
                    "chapter_title": "测试章节",
                    "sections": [
                        {
                            "topic": "测试主题",
                            "explanation": "测试解释",
                        }
                    ],
                }
            ],
            glossary={},  # 空术语表
        )

        markdown = doc.to_markdown(self_check_mode="default")

        # 空术语表不应该渲染术语表部分
        # 或者渲染但为空
        if "## 📖 关键术语表 (Glossary)" in markdown:
            # 如果渲染了，应该没有术语条目
            glossary_section = markdown.split("## 📖 关键术语表 (Glossary)")[-1]
            next_section = (
                glossary_section.split("##")[0]
                if "##" in glossary_section
                else glossary_section
            )
            assert "**" not in next_section or next_section.strip() == "", (
                "空术语表不应有内容"
            )

    def test_section_without_self_check(self):
        """测试没有自测题的章节"""
        doc = KnowledgeDocument(
            title="测试",
            one_sentence_summary="测试",
            key_takeaways=["测试"],
            deep_dive=[
                {
                    "chapter_title": "测试章节",
                    "sections": [
                        {
                            "topic": "测试主题",
                            "explanation": "测试解释",
                            # 没有 self_check 字段
                        }
                    ],
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown(self_check_mode="default")

        # 应该能正常渲染，不会崩溃
        assert "测试主题" in markdown
