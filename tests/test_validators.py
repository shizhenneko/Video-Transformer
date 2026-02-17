# pyright: reportMissingImports=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false

import pytest

from analyzer.models import KnowledgeDocument
from analyzer.validators import (
    detect_stub_output,
    validate_knowledge_document,
    validate_markdown_structure,
)


@pytest.fixture
def minimal_knowledge_doc():
    return KnowledgeDocument(
        title="测试标题：验证器",
        one_sentence_summary="一句话总结。",
        key_takeaways=["关键结论1", "关键结论2"],
        deep_dive=[
            {
                "chapter_title": "测试章节",
                "chapter_summary": "章节摘要。",
                "sections": [
                    {
                        "topic": "测试主题",
                        "explanation": "测试解释内容。",
                        "example": "示例内容。",
                        "code": "print('ok')",
                        "challenge": ["挑战问题"],
                        "self_check": [{"q": "问题1?", "a": "答案1"}],
                    }
                ],
            }
        ],
        glossary={"术语": "定义"},
    )


def test_validate_default_mode_passes(minimal_knowledge_doc):
    markdown = minimal_knowledge_doc.to_markdown(self_check_mode="default")
    is_valid, errors = validate_markdown_structure(markdown, "default")
    assert is_valid, errors


def test_validate_legacy_mode_passes(minimal_knowledge_doc):
    markdown = minimal_knowledge_doc.to_markdown(self_check_mode="static")
    is_valid, errors = validate_markdown_structure(markdown, "static")
    assert is_valid, errors


def test_missing_required_headings_fail():
    markdown = "# 标题\n\n## 🔍 深度解析 (Deep Dive)\n"
    is_valid, errors = validate_markdown_structure(markdown, "default")
    assert not is_valid
    assert any("覆盖清单" in err for err in errors)
    assert any("附录" in err for err in errors)


def test_forbidden_patterns_fail():
    markdown = "# 标题\n\n**🧩 挑战（先想 20 秒再往下看）**：\n"
    is_valid, errors = validate_markdown_structure(markdown, "default")
    assert not is_valid
    assert any("禁用内容" in err for err in errors)


def test_code_fence_before_appendix_fails():
    markdown = (
        "# 标题\n\n"
        "```python\nprint('x')\n```\n\n"
        "## 📌 覆盖清单 (Coverage Index)\n\n- item\n\n"
        "## 📎 附录 (Appendix)\n"
    )
    is_valid, errors = validate_markdown_structure(markdown, "default")
    assert not is_valid
    assert any("代码围栏" in err for err in errors)


def test_detect_stub_output_final_report():
    assert detect_stub_output("final report") is True


def test_detect_stub_output_empty_sections():
    markdown = (
        "# 标题\n\n"
        "## 📝 关键结论 (Key Takeaways)\n\n"
        "## 📌 覆盖清单 (Coverage Index)\n\n"
        "## 📎 附录 (Appendix)\n"
    )
    assert detect_stub_output(markdown) is True


def test_validate_knowledge_document_detects_stub():
    doc = KnowledgeDocument(
        title="测试",
        one_sentence_summary="测试",
        key_takeaways=[],
        deep_dive=[],
        glossary={},
    )
    is_valid, errors = validate_knowledge_document(doc, "default")
    assert not is_valid
    assert any("占位" in err or "空内容" in err for err in errors)
