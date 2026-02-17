"""
JSON 修复功能单元测试

测试 ContentAnalyzer 中的 _sanitize_json_escapes, _close_truncated_json,
_truncate_to_last_complete_item, _try_repair_json 方法。
"""

import json
import pytest

from analyzer.content_analyzer import ContentAnalyzer


class TestSanitizeJsonEscapes:
    """测试非法转义序列修复"""

    def test_valid_json_passthrough(self):
        """合法 JSON 不应被修改"""
        text = '{"key": "value", "num": 123, "arr": [1, 2]}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        assert json.loads(result) == json.loads(text)

    def test_legal_escapes_preserved(self):
        """合法的 JSON 转义序列应保留"""
        text = r'{"msg": "line1\nline2\ttab\\backslash\"quote"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        parsed = json.loads(result)
        assert "line1\nline2\ttab\\backslash\"quote" == parsed["msg"]

    def test_fix_latex_frac(self):
        """\\frac 等 LaTeX 转义应被修复为 \\\\frac"""
        text = r'{"formula": "C_N = \frac{1}{N}"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        parsed = json.loads(result)
        assert "\\frac" in parsed["formula"]

    def test_fix_latex_sum(self):
        """\\sum 应被修复"""
        text = r'{"formula": "\sum_{k=0}^{N}"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        parsed = json.loads(result)
        assert "\\sum" in parsed["formula"]

    def test_fix_latex_ln(self):
        """\\l 不是合法转义, 应被修复 (\\ln 的 \\l 部分)"""
        text = r'{"formula": "2N \ln N"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        parsed = json.loads(result)
        assert "\\ln" in parsed["formula"]

    def test_fix_multiple_latex_in_one_value(self):
        """单个值中多个 LaTeX 转义均应被修复"""
        text = r'{"math": "$C_N = \frac{1}{N} \sum_{k=0}^{N-1} (C_k + C_{N-1-k})$"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        parsed = json.loads(result)
        assert "\\frac" in parsed["math"]
        assert "\\sum" in parsed["math"]

    def test_escapes_outside_strings_untouched(self):
        """字符串外部不应被修改"""
        text = '{"key": "value"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        assert result == text

    def test_unicode_escape_preserved(self):
        """\\uXXXX 应保留"""
        text = '{"emoji": "\\u0048ello"}'
        result = ContentAnalyzer._sanitize_json_escapes(text)
        parsed = json.loads(result)
        assert parsed["emoji"] == "Hello"


class TestCloseTruncatedJson:
    """测试截断 JSON 闭合"""

    def test_unclosed_string(self):
        """未闭合的字符串应被闭合"""
        text = '{"key": "value'
        result = ContentAnalyzer._close_truncated_json(text)
        assert result.endswith('}')
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_unclosed_array(self):
        """未闭合的数组应被闭合"""
        text = '{"arr": [1, 2, 3'
        result = ContentAnalyzer._close_truncated_json(text)
        parsed = json.loads(result)
        assert parsed["arr"] == [1, 2, 3]

    def test_unclosed_object(self):
        """未闭合的对象应被闭合"""
        text = '{"nested": {"a": 1'
        result = ContentAnalyzer._close_truncated_json(text)
        parsed = json.loads(result)
        assert parsed["nested"]["a"] == 1

    def test_trailing_comma_removed(self):
        """尾部逗号应被移除"""
        text = '{"arr": [1, 2,'
        result = ContentAnalyzer._close_truncated_json(text)
        parsed = json.loads(result)
        assert parsed["arr"] == [1, 2]


class TestTruncateToLastCompleteItem:
    """测试截断到最后完整项"""

    def test_truncate_array(self):
        """应截断到最后一个完整逗号处"""
        text = '{"arr": [1, 2, 3'
        result = ContentAnalyzer._truncate_to_last_complete_item(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["arr"] == [1, 2]

    def test_no_comma(self):
        """无逗号时返回 None"""
        text = '{"key": "value"}'
        # rfind(',') 会找不到合适的位置
        result = ContentAnalyzer._truncate_to_last_complete_item(text)
        # 无论返回 None 或有效结果，都不应崩溃
        assert result is None or json.loads(result) is not None


class TestTryRepairJson:
    """测试多轮 JSON 修复"""

    def test_valid_json_passthrough(self):
        """合法 JSON 直接通过"""
        text = '{"title": "test", "value": 42}'
        result = ContentAnalyzer._try_repair_json(text)
        assert result is not None
        assert result["title"] == "test"

    def test_fix_latex_escapes(self):
        """LaTeX 转义应被修复并解析成功"""
        text = r'{"explanation": "$C_N = \frac{1}{N} \sum_{k=0}^{N-1}$"}'
        result = ContentAnalyzer._try_repair_json(text)
        assert result is not None
        assert "\\frac" in result["explanation"]

    def test_fix_truncated_string(self):
        """未闭合字符串应被修复"""
        text = '{"title": "Hello World'
        result = ContentAnalyzer._try_repair_json(text)
        assert result is not None
        assert result["title"] == "Hello World"

    def test_fix_unclosed_brackets(self):
        """未闭合的括号应被修复"""
        text = '{"items": [{"name": "A"}, {"name": "B"'
        result = ContentAnalyzer._try_repair_json(text)
        assert result is not None
        assert len(result["items"]) >= 1

    def test_fix_trailing_comma(self):
        """尾部多余逗号应被处理"""
        text = '{"arr": [1, 2, 3,]}'
        result = ContentAnalyzer._try_repair_json(text)
        assert result is not None

    def test_combined_latex_and_truncation(self):
        """同时存在 LaTeX 转义和截断"""
        text = r'{"formula": "\frac{a}{b}", "items": [1, 2'
        result = ContentAnalyzer._try_repair_json(text)
        assert result is not None
        assert "\\frac" in result["formula"]

    def test_unfixable_returns_none(self):
        """完全无法修复的文本返回 None"""
        text = "This is not JSON at all - just plain text with no braces"
        result = ContentAnalyzer._try_repair_json(text)
        assert result is None

    def test_real_world_latex_case(self):
        """模拟日志中实际出现的 LaTeX 数学公式场景"""
        text = (
            '{"title": "Quicksort", '
            '"explanation": "平均比较次数。\\n'
            '$C_N = (N+1) + \\\\frac{1}{N} \\\\sum_{k=0}^{N-1} (C_k + C_{N-1-k})$"}'
        )
        # 注意: 原始 Gemini 响应中 \frac 是单个反斜杠
        # 在 Python 字符串中用 \\ 表示单个 \
        # 但实际 JSON 文件中它们是裸字节 \frac
        raw = text.replace("\\\\frac", "\\frac").replace("\\\\sum", "\\sum")
        result = ContentAnalyzer._try_repair_json(raw)
        assert result is not None
        assert result["title"] == "Quicksort"

    def test_nested_json_with_code_field(self):
        """包含 code 字段的嵌套 JSON（类似 Gemini 响应）"""
        text = json.dumps({
            "title": "Test",
            "deep_dive": [
                {
                    "topic": "Math",
                    "explanation": "Formula: \\frac{a}{b}",
                    "code": "int x = 1;",
                }
            ],
            "glossary": {"term": "def"},
        })
        # json.dumps 会将 \ 转为 \\, 模拟 Gemini 的裸 \ 通过替换
        raw = text.replace("\\\\frac", "\\frac")
        result = ContentAnalyzer._try_repair_json(raw)
        assert result is not None
        assert result["title"] == "Test"
