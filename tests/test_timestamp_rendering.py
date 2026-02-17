"""
测试时间戳渲染功能

验证 KnowledgeDocument 在 section 标题中正确显示时间戳
"""

from analyzer.models import KnowledgeDocument


class TestTimestampRendering:
    """测试时间戳在 section 标题中的渲染"""

    def test_timestamp_string_format(self):
        """测试字符串格式的 timestamp 字段"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "topic": "知识点A",
                    "explanation": "解释内容",
                    "timestamp": "00:12:34-00:13:10",
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown()
        assert "#### 1. 知识点A (00:12:34–00:13:10)" in markdown

    def test_timestamp_numeric_start_end(self):
        """测试数值格式的 start_time/end_time 字段"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "topic": "知识点B",
                    "explanation": "解释内容",
                    "start_time": 754,
                    "end_time": 790,
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown()
        assert "#### 1. 知识点B (00:12:34–00:13:10)" in markdown

    def test_no_timestamp_backward_compatibility(self):
        """测试没有时间戳字段时的向后兼容性"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "topic": "知识点C",
                    "explanation": "解释内容",
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown()
        assert "#### 1. 知识点C\n" in markdown
        assert "(00:" not in markdown

    def test_timestamp_in_compact_mode(self):
        """测试 compact 模式下的时间戳渲染"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "chapter_title": "第一章",
                    "chapter_summary": "章节摘要",
                    "sections": [
                        {
                            "topic": "知识点D",
                            "explanation": "解释内容",
                            "timestamp": "00:05:20-00:06:15",
                        }
                    ],
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown(self_check_mode="default")
        assert "#### 1. 知识点D (00:05:20–00:06:15)" in markdown

    def test_timestamp_in_appendix_mode(self):
        """测试 appendix 模式下的时间戳渲染"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "chapter_title": "第一章",
                    "chapter_summary": "章节摘要",
                    "sections": [
                        {
                            "topic": "知识点E",
                            "explanation": "解释内容",
                            "code": "print('test')",
                            "start_time": 320,
                            "end_time": 375,
                        }
                    ],
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown(self_check_mode="default")
        appendix_section = markdown.split("## 📎 附录 (Appendix)")[1]
        assert "#### 1. 知识点E (00:05:20–00:06:15)" in appendix_section

    def test_timestamp_time_range_field(self):
        """测试 time_range 字段"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "topic": "知识点F",
                    "explanation": "解释内容",
                    "time_range": "01:23:45-01:24:30",
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown()
        assert "#### 1. 知识点F (01:23:45–01:24:30)" in markdown

    def test_timestamp_single_value(self):
        """测试只有开始时间的情况"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "topic": "知识点G",
                    "explanation": "解释内容",
                    "start_time": 900,
                }
            ],
            glossary={},
        )

        markdown = doc.to_markdown()
        assert "#### 1. 知识点G (00:15:00)" in markdown

    def test_multiple_sections_with_mixed_timestamps(self):
        """测试多个 section，部分有时间戳，部分没有"""
        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试摘要",
            key_takeaways=["要点1"],
            deep_dive=[
                {
                    "topic": "知识点H",
                    "explanation": "解释内容",
                    "timestamp": "00:01:00-00:02:00",
                },
                {
                    "topic": "知识点I",
                    "explanation": "解释内容",
                },
                {
                    "topic": "知识点J",
                    "explanation": "解释内容",
                    "start_time": 180,
                    "end_time": 240,
                },
            ],
            glossary={},
        )

        markdown = doc.to_markdown()
        assert "#### 1. 知识点H (00:01:00–00:02:00)" in markdown
        assert "#### 2. 知识点I\n" in markdown
        assert "#### 3. 知识点J (00:03:00–00:04:00)" in markdown
