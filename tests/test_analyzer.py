"""
视频内容分析模块单元测试
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from analyzer.content_analyzer import ContentAnalyzer
from analyzer.models import KnowledgeDocument, AnalysisResult
from analyzer.prompt_loader import load_prompts, render_prompt
from utils.counter import APICounter


class TestPromptLoader:
    """测试 Prompt 加载器"""

    def test_load_prompts(self):
        prompts = load_prompts()

        assert isinstance(prompts, dict)
        assert "gemini_analysis" in prompts
        assert "system_role" in prompts["gemini_analysis"]
        assert "main_prompt" in prompts["gemini_analysis"]

    def test_render_prompt(self):
        template = "分析视频: {video_name}, 格式: {format}"
        result = render_prompt(template, video_name="test.mp4", format="mp4")

        assert result == "分析视频: test.mp4, 格式: mp4"


class TestKnowledgeDocument:
    """测试知识笔记数据模型"""

    def test_knowledge_document_creation(self):
        doc = KnowledgeDocument(
            title="测试标题",
            one_sentence_summary="一句话核心",
            key_takeaways=["结论1", "结论2"],
            deep_dive=[{"topic": "主题1", "explanation": "解释1", "example": "例子1"}],
            glossary={"术语1": "定义1"},
            visual_schemas=[],
        )

        assert doc.title == "测试标题"
        assert doc.one_sentence_summary == "一句话核心"
        assert len(doc.key_takeaways) == 2
        assert len(doc.deep_dive) == 1
        assert "术语1" in doc.glossary

    def test_to_markdown(self):
        doc = KnowledgeDocument(
            title="测试标题",
            one_sentence_summary="一句话核心",
            key_takeaways=["结论1", "结论2"],
            deep_dive=[{"topic": "主题1", "explanation": "解释1", "example": "例子1"}],
            glossary={"术语1": "定义1"},
            visual_schemas=[],
        )

        markdown = doc.to_markdown()

        assert "# 测试标题" in markdown
        assert "🎯 **一句话核心**" in markdown
        assert "一句话核心" in markdown
        assert "## 📝 关键结论 (Key Takeaways)" in markdown
        assert "- 结论1" in markdown
        assert "## 🔍 深度解析 (Deep Dive)" in markdown
        assert "#### 1. 主题1" in markdown
        assert "**💡 原理解析**：" in markdown
        assert "解释1" in markdown
        assert "**🌰 举个栗子**：" in markdown
        assert "例子1" in markdown
        assert "## 📖 关键术语表 (Glossary)" in markdown
        assert "**术语1**: 定义1" in markdown


class TestAnalysisResult:
    """测试分析结果数据模型"""

    def test_analysis_result_creation(self):
        doc = KnowledgeDocument(
            title="测试标题",
            one_sentence_summary="一句话核心",
            key_takeaways=["结论1"],
            deep_dive=[],
            glossary={"术语1": "定义1"},
            visual_schemas=[],
        )

        result = AnalysisResult(
            video_path="test.mp4",
            knowledge_doc=doc,
            metadata={"video_name": "test.mp4"},
        )

        assert result.video_path == "test.mp4"
        assert result.title == "测试标题"
        assert "术语1" in result.glossary

    def test_from_api_response(self):
        response_data = {
            "title": "API 测试标题",
            "one_sentence_summary": "API 测试核心",
            "key_takeaways": ["结论A", "结论B"],
            "deep_dive": [{"topic": "A", "explanation": "Exp A"}],
            "glossary": {"术语A": "定义A"},
            "visual_schema": "graph TD; A-->B",
        }

        result = AnalysisResult.from_api_response(
            video_path="api_test.mp4",
            response_data=response_data,
        )

        assert result.title == "API 测试标题"
        assert len(result.knowledge_doc.key_takeaways) == 2
        assert len(result.knowledge_doc.visual_schemas) == 1
        assert "graph TD" in result.knowledge_doc.visual_schemas[0].schema

    def test_from_api_response_missing_fields(self):
        response_data = {
            "title": "不完整的响应",
            "one_sentence_summary": "缺少字段",
        }

        with pytest.raises(ValueError, match="API 响应缺少必需字段"):
            AnalysisResult.from_api_response(
                video_path="test.mp4",
                response_data=response_data,
            )

    def test_to_markdown_with_mind_map(self):
        doc = KnowledgeDocument(
            title="完整测试",
            one_sentence_summary="完整核心",
            key_takeaways=["结论1"],
            deep_dive=[],
            glossary={"术语1": "定义1"},
            visual_schemas=[],
        )

        result = AnalysisResult(
            video_path="test.mp4",
            knowledge_doc=doc,
        )

        markdown = result.to_markdown()

        assert "## 🖼️ 核心图解 (Visual Architecture)" not in markdown


class TestContentAnalyzer:
    """测试内容分析器"""

    @pytest.fixture
    def mock_config(self):
        return {
            "proxy": {
                "base_url": "http://localhost:8000",
                "timeout": 60,
            },
            "analyzer": {
                "model": "gemini-2.5-flash",
                "temperature": 0.7,
                "max_output_tokens": 8192,
                "retry_times": 3,
                "timeout": 120,
            },
        }

    @pytest.fixture
    def mock_api_counter(self):
        return APICounter(max_calls=10, current_count=0)

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_init_with_fixed_api_key(self, mock_config, mock_api_counter, mock_logger):
        with patch("analyzer.content_analyzer.genai.Client") as mock_client_class:
            analyzer = ContentAnalyzer(
                config=mock_config,
                api_counter=mock_api_counter,
                logger=mock_logger,
                api_key="test_api_key",
            )

            mock_client_class.assert_called_once_with(
                api_key="test_api_key", http_options={"timeout": 600000}
            )
            assert analyzer._fixed_api_key == "test_api_key"
            assert analyzer._client is not None
            assert analyzer.model_name == "gemini-2.5-flash"
            assert analyzer.temperature == 0.7
            assert analyzer.max_output_tokens == 8192
            assert analyzer.retry_times == 3
            assert analyzer.timeout == 120

    def test_init_proxy_mode_no_configure(
        self, mock_config, mock_api_counter, mock_logger
    ):
        # 不需要 mock,因为没有固定 key 时不会创建 client
        analyzer = ContentAnalyzer(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
            api_key=None,
        )

        assert analyzer._fixed_api_key is None
        assert analyzer._client is None
        assert analyzer.proxy_base_url == "http://localhost:8000"

    # _allocate_key_from_pool tests removed as method is deleted

    def test_report_usage_to_pool(self, mock_config, mock_api_counter, mock_logger):
        analyzer = ContentAnalyzer(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )
        analyzer._allocated_key_id = "key_1"

        with patch("analyzer.content_analyzer.requests.post") as mock_post:
            analyzer._report_usage_to_pool()
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["json"] == {"key_id": "key_1"}

    def test_report_usage_skipped_without_allocation(
        self, mock_config, mock_api_counter, mock_logger
    ):
        analyzer = ContentAnalyzer(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        with patch("analyzer.content_analyzer.requests.post") as mock_post:
            analyzer._report_usage_to_pool()
            mock_post.assert_not_called()

    def test_report_error_to_pool(self, mock_config, mock_api_counter, mock_logger):
        analyzer = ContentAnalyzer(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )
        analyzer._allocated_key_id = "key_2"

        with patch("analyzer.content_analyzer.requests.post") as mock_post:
            analyzer._report_error_to_pool(is_rpd_limit=True)
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args
            assert call_kwargs[1]["json"] == {"key_id": "key_2", "is_rpd_limit": True}

    def test_generate_report(self, mock_config, mock_api_counter, mock_logger):
        analyzer = ContentAnalyzer(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        doc = KnowledgeDocument(
            title="测试文档",
            one_sentence_summary="测试核心",
            key_takeaways=["结论1"],
            deep_dive=[],
            glossary={"术语1": "定义1"},
            visual_schemas=[],
        )

        result = AnalysisResult(
            video_path="test.mp4",
            knowledge_doc=doc,
        )

        markdown = analyzer.generate_report(result)

        assert "# 测试文档" in markdown
        assert "🎯 **一句话核心**" in markdown
