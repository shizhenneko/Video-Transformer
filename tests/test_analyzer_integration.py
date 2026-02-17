"""
视频内容分析模块集成测试

注意：此测试需要：
1. 本地代理号池服务运行在 localhost:8000（提供 /sdk/allocate-key 端点）
2. 测试视频文件
"""

import os
from pathlib import Path

import pytest
import requests

from analyzer import AnalysisResult, ContentAnalyzer
from utils.config import load_config
from utils.counter import APICounter
from utils.logger import setup_logging
from utils.proxy import verify_proxy_connection, verify_sdk_endpoint


PROXY_BASE_URL = "http://localhost:8000"


def check_proxy_available():
    return verify_proxy_connection(PROXY_BASE_URL, timeout=2)


def check_sdk_endpoint_available():
    return verify_sdk_endpoint(PROXY_BASE_URL, timeout=2)


skip_if_no_proxy = pytest.mark.skipif(
    not check_proxy_available(),
    reason="需要本地代理服务运行在 localhost:8000",
)

skip_if_no_sdk_endpoint = pytest.mark.skipif(
    not check_sdk_endpoint_available(),
    reason="代理服务缺少 /sdk/allocate-key 端点",
)


class TestContentAnalyzerIntegration:
    """内容分析器集成测试（通过号池分配 Key）"""

    @pytest.fixture
    def config(self):
        return load_config()

    @pytest.fixture
    def api_counter(self):
        return APICounter(max_calls=10, current_count=0)

    @pytest.fixture
    def logger(self, tmp_path):
        return setup_logging(log_dir=tmp_path, log_name="test_analyzer.log")

    @pytest.fixture
    def analyzer(self, config, api_counter, logger):
        return ContentAnalyzer(
            config=config,
            api_counter=api_counter,
            logger=logger,
            api_key=None,
        )

    @pytest.fixture
    def test_video_path(self):
        project_root = Path(__file__).resolve().parents[1]
        temp_videos = project_root / "data" / "temp" / "videos"

        if temp_videos.exists():
            video_files = list(temp_videos.glob("*.mp4"))
            if video_files:
                return video_files[0]

        pytest.skip("未找到测试视频文件")

    @skip_if_no_proxy
    @skip_if_no_sdk_endpoint
    def test_video_analysis_end_to_end(self, analyzer, test_video_path, api_counter):
        initial_count = api_counter.current_count

        result = analyzer.analyze_video(test_video_path)

        assert isinstance(result, AnalysisResult)
        assert result.video_path == test_video_path
        assert result.title, "文档标题不能为空"
        assert result.knowledge_doc.one_sentence_summary, "一句话核心不能为空"
        assert len(result.knowledge_doc.key_takeaways) >= 3, "关键结论应至少有 3 个"
        assert len(result.glossary) >= 1, "术语表应至少有 1 个术语"
        assert result.knowledge_doc.visual_schemas, "知识蓝图结构不能为空"
        assert api_counter.current_count == initial_count + 1, "应该只调用 1 次 API"

    @skip_if_no_proxy
    @skip_if_no_sdk_endpoint
    def test_document_quality(self, analyzer, test_video_path):
        result = analyzer.analyze_video(test_video_path)

        assert result.knowledge_doc.title, "标题不能为空"
        assert len(result.knowledge_doc.one_sentence_summary) >= 20, (
            "一句话核心应至少 20 字"
        )

        for takeaway in result.knowledge_doc.key_takeaways:
            assert isinstance(takeaway, str), "关键结论应为字符串"
            assert len(takeaway) > 0, "关键结论不能为空"

        for item in result.knowledge_doc.deep_dive:
            assert "topic" in item, "深度解析应包含 topic 字段"
            assert "explanation" in item, "深度解析应包含 explanation 字段"

        for term, definition in result.glossary.items():
            assert isinstance(term, str), "术语应为字符串"
            assert isinstance(definition, str), "定义应为字符串"
            assert len(definition) > 0, "定义不能为空"

        schemas = result.knowledge_doc.visual_schemas
        assert len(schemas) >= 3, "知识蓝图结构应至少有 3 个节点"

    @skip_if_no_proxy
    @skip_if_no_sdk_endpoint
    def test_markdown_generation(self, analyzer, test_video_path):
        result = analyzer.analyze_video(test_video_path)
        markdown = analyzer.generate_report(result)

        assert markdown.startswith("# "), "应以一级标题开始"
        assert "一句话核心" in markdown, "应包含一句话核心部分"
        assert "关键结论" in markdown, "应包含关键结论部分"
        assert "深度解析" in markdown, "应包含深度解析部分"
        assert "关键术语表" in markdown, "应包含术语表部分"
        assert "知识蓝图结构" in markdown, "应包含知识蓝图结构部分"

        for term in result.glossary.keys():
            assert f"**{term}**" in markdown, f"术语 '{term}' 应以粗体显示"

    @skip_if_no_proxy
    @skip_if_no_sdk_endpoint
    def test_api_counter_integration(self, analyzer, test_video_path, api_counter):
        initial_count = api_counter.current_count

        analyzer.analyze_video(test_video_path)

        assert api_counter.current_count == initial_count + 1
        assert api_counter.can_call(), "应该还能继续调用 API"
        assert api_counter.remaining() == 10 - api_counter.current_count


@skip_if_no_proxy
@skip_if_no_sdk_endpoint
def test_generate_example_documents(tmp_path):
    config = load_config()
    api_counter = APICounter(max_calls=10, current_count=0)
    logger = setup_logging(log_dir=tmp_path, log_name="example_generation.log")

    api_key = os.getenv("GEMINI_API_KEY")
    analyzer = ContentAnalyzer(
        config=config,
        api_counter=api_counter,
        logger=logger,
        api_key=api_key,
    )

    project_root = Path(__file__).resolve().parents[1]
    temp_videos = project_root / "data" / "temp" / "videos"

    if not temp_videos.exists():
        pytest.skip("未找到测试视频目录")

    video_files = list(temp_videos.glob("*.mp4"))[:3]

    if not video_files:
        pytest.skip("未找到测试视频文件")

    output_dir = project_root / "data" / "output" / "documents"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, video_path in enumerate(video_files, 1):
        logger.info(f"处理视频 {idx}/{len(video_files)}: {video_path.name}")

        try:
            result = analyzer.analyze_video(video_path)
            markdown = analyzer.generate_report(result)

            output_file = output_dir / f"doc_example_{idx}.md"
            output_file.write_text(markdown, encoding="utf-8")

            logger.info(f"示例文档已保存: {output_file}")

        except Exception as e:
            logger.error(f"处理视频失败: {e}")
