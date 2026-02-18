"""
验证 Gemini API Key 复用逻辑的测试
"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from analyzer.models import AnalysisResult, KnowledgeDocument, VisualSchemaItem
from pipeline import VideoPipeline
from utils.counter import APICounter
from utils.logger import setup_logging

ConfigDict = dict[str, dict[str, object]]


class StubAnalyzer:
    def __init__(self, analysis_result: AnalysisResult) -> None:
        self._analysis_result: AnalysisResult = analysis_result

    def analyze_video(self, _video_path: str) -> AnalysisResult:
        return self._analysis_result

    def generate_report(
        self,
        _analysis_result: AnalysisResult,
        _image_relative_path: str | None,
        _self_check_mode: str | None = None,
        **_kwargs: object,
    ) -> str:
        return "# Report"


class StubValidatorResult:
    def __init__(self) -> None:
        self.passed: bool = True
        self.total_score: float = 90.0


class StubValidator:
    def validate(self, *_args: object, **_kwargs: object) -> StubValidatorResult:
        return StubValidatorResult()


class StubGenerator:
    def generate_blueprint(self, *_args: object, **_kwargs: object) -> None:
        return None


class StubAuditor:
    def audit_image(self, **_kwargs: object) -> object:
        return type("Audit", (), {"score": 80.0, "passed": True})()


class StubDownloader:
    def __init__(self, path: str) -> None:
        self._path: str = path

    def download_video(self, _url: str) -> str:
        return self._path


def build_analysis_result(video_path: str) -> AnalysisResult:
    doc = KnowledgeDocument(
        title="Test",
        one_sentence_summary="Summary",
        key_takeaways=["Takeaway"],
        deep_dive=[
            {
                "chapter_title": "Chapter",
                "chapter_summary": "Summary",
                "sections": [
                    {
                        "topic": "Topic",
                        "explanation": "Explain",
                        "example": "Example",
                    }
                ],
            }
        ],
        glossary={},
        visual_schemas=[
            VisualSchemaItem(type="overview", description="", schema="initial schema")
        ],
    )
    return AnalysisResult(
        video_path=video_path,
        knowledge_doc=doc,
        metadata={"duration": 60.0},
    )


class StubResponse:
    def __init__(self, payload: dict[str, str]) -> None:
        self.status_code: int = 200
        self._payload: dict[str, str] = payload

    def json(self) -> dict[str, str]:
        return self._payload


class TestAPIKeyReuse:
    """验证 API Key 仅分配一次的测试类"""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> ConfigDict:
        """模拟配置"""
        return {
            "system": {
                "max_api_calls": 10,
                "temp_dir": str(tmp_path / "temp"),
                "output_dir": str(tmp_path / "output"),
                "log_dir": str(tmp_path / "logs"),
            },
            "proxy": {"base_url": "http://localhost:8080", "timeout": 5},
            "api_keys": {},
            "validator": {"threshold": 75.0, "max_rounds": 3},
            "auditor": {"model": "gemini-2.5-flash", "threshold": 75.0},
        }

    @pytest.fixture
    def logger(self, tmp_path: Path) -> logging.Logger:
        """测试日志"""
        return setup_logging(str(tmp_path / "logs"), "test_reuse.log")

    @pytest.fixture
    def api_counter(self) -> APICounter:
        """API 计数器"""
        return APICounter(max_calls=10)

    @patch("requests.post")
    def test_key_allocation_once(
        self,
        mock_post: Mock,
        mock_config: ConfigDict,
        logger: logging.Logger,
        api_counter: APICounter,
    ) -> None:
        """测试: 统一 Key 分配逻辑仅调用一次"""

        # 1. 模拟号池响应
        mock_post.return_value = StubResponse(
            {"key_id": "test-key-001", "api_key": "AIza-test-key-content"}
        )

        analysis_result = build_analysis_result("test.mp4")
        stub_analyzer = StubAnalyzer(analysis_result)

        with (
            patch("pipeline.VideoDownloader", return_value=StubDownloader("test.mp4")),
            patch("pipeline.ContentAnalyzer", return_value=stub_analyzer),
            patch("pipeline.ConsistencyValidator", return_value=StubValidator()),
            patch("pipeline.ImageGenerator", return_value=StubGenerator()),
            patch("pipeline.QualityAuditor", return_value=StubAuditor()),
        ):
            pipeline = VideoPipeline(
                config=mock_config, logger=logger, api_counter=api_counter
            )

            result = pipeline.process_single_video("http://test.com/v1")

            assert result.success
            assert mock_post.call_count == 1

    @patch("requests.post")
    def test_process_allocates_per_video(
        self,
        mock_post: Mock,
        mock_config: ConfigDict,
        logger: logging.Logger,
        api_counter: APICounter,
    ) -> None:

        # 1. 设置号池模拟
        mock_post.return_value = StubResponse({"key_id": "k1", "api_key": "v1"})

        analysis_result = build_analysis_result("test.mp4")
        stub_analyzer = StubAnalyzer(analysis_result)

        with (
            patch("pipeline.VideoDownloader", return_value=StubDownloader("test.mp4")),
            patch("pipeline.ContentAnalyzer", return_value=stub_analyzer),
            patch("pipeline.ConsistencyValidator", return_value=StubValidator()),
            patch("pipeline.ImageGenerator", return_value=StubGenerator()),
            patch("pipeline.QualityAuditor", return_value=StubAuditor()),
        ):
            pipeline = VideoPipeline(mock_config, logger, api_counter)
            _ = pipeline.process_single_video("http://test.com/v1")
            _ = pipeline.process_single_video("http://test.com/v2")

            assert mock_post.call_count == 2

    def test_fixed_key_no_allocation(
        self, mock_config: ConfigDict, logger: logging.Logger, api_counter: APICounter
    ) -> None:
        """测试: 如果配置了固定 Key,则根本不显式调用号池"""

        # 在配置中加入固定 Key
        mock_config["api_keys"]["gemini"] = "AIza-fixed-key"

        analysis_result = build_analysis_result("test.mp4")
        stub_analyzer = StubAnalyzer(analysis_result)

        with (
            patch("requests.post") as mock_post,
            patch("pipeline.VideoDownloader", return_value=StubDownloader("test.mp4")),
            patch("pipeline.ContentAnalyzer", return_value=stub_analyzer),
            patch("pipeline.ConsistencyValidator", return_value=StubValidator()),
            patch("pipeline.ImageGenerator", return_value=StubGenerator()),
            patch("pipeline.QualityAuditor", return_value=StubAuditor()),
        ):
            pipeline = VideoPipeline(mock_config, logger, api_counter)
            result = pipeline.process_single_video("http://test.com/v1")

            assert result.success
            assert mock_post.call_count == 0
