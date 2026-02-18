"""
Tests for VideoPipeline._validation_loop() analyzer parameter fix
"""

import logging
from pathlib import Path
import pytest
from typing import cast
from unittest.mock import patch

from analyzer.content_analyzer import ContentAnalyzer
from analyzer.models import AnalysisResult, KnowledgeDocument, VisualSchemaItem
from pipeline import VideoPipeline
from utils.counter import APICounter
from utils.logger import setup_logging
from validator.consistency_validator import ConsistencyValidator, ValidationResult

ConfigDict = dict[str, dict[str, object]]


class StubAnalyzer:
    def __init__(self, rewrite_return: str = "rewritten schema") -> None:
        self.rewrite_return: str = rewrite_return
        self.rewrite_calls: list[tuple[str, str]] = []

    def rewrite_visual_schema(self, original_structure: str, feedback: str) -> str:
        self.rewrite_calls.append((original_structure, feedback))
        return self.rewrite_return


class StubContentAnalyzer(StubAnalyzer):
    def __init__(self, analysis_result: AnalysisResult) -> None:
        super().__init__(rewrite_return="rewritten schema")
        self._analysis_result: AnalysisResult = analysis_result
        self.generate_report_calls: list[tuple[AnalysisResult, str | None, str]] = []

    def analyze_video(self, _video_path: str) -> AnalysisResult:
        return self._analysis_result

    def generate_report(
        self,
        analysis_result: AnalysisResult,
        image_relative_path: str | None,
        self_check_mode: str,
    ) -> str:
        self.generate_report_calls.append(
            (analysis_result, image_relative_path, self_check_mode)
        )
        return "final report"


class StubValidator:
    def __init__(self, results: list[ValidationResult]) -> None:
        self._results: list[ValidationResult] = results
        self.validate_calls: list[tuple[str, str]] = []

    def validate(
        self, mind_map_structure: str, knowledge_doc_content: str
    ) -> ValidationResult:
        self.validate_calls.append((mind_map_structure, knowledge_doc_content))
        return self._results[len(self.validate_calls) - 1]


class StubDownloader:
    def __init__(self, path: str) -> None:
        self._path: str = path

    def download_video(self, _url: str) -> str:
        return self._path


class StubGenerator:
    def __init__(self, image_data: bytes | None) -> None:
        self._image_data: bytes | None = image_data

    def generate_blueprint(self, _structure: str) -> bytes | None:
        return self._image_data


class StubAuditResult:
    def __init__(self, passed: bool, score: float, feedback: str) -> None:
        self.passed: bool = passed
        self.score: float = score
        self.feedback: str = feedback


class StubAuditor:
    def __init__(self, result: StubAuditResult) -> None:
        self._result: StubAuditResult = result

    def audit_image(self, **_kwargs: object) -> StubAuditResult:
        return self._result


class StubThrottle:
    def __init__(self, **_kwargs: object) -> None:
        pass


class TestValidationLoopAnalyzerFix:
    """Tests for _validation_loop analyzer parameter fix"""

    @pytest.fixture
    def mock_config(self) -> ConfigDict:
        return {
            "system": {
                "max_api_calls": 10,
                "temp_dir": "./data/temp",
                "output_dir": "./data/output",
            },
            "validator": {"threshold": 75.0, "max_rounds": 3},
            "api_keys": {},
            "proxy": {"base_url": "http://localhost:8000"},
            "analyzer": {
                "min_call_interval": 4.0,
                "retry_times": 10,
                "max_retry_wait": 600.0,
            },
        }

    @pytest.fixture
    def logger(self, tmp_path: Path) -> logging.Logger:
        return setup_logging(str(tmp_path), "test.log")

    @pytest.fixture
    def api_counter(self) -> APICounter:
        return APICounter(max_calls=10)

    def test_validation_loop_with_analyzer_parameter(
        self,
        mock_config: ConfigDict,
        logger: logging.Logger,
        api_counter: APICounter,
    ) -> None:
        """Test that _validation_loop accepts analyzer parameter"""
        validation_result = ValidationResult(
            total_score=80.0,
            accuracy=32.0,
            completeness=24.0,
            visualization=16.0,
            logic=8.0,
            feedback="Good structure",
            passed=True,
        )
        analyzer = StubAnalyzer()
        validator = StubValidator([validation_result])

        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )
        pipeline.validator = cast(ConsistencyValidator, cast(object, validator))

        initial_structure = "initial schema"
        knowledge_content = "knowledge content"

        result = pipeline._validation_loop(  # pyright: ignore[reportPrivateUsage]
            initial_structure,
            knowledge_content,
            cast(ContentAnalyzer, cast(object, analyzer)),
        )

        assert result == initial_structure
        assert len(validator.validate_calls) == 1
        assert analyzer.rewrite_calls == []

    def test_validation_loop_no_nameerror(
        self,
        mock_config: ConfigDict,
        logger: logging.Logger,
        api_counter: APICounter,
    ) -> None:
        """Test that no NameError is raised when validation fails and rewrite is attempted"""
        analyzer = StubAnalyzer()
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        validation_result_fail = ValidationResult(
            total_score=60.0,
            accuracy=24.0,
            completeness=18.0,
            visualization=12.0,
            logic=6.0,
            feedback="Needs improvement",
            passed=False,
        )
        validation_result_pass = ValidationResult(
            total_score=85.0,
            accuracy=34.0,
            completeness=25.5,
            visualization=17.0,
            logic=8.5,
            feedback="Good after rewrite",
            passed=True,
        )
        validator = StubValidator([validation_result_fail, validation_result_pass])
        pipeline.validator = cast(ConsistencyValidator, cast(object, validator))

        initial_structure = "initial schema"
        knowledge_content = "knowledge content"
        rewritten_structure = "rewritten schema"
        analyzer.rewrite_return = rewritten_structure

        result = pipeline._validation_loop(  # pyright: ignore[reportPrivateUsage]
            initial_structure,
            knowledge_content,
            cast(ContentAnalyzer, cast(object, analyzer)),
        )

        assert result == rewritten_structure
        assert len(validator.validate_calls) == 2
        assert analyzer.rewrite_calls == [(initial_structure, "Needs improvement")]

    def test_process_single_video_validation_rewrite(
        self,
        mock_config: ConfigDict,
        logger: logging.Logger,
        api_counter: APICounter,
        tmp_path: Path,
    ) -> None:
        """Test full flow: validation fails, rewrite is invoked via analyzer parameter"""
        download_path = str(tmp_path / "test.mp4")
        knowledge_doc = KnowledgeDocument(
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
                VisualSchemaItem(
                    type="overview", description="", schema="initial schema"
                )
            ],
        )
        analysis_result = AnalysisResult(
            video_path=download_path,
            knowledge_doc=knowledge_doc,
            metadata={"duration": 60.0},
        )
        validation_fail = ValidationResult(
            total_score=60.0,
            accuracy=24.0,
            completeness=18.0,
            visualization=12.0,
            logic=6.0,
            feedback="Needs improvement",
            passed=False,
        )
        validation_pass = ValidationResult(
            total_score=85.0,
            accuracy=34.0,
            completeness=25.5,
            visualization=17.0,
            logic=8.5,
            feedback="Good after rewrite",
            passed=True,
        )
        stub_analyzer = StubContentAnalyzer(analysis_result)
        stub_validator = StubValidator([validation_fail, validation_pass])

        with (
            patch(
                "pipeline.VideoDownloader", return_value=StubDownloader(download_path)
            ),
            patch("pipeline.ContentAnalyzer", return_value=stub_analyzer),
            patch("pipeline.ConsistencyValidator", return_value=stub_validator),
            patch("pipeline.ImageGenerator", return_value=StubGenerator(b"image_data")),
            patch(
                "pipeline.QualityAuditor",
                return_value=StubAuditor(
                    StubAuditResult(passed=True, score=90.0, feedback="Good")
                ),
            ),
            patch("pipeline.GeminiThrottle", StubThrottle),
        ):
            mock_config["system"]["output_dir"] = str(tmp_path)
            pipeline = VideoPipeline(
                config=mock_config, logger=logger, api_counter=api_counter
            )

            result = pipeline.process_single_video("https://example.com/video")

            assert result.success
            assert stub_validator.validate_calls[0][0] == "initial schema"
            assert stub_analyzer.rewrite_calls == [
                ("initial schema", "Needs improvement")
            ]
