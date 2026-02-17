"""
Tests for VideoPipeline._validation_loop() analyzer parameter fix
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from pipeline import VideoPipeline
from utils.counter import APICounter
from utils.logger import setup_logging
from validator.consistency_validator import ValidationResult


class TestValidationLoopAnalyzerFix:
    """Tests for _validation_loop analyzer parameter fix"""

    @pytest.fixture
    def mock_config(self):
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
    def logger(self, tmp_path):
        return setup_logging(str(tmp_path), "test.log")

    @pytest.fixture
    def api_counter(self):
        return APICounter(max_calls=10)

    @pytest.fixture
    def mock_analyzer(self):
        analyzer = Mock()
        analyzer.rewrite_visual_schema = Mock(return_value="rewritten schema")
        return analyzer

    @pytest.fixture
    def mock_validator(self):
        validator = Mock()
        return validator

    def test_validation_loop_with_analyzer_parameter(
        self, mock_config, logger, api_counter, mock_analyzer, mock_validator
    ):
        """Test that _validation_loop accepts analyzer parameter"""
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )
        pipeline.validator = mock_validator

        validation_result = ValidationResult(
            total_score=80.0,
            accuracy=32.0,
            completeness=24.0,
            visualization=16.0,
            logic=8.0,
            feedback="Good structure",
            passed=True,
        )
        mock_validator.validate.return_value = validation_result

        initial_structure = "initial schema"
        knowledge_content = "knowledge content"

        result = pipeline._validation_loop(
            initial_structure, knowledge_content, mock_analyzer
        )

        assert result == initial_structure
        mock_validator.validate.assert_called_once()
        mock_analyzer.rewrite_visual_schema.assert_not_called()

    def test_validation_loop_no_nameerror(
        self, mock_config, logger, api_counter, mock_analyzer, mock_validator
    ):
        """Test that no NameError is raised when validation fails and rewrite is attempted"""
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )
        pipeline.validator = mock_validator

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
        mock_validator.validate.side_effect = [
            validation_result_fail,
            validation_result_pass,
        ]

        initial_structure = "initial schema"
        knowledge_content = "knowledge content"
        rewritten_structure = "rewritten schema"
        mock_analyzer.rewrite_visual_schema.return_value = rewritten_structure

        result = pipeline._validation_loop(
            initial_structure, knowledge_content, mock_analyzer
        )

        assert result == rewritten_structure
        assert mock_validator.validate.call_count == 2
        mock_analyzer.rewrite_visual_schema.assert_called_once_with(
            original_structure=initial_structure,
            feedback="Needs improvement",
        )

    def test_process_single_video_validation_rewrite(
        self, mock_config, logger, api_counter, mock_analyzer
    ):
        """Test full flow: validation fails, rewrite is invoked via analyzer parameter"""
        with (
            patch("pipeline.VideoDownloader") as MockDownloader,
            patch("pipeline.ContentAnalyzer") as MockAnalyzer,
            patch("pipeline.ConsistencyValidator") as MockValidator,
            patch("pipeline.ImageGenerator") as MockGenerator,
            patch("pipeline.QualityAuditor") as MockAuditor,
            patch("pipeline.GeminiThrottle") as MockThrottle,
        ):
            # Setup mocks
            mock_downloader = MockDownloader.return_value
            mock_downloader.download_video.return_value = "/tmp/test.mp4"

            mock_analyzer_instance = MockAnalyzer.return_value
            mock_analysis_result = Mock()
            mock_analysis_result.knowledge_doc.deep_dive = ["point1", "point2"]
            mock_analysis_result.knowledge_doc.visual_schemas = [
                Mock(schema="initial schema")
            ]
            mock_analysis_result.knowledge_doc.to_markdown.return_value = (
                "knowledge content"
            )
            mock_analyzer_instance.analyze_video.return_value = mock_analysis_result
            mock_analyzer_instance.rewrite_visual_schema.return_value = (
                "rewritten schema"
            )
            mock_analyzer_instance.generate_report.return_value = "final report"

            mock_validator_instance = MockValidator.return_value
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
            mock_validator_instance.validate.side_effect = [
                validation_fail,
                validation_pass,
            ]

            mock_generator_instance = MockGenerator.return_value
            mock_generator_instance.generate_blueprint.return_value = b"image_data"

            mock_auditor_instance = MockAuditor.return_value
            mock_auditor_instance.audit_image.return_value = Mock(
                passed=True, score=90.0, feedback="Good"
            )

            # Create pipeline
            pipeline = VideoPipeline(
                config=mock_config, logger=logger, api_counter=api_counter
            )

            # Process video
            result = pipeline.process_single_video("https://example.com/video")

            # Verify rewrite was called
            assert result.success
            mock_analyzer_instance.rewrite_visual_schema.assert_called_once_with(
                original_structure="initial schema",
                feedback="Needs improvement",
            )
