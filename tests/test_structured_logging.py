"""
Structured logging tests for pipeline events
"""

import pytest
import re
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import logging

from pipeline import VideoPipeline  # type: ignore[reportMissingImports]
from models import ProcessResult  # type: ignore[reportMissingImports]
from utils.counter import APICounter  # type: ignore[reportMissingImports]
from utils.gemini_throttle import GeminiThrottle  # type: ignore[reportMissingImports]


class TestStructuredLogging:
    """Test structured logging in pipeline"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration"""
        return {
            "system": {
                "max_api_calls": 10,
                "temp_dir": "./data/temp",
                "output_dir": "./data/output",
            },
            "validator": {"threshold": 75.0, "max_rounds": 3},
            "api_keys": {},
            "proxy": {"base_url": "http://localhost:8000"},
        }

    @pytest.fixture
    def logger_with_capture(self):
        """Logger that captures output to string"""
        logger = logging.getLogger("test_structured_logging")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # Create string handler to capture logs
        string_handler = logging.StreamHandler(StringIO())
        string_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        string_handler.setFormatter(formatter)
        logger.addHandler(string_handler)

        return logger, string_handler

    @pytest.fixture
    def api_counter(self):
        """API counter"""
        return APICounter(max_calls=10)

    def get_log_output(self, handler):
        """Extract log output from handler"""
        return handler.stream.getvalue()

    @patch("pipeline.VideoDownloader")
    @patch("pipeline.ContentAnalyzer")
    @patch("pipeline.ConsistencyValidator")
    @patch("pipeline.ImageGenerator")
    @patch("pipeline.QualityAuditor")
    def test_structured_logging_video_start(
        self,
        mock_auditor,
        mock_generator,
        mock_validator,
        mock_analyzer,
        mock_downloader,
        mock_config,
        logger_with_capture,
        api_counter,
        tmp_path,
    ):
        """Test that video_start event is logged with video_id"""
        logger, handler = logger_with_capture
        mock_config["system"]["output_dir"] = str(tmp_path)

        # Mock downloader
        mock_downloader_inst = Mock()
        mock_downloader_inst.download_video.return_value = str(
            tmp_path / "test_video.mp4"
        )
        mock_downloader.return_value = mock_downloader_inst

        # Mock analyzer
        mock_analysis_result = Mock()
        mock_analysis_result.knowledge_doc.visual_schemas = []
        mock_analysis_result.knowledge_doc.deep_dive = []  # Add deep_dive as list
        mock_analysis_result.knowledge_doc.to_markdown.return_value = "# Test"
        mock_analyzer_inst = Mock()
        mock_analyzer_inst.analyze_video.return_value = mock_analysis_result
        mock_analyzer_inst.generate_report.return_value = "# Report"
        mock_analyzer.return_value = mock_analyzer_inst

        # Mock validator
        mock_validation_result = Mock()
        mock_validation_result.passed = True
        mock_validator_inst = Mock()
        mock_validator_inst.validate.return_value = mock_validation_result
        mock_validator.return_value = mock_validator_inst

        # Mock generator
        mock_generator_inst = Mock()
        mock_generator_inst.generate_blueprint.return_value = None
        mock_generator.return_value = mock_generator_inst

        # Mock auditor
        mock_auditor_inst = Mock()
        mock_auditor.return_value = mock_auditor_inst

        # Create pipeline
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        # Process video
        url = "https://www.bilibili.com/video/BV1xx411c7mD"
        result = pipeline.process_single_video(url)

        # Get log output
        log_output = self.get_log_output(handler)

        # Assert video_start event is present with video_id
        assert "event=video_start" in log_output
        assert "video_id=BV1xx411c7mD" in log_output

        # Verify structured format
        video_start_match = re.search(r"event=video_start\s+video_id=(\S+)", log_output)
        assert video_start_match is not None
        assert video_start_match.group(1) == "BV1xx411c7mD"

    @patch("pipeline.VideoDownloader")
    @patch("pipeline.ContentAnalyzer")
    @patch("pipeline.ConsistencyValidator")
    @patch("pipeline.ImageGenerator")
    @patch("pipeline.QualityAuditor")
    def test_structured_logging_video_complete(
        self,
        mock_auditor,
        mock_generator,
        mock_validator,
        mock_analyzer,
        mock_downloader,
        mock_config,
        logger_with_capture,
        api_counter,
        tmp_path,
    ):
        """Test that video_complete event is logged with elapsed_s"""
        logger, handler = logger_with_capture
        mock_config["system"]["output_dir"] = str(tmp_path)

        # Mock downloader
        mock_downloader_inst = Mock()
        mock_downloader_inst.download_video.return_value = str(
            tmp_path / "test_video.mp4"
        )
        mock_downloader.return_value = mock_downloader_inst

        # Mock analyzer
        mock_analysis_result = Mock()
        mock_analysis_result.knowledge_doc.visual_schemas = []
        mock_analysis_result.knowledge_doc.deep_dive = []  # Add deep_dive as list
        mock_analysis_result.knowledge_doc.to_markdown.return_value = "# Test"
        mock_analyzer_inst = Mock()
        mock_analyzer_inst.analyze_video.return_value = mock_analysis_result
        mock_analyzer_inst.generate_report.return_value = "# Report"
        mock_analyzer.return_value = mock_analyzer_inst

        # Mock validator
        mock_validation_result = Mock()
        mock_validation_result.passed = True
        mock_validator_inst = Mock()
        mock_validator_inst.validate.return_value = mock_validation_result
        mock_validator.return_value = mock_validator_inst

        # Mock generator
        mock_generator_inst = Mock()
        mock_generator_inst.generate_blueprint.return_value = None
        mock_generator.return_value = mock_generator_inst

        # Mock auditor
        mock_auditor_inst = Mock()
        mock_auditor.return_value = mock_auditor_inst

        # Create pipeline
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        # Process video
        url = "https://www.bilibili.com/video/BV1xx411c7mD"
        result = pipeline.process_single_video(url)

        # Get log output
        log_output = self.get_log_output(handler)

        # Assert video_complete event is present with video_id and elapsed_s
        assert "event=video_complete" in log_output
        assert "video_id=BV1xx411c7mD" in log_output
        assert "elapsed_s=" in log_output

        # Verify structured format with elapsed time
        video_complete_match = re.search(
            r"event=video_complete\s+video_id=(\S+)\s+elapsed_s=([\d.]+)", log_output
        )
        assert video_complete_match is not None
        assert video_complete_match.group(1) == "BV1xx411c7mD"
        elapsed_s = float(video_complete_match.group(2))
        assert elapsed_s >= 0.0

    def test_log_parsing(self):
        """Test that structured log output can be parsed"""
        # Sample log lines with structured fields
        log_lines = [
            "INFO - event=video_start video_id=BV1xx411c7mD",
            "INFO - event=video_complete video_id=BV1xx411c7mD elapsed_s=45.2",
            "ERROR - event=video_failed video_id=BV1yy422d8nE elapsed_s=12.5 error=API_LIMIT_EXCEEDED",
            "ERROR - event=json_parse_failed reason=llm_repair_exhausted",
            "ERROR - event=validation_failed reason=missing_required_fields fields=title,glossary",
        ]

        # Parse each log line and extract structured fields
        for log_line in log_lines:
            # Extract key=value pairs
            fields = {}
            for match in re.finditer(r"(\w+)=([\w.,]+)", log_line):
                key, value = match.groups()
                fields[key] = value

            # Verify we can extract fields
            assert "event" in fields

            # Verify specific event fields
            if fields["event"] == "video_start":
                assert "video_id" in fields
                assert fields["video_id"] == "BV1xx411c7mD"

            elif fields["event"] == "video_complete":
                assert "video_id" in fields
                assert "elapsed_s" in fields
                assert float(fields["elapsed_s"]) == 45.2

            elif fields["event"] == "video_failed":
                assert "video_id" in fields
                assert "elapsed_s" in fields
                assert "error" in fields

            elif fields["event"] == "json_parse_failed":
                assert "reason" in fields

            elif fields["event"] == "validation_failed":
                assert "reason" in fields
                assert "fields" in fields

    def test_429_logging_includes_required_fields(
        self,
        logger_with_capture,
        monkeypatch,
    ):
        """Test that 429 logs include structured fields."""
        logger, handler = logger_with_capture

        throttle = GeminiThrottle(
            min_interval=0.0,
            max_retries=2,
            max_total_wait=1.0,
            logger=logger,
        )

        monkeypatch.setattr("utils.gemini_throttle.time.sleep", lambda *_: None)

        def _always_429() -> None:
            raise RuntimeError("429 resource exhausted retry in 1s")

        with pytest.raises(RuntimeError):
            throttle.call_with_retry(
                _always_429,
                log_context={
                    "endpoint": "models.generate_content_stream",
                    "model": "gemini-2.5-flash",
                    "key_id": "key-1",
                },
            )

        log_output = self.get_log_output(handler)
        rate_limit_lines = [
            line for line in log_output.splitlines() if "⚠️ 429 detected" in line
        ]
        assert rate_limit_lines

        log_line = rate_limit_lines[-1]
        required_fields = [
            "timestamp=",
            "endpoint=",
            "model=",
            "key_id=",
            "attempt=",
            "status_code=429",
            "retry_after=",
            "error=",
        ]
        for field in required_fields:
            assert field in log_line
