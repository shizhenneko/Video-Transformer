"""
Pipeline initialization fix tests - verify no duplicate initialization
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipeline import VideoPipeline
from utils.counter import APICounter
from utils.logger import setup_logging


class TestPipelineInitFix:
    """Test that pipeline components are initialized only once"""

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
    def logger(self, tmp_path):
        """Test logger"""
        return setup_logging(str(tmp_path), "test.log")

    @pytest.fixture
    def api_counter(self):
        """API counter"""
        return APICounter(max_calls=10)

    @patch("pipeline.ConsistencyValidator")
    @patch("pipeline.ImageGenerator")
    @patch("pipeline.VideoDownloader")
    def test_pipeline_init_single_validator(
        self,
        mock_downloader,
        mock_generator,
        mock_validator,
        mock_config,
        logger,
        api_counter,
    ):
        """Test that ConsistencyValidator is instantiated exactly once"""
        # Create pipeline
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        # Verify ConsistencyValidator was called exactly once
        assert mock_validator.call_count == 1, (
            f"Expected 1 call, got {mock_validator.call_count}"
        )

    @patch("pipeline.ConsistencyValidator")
    @patch("pipeline.ImageGenerator")
    @patch("pipeline.VideoDownloader")
    def test_pipeline_init_single_generator(
        self,
        mock_downloader,
        mock_generator,
        mock_validator,
        mock_config,
        logger,
        api_counter,
    ):
        """Test that ImageGenerator is instantiated exactly once"""
        # Create pipeline
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        # Verify ImageGenerator was called exactly once
        assert mock_generator.call_count == 1, (
            f"Expected 1 call, got {mock_generator.call_count}"
        )

    @patch("pipeline.ConsistencyValidator")
    @patch("pipeline.ImageGenerator")
    @patch("pipeline.VideoDownloader")
    def test_pipeline_components_accessible(
        self,
        mock_downloader,
        mock_generator,
        mock_validator,
        mock_config,
        logger,
        api_counter,
    ):
        """Test that validator and generator components are accessible after init"""
        # Create pipeline
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        # Verify components are accessible
        assert pipeline.validator is not None, "validator should be accessible"
        assert pipeline.generator is not None, "generator should be accessible"
        assert pipeline.downloader is not None, "downloader should be accessible"
