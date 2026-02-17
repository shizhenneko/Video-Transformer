"""
Pipeline 单元测试
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pipeline import VideoPipeline
from models import ProcessResult
from utils.counter import APICounter
from utils.logger import setup_logging


class TestVideoPipeline:
    """VideoPipeline 单元测试"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
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
        """测试日志"""
        return setup_logging(str(tmp_path), "test.log")

    @pytest.fixture
    def api_counter(self):
        """API 计数器"""
        return APICounter(max_calls=10)

    def test_initialization(self, mock_config, logger, api_counter):
        """测试初始化"""
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        assert pipeline.max_validation_rounds == 3
        assert pipeline.validation_threshold == 75.0
        assert pipeline.downloader is not None

    def test_extract_video_id_bilibili(self, mock_config, logger, api_counter):
        """测试提取 Bilibili 视频 ID"""
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        url = "https://www.bilibili.com/video/BV1xx411c7mD"
        video_id = pipeline._extract_video_id(url)

        assert video_id == "BV1xx411c7mD"

    def test_extract_video_id_youtube(self, mock_config, logger, api_counter):
        """测试提取 YouTube 视频 ID"""
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = pipeline._extract_video_id(url)

        assert video_id == "dQw4w9WgXcQ"

    @patch("pipeline.VideoDownloader")
    @patch("pipeline.ContentAnalyzer")
    @patch("pipeline.ConsistencyValidator")
    @patch("pipeline.ImageGenerator")
    @patch("pipeline.QualityAuditor")
    def test_process_single_video_mock(
        self,
        mock_auditor,
        mock_generator,
        mock_validator,
        mock_analyzer,
        mock_downloader,
        mock_config,
        logger,
        api_counter,
        tmp_path,
    ):
        """测试处理单个视频(模拟)"""
        # 设置输出目录
        mock_config["system"]["output_dir"] = str(tmp_path)

        # 模拟下载器
        mock_downloader_inst = Mock()
        mock_downloader_inst.download_video.return_value = str(
            tmp_path / "test_video.mp4"
        )
        mock_downloader.return_value = mock_downloader_inst

        # 模拟分析器
        mock_analysis_result = Mock()
        mock_analysis_result.knowledge_notes = "# Test Notes"
        mock_analysis_result.knowledge_doc.visual_schemas = []

        mock_analyzer_inst = Mock()
        mock_analyzer_inst.analyze_video.return_value = mock_analysis_result
        mock_analyzer_inst.generate_report.return_value = "# Report"
        mock_analyzer.return_value = mock_analyzer_inst

        # 模拟校验器
        mock_validation_result = Mock()
        mock_validation_result.passed = True
        mock_validation_result.total_score = 80.0

        mock_validator_inst = Mock()
        mock_validator_inst.validate.return_value = mock_validation_result
        mock_validator.return_value = mock_validator_inst

        # 模拟图像生成器
        mock_generator_inst = Mock()
        mock_generator_inst.generate_blueprint.return_value = b"fake_image_data"
        mock_generator_inst.save_image.return_value = str(tmp_path / "blueprint.png")
        mock_generator.return_value = mock_generator_inst

        # 模拟审核器
        mock_audit_result = Mock()
        mock_audit_result.score = 85.0

        mock_auditor_inst = Mock()
        mock_auditor_inst.audit_image.return_value = mock_audit_result
        mock_auditor.return_value = mock_auditor_inst

        # 创建 pipeline
        pipeline = VideoPipeline(
            config=mock_config, logger=logger, api_counter=api_counter
        )

        # 处理视频
        result = pipeline.process_single_video("https://test.com/video")

        # 断言
        assert result.success is True
        assert result.video_id is not None
        assert mock_downloader_inst.download_video.called
        assert mock_analyzer_inst.analyze_video.called
        assert mock_validator_inst.validate.called
        assert mock_generator_inst.generate_blueprint.called
        assert mock_auditor_inst.audit_image.called

    def test_resolve_self_check_mode_default(self):
        config_static = {"system": {"self_check_mode": "static"}}
        config_interactive = {"system": {"self_check_mode": "interactive"}}
        config_questions_only = {"system": {"self_check_mode": "questions_only"}}
        config_default = {"system": {"self_check_mode": "default"}}
        config_invalid = {"system": {"self_check_mode": "invalid_mode"}}
        config_missing = {"system": {}}

        assert VideoPipeline._resolve_self_check_mode(config_static) == "static"
        assert (
            VideoPipeline._resolve_self_check_mode(config_interactive) == "interactive"
        )
        assert (
            VideoPipeline._resolve_self_check_mode(config_questions_only)
            == "questions_only"
        )
        assert VideoPipeline._resolve_self_check_mode(config_default) == "default"
        assert VideoPipeline._resolve_self_check_mode(config_invalid) == "static"
        assert VideoPipeline._resolve_self_check_mode(config_missing) == "static"
