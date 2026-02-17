"""
验证 Gemini API Key 复用逻辑的测试
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pipeline import VideoPipeline
from utils.counter import APICounter
from utils.logger import setup_logging

class TestAPIKeyReuse:
    """验证 API Key 仅分配一次的测试类"""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """模拟配置"""
        return {
            "system": {
                "max_api_calls": 10,
                "temp_dir": str(tmp_path / "temp"),
                "output_dir": str(tmp_path / "output"),
                "log_dir": str(tmp_path / "logs"),
            },
            "proxy": {
                "base_url": "http://localhost:8080",
                "timeout": 5
            },
            "api_keys": {},
            "validator": {"threshold": 75.0, "max_rounds": 3},
            "auditor": {"model": "gemini-2.5-flash", "threshold": 75.0}
        }

    @pytest.fixture
    def logger(self, tmp_path):
        """测试日志"""
        return setup_logging(str(tmp_path / "logs"), "test_reuse.log")

    @pytest.fixture
    def api_counter(self):
        """API 计数器"""
        return APICounter(max_calls=10)

    @patch("requests.post")
    def test_key_allocation_once(self, mock_post, mock_config, logger, api_counter):
        """测试: 统一 Key 分配逻辑仅调用一次"""
        
        # 1. 模拟号池响应
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "key_id": "test-key-001",
            "api_key": "AIza-test-key-content"
        }
        mock_post.return_value = mock_resp

        # 2. 初始化 Pipeline
        # 此时应该触发第一次 allocate-key
        pipeline = VideoPipeline(
            config=mock_config,
            logger=logger,
            api_counter=api_counter
        )

        # 验证号池接口被调用
        assert mock_post.called
        assert mock_post.call_count == 1
        assert pipeline._gemini_api_key == "AIza-test-key-content"

        # 3. 验证子模块获取到了正确的 Key
        assert pipeline.analyzer._allocated_api_key == "AIza-test-key-content"
        assert pipeline.auditor._fixed_api_key == "AIza-test-key-content"

    @patch("requests.post")
    @patch("analyzer.content_analyzer.genai.Client")
    @patch("auditor.quality_auditor.genai.Client")
    def test_process_does_not_reallocate(self, mock_auditor_client, mock_analyzer_client, mock_post, mock_config, logger, api_counter):
        """测试: 在处理视频过程中不会再次分配 Key"""
        
        # 1. 设置号池模拟
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"key_id": "k1", "api_key": "v1"}
        mock_post.return_value = mock_resp

        # 2. 初始化 Pipeline
        pipeline = VideoPipeline(mock_config, logger, api_counter)
        initial_call_count = mock_post.call_count
        assert initial_call_count == 1

        # 3. 模拟视频处理流程中的相关组件
        # 模拟下载
        pipeline.downloader = Mock()
        pipeline.downloader.download_video.return_value = Path("test.mp4")
        
        # 模拟分析
        mock_analysis = Mock()
        mock_analysis.knowledge_notes = "notes"
        mock_analysis.mind_map_structure = "structure"
        pipeline.analyzer.analyze_video = Mock(return_value=mock_analysis)
        pipeline.analyzer.generate_report = Mock(return_value="report")
        
        # 模拟校验
        mock_validation = Mock()
        mock_validation.passed = True
        mock_validation.total_score = 90.0
        pipeline.validator.validate = Mock(return_value=mock_validation)
        
        # 模拟生成
        pipeline.generator.generate_blueprint = Mock(return_value=b"img")
        pipeline.generator.save_image = Mock()
        
        # 模拟审计
        mock_audit = Mock()
        mock_audit.score = 80.0
        pipeline.auditor.audit_image = Mock(return_value=mock_audit)

        # 4. 执行处理
        pipeline.process_single_video("http://test.com/v1")

        # 5. 断言: 号池分配接口的调用次数仍然为 1 (初始化时那一次)
        assert mock_post.call_count == 1
        
    def test_fixed_key_no_allocation(self, mock_config, logger, api_counter):
        """测试: 如果配置了固定 Key,则根本不显式调用号池"""
        
        # 在配置中加入固定 Key
        mock_config["api_keys"]["gemini"] = "AIza-fixed-key"
        
        with patch("requests.post") as mock_post:
            pipeline = VideoPipeline(mock_config, logger, api_counter)
            
            # 应该直接使用,不调用号池
            assert mock_post.call_count == 0
            assert pipeline._gemini_api_key == "AIza-fixed-key"
            assert pipeline.analyzer._allocated_api_key == "AIza-fixed-key"
