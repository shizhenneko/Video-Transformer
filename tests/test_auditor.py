"""
质量审核器单元测试
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from auditor.quality_auditor import QualityAuditor, AuditResult
from utils.counter import APICounter, APILimitExceeded


class TestQualityAuditor:
    """测试质量审核器"""

    @pytest.fixture
    def mock_config(self):
        return {
            "auditor": {
                "model": "gemini-2.5-flash",
                "threshold": 75.0,
                "timeout": 60,
            },
            "proxy": {
                "base_url": "http://localhost:8000",
                "timeout": 10,
            },
        }

    @pytest.fixture
    def mock_api_counter(self):
        return APICounter(max_calls=10, current_count=0)

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_init_with_fixed_key(self, mock_config, mock_api_counter, mock_logger):
        """测试:使用固定 Key 初始化"""
        with patch("auditor.quality_auditor.genai.Client") as mock_client:
            auditor = QualityAuditor(
                config=mock_config,
                api_counter=mock_api_counter,
                logger=mock_logger,
                api_key="test-gemini-key",
            )

            mock_client.assert_called_once_with(api_key="test-gemini-key")
            assert auditor._fixed_api_key == "test-gemini-key"

    def test_init_without_key(self, mock_config, mock_api_counter, mock_logger):
        """测试:不使用固定 Key 初始化"""
        auditor = QualityAuditor(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        assert auditor._fixed_api_key is None
        assert auditor._client is None

    @patch("auditor.quality_auditor.genai.Client")
    def test_audit_image_success(
        self,
        mock_client_class,
        mock_config,
        mock_api_counter,
        mock_logger,
        tmp_path,
    ):
        """测试:成功审核图片"""
        # 创建临时图片文件
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        # Mock Client
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock 文件上传
        mock_file = MagicMock()
        mock_file.uri = "file://test"
        mock_file.mime_type = "image/png"
        mock_client.files.upload.return_value = mock_file

        # Mock API 响应
        mock_response = MagicMock()
        mock_response.text = """评分: 85
反馈: 图片质量优秀
通过: 是"""
        mock_client.models.generate_content.return_value = mock_response

        # 初始化审核器(使用固定 Key)
        auditor = QualityAuditor(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
            api_key="test-key",
        )

        # 执行
        result = auditor.audit_image(
            image_path=image_path,
            knowledge_doc_content="测试内容",
        )

        # 断言
        assert isinstance(result, AuditResult)
        assert result.score == 85
        assert result.passed is True
        assert mock_api_counter.current_count == 1
        mock_client.files.delete.assert_called_once_with(name=mock_file.name)

    @patch("auditor.quality_auditor.genai.Client")
    def test_audit_image_api_limit(
        self,
        mock_client_class,
        mock_config,
        mock_api_counter,
        mock_logger,
        tmp_path,
    ):
        """测试:API 调用次数超限"""
        mock_api_counter.current_count = 10  # 已达上限

        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        auditor = QualityAuditor(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
            api_key="test-key",
        )

        with pytest.raises(APILimitExceeded):
            auditor.audit_image(
                image_path=image_path,
                knowledge_doc_content="测试",
            )

    def test_audit_image_file_not_found(
        self,
        mock_config,
        mock_api_counter,
        mock_logger,
    ):
        """测试:图片文件不存在"""
        auditor = QualityAuditor(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
            api_key="test-key",
        )

        with pytest.raises(FileNotFoundError):
            auditor.audit_image(
                image_path="/nonexistent/image.png",
                knowledge_doc_content="测试",
            )
