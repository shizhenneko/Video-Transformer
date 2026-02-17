"""
一致性校验器单元测试
"""

import pytest
import requests
from unittest.mock import MagicMock, patch

from validator.consistency_validator import ConsistencyValidator, ValidationResult
from utils.counter import APICounter, APILimitExceeded


class TestConsistencyValidator:
    """测试一致性校验器"""

    @pytest.fixture
    def mock_config(self):
        return {
            "validator": {
                "threshold": 75.0,
                "max_rounds": 3,
                "timeout": 30,
            },
            "api_keys": {
                "kimi": "test-kimi-key",
            },
        }

    @pytest.fixture
    def mock_api_counter(self):
        return APICounter(max_calls=10, current_count=0)

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    def test_init(self, mock_config, mock_api_counter, mock_logger):
        """测试:初始化"""
        validator = ConsistencyValidator(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        assert validator.threshold == 75.0
        assert validator.max_rounds == 3
        assert validator.kimi_api_key == "test-kimi-key"

    @patch("validator.consistency_validator.requests.post")
    def test_validate_success(
        self,
        mock_post,
        mock_config,
        mock_api_counter,
        mock_logger,
    ):
        """测试:校验成功"""
        validator = ConsistencyValidator(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        # 模拟 Kimi 返回高分
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
  "total_score": 85,
  "accuracy": 38,
  "completeness": 28,
  "visualization": 15,
  "logic": 4,
  "feedback": "结构准确完整"
}
```"""
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        # 执行校验
        result = validator.validate(
            mind_map_structure=["root: 主题", "  - 节点1"],
            knowledge_doc_content="测试内容",
        )

        # 断言
        assert isinstance(result, ValidationResult)
        assert result.total_score == 85
        assert result.passed is True
        assert mock_api_counter.current_count == 1

    @patch("validator.consistency_validator.requests.post")
    def test_validate_low_score(
        self,
        mock_post,
        mock_config,
        mock_api_counter,
        mock_logger,
    ):
        """测试:校验得分低于阈值"""
        validator = ConsistencyValidator(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        # 模拟 Kimi 返回低分
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
  "total_score": 65,
  "accuracy": 25,
  "completeness": 20,
  "visualization": 15,
  "logic": 5,
  "feedback": "缺少核心知识点"
}
```"""
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        # 执行校验
        result = validator.validate(
            mind_map_structure=["root: 主题"],
            knowledge_doc_content="测试内容",
        )

        # 断言
        assert result.total_score == 65
        assert result.passed is False
        assert result.feedback == "缺少核心知识点"

    @patch("validator.consistency_validator.requests.post")
    def test_validate_api_limit(
        self,
        mock_post,
        mock_config,
        mock_api_counter,
        mock_logger,
    ):
        """测试:API 调用次数超限"""
        mock_api_counter.current_count = 10  # 已达上限

        validator = ConsistencyValidator(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        with pytest.raises(APILimitExceeded):
            validator.validate(
                mind_map_structure=["root: 主题"],
                knowledge_doc_content="测试",
            )

    @patch("validator.consistency_validator.requests.post")
    def test_validate_api_error(
        self,
        mock_post,
        mock_config,
        mock_api_counter,
        mock_logger,
    ):
        """测试:API 调用失败"""
        validator = ConsistencyValidator(
            config=mock_config,
            api_counter=mock_api_counter,
            logger=mock_logger,
        )

        # 模拟 API 错误（使用 requests.RequestException 以匹配源码的 except 子句）
        mock_post.side_effect = requests.RequestException("Network error")

        with pytest.raises(RuntimeError):
            validator.validate(
                mind_map_structure=["root: 主题"],
                knowledge_doc_content="测试",
            )
