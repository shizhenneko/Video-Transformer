"""
数据模型测试
"""

import pytest
from models import BatchResult, ProcessResult


class TestProcessResult:
    """ProcessResult 单元测试"""

    def test_create_success_result(self):
        """测试创建成功结果"""
        result = ProcessResult(
            video_id="BV1xx411c7mD",
            url="https://test.com/video",
            success=True,
            document_path="/path/to/doc.md",
            blueprint_path="/path/to/blueprint.png",
            api_calls_used=5,
            processing_time=120.5,
        )

        assert result.video_id == "BV1xx411c7mD"
        assert result.success is True
        assert result.api_calls_used == 5
        assert result.error_message is None

    def test_create_failure_result(self):
        """测试创建失败结果"""
        result = ProcessResult(
            video_id="BV1xx411c7mD",
            url="https://test.com/video",
            success=False,
            error_message="下载失败",
            processing_time=10.0,
        )

        assert result.success is False
        assert result.error_message == "下载失败"
        assert result.document_path is None

    def test_str_representation(self):
        """测试字符串表示"""
        result = ProcessResult(
            video_id="BV1xx411c7mD",
            url="https://test.com/video",
            success=True,
            api_calls_used=3,
            processing_time=60.0,
        )

        str_repr = str(result)
        assert "BV1xx411c7mD" in str_repr
        assert "成功" in str_repr
        assert "3" in str_repr


class TestBatchResult:
    """BatchResult 单元测试"""

    def test_create_empty_batch(self):
        """测试创建空批量结果"""
        result = BatchResult(total=5, successful=0, failed=0)

        assert result.total == 5
        assert result.successful == 0
        assert result.failed == 0
        assert len(result.results) == 0

    def test_add_result(self):
        """测试添加结果"""
        batch = BatchResult(total=2, successful=0, failed=0)

        result1 = ProcessResult(
            video_id="vid1",
            url="url1",
            success=True,
            api_calls_used=3,
            processing_time=60.0,
        )

        result2 = ProcessResult(
            video_id="vid2",
            url="url2",
            success=True,
            api_calls_used=2,
            processing_time=45.0,
        )

        batch.add_result(result1)
        batch.add_result(result2)

        assert len(batch.results) == 2
        assert batch.total_api_calls == 5
        assert batch.total_time == 105.0

    def test_to_dict(self):
        """测试转换为字典"""
        batch = BatchResult(total=1, successful=1, failed=0)

        result = ProcessResult(
            video_id="vid1",
            url="url1",
            success=True,
            api_calls_used=3,
            processing_time=60.0,
        )

        batch.add_result(result)
        data = batch.to_dict()

        assert data["total"] == 1
        assert data["successful"] == 1
        assert len(data["results"]) == 1
        assert data["results"][0]["video_id"] == "vid1"

    def test_str_representation(self):
        """测试字符串表示"""
        batch = BatchResult(total=10, successful=8, failed=2, total_api_calls=50)

        str_repr = str(batch)
        assert "8/10" in str_repr
        assert "80" in str_repr  # 成功率
