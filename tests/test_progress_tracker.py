"""
进度追踪器测试
"""

import json
import pytest
from pathlib import Path
from utils.progress_tracker import ProgressTracker
from utils.logger import setup_logging


class TestProgressTracker:
    """ProgressTracker 单元测试"""

    @pytest.fixture
    def temp_progress_file(self, tmp_path):
        """临时进度文件"""
        return tmp_path / "progress.json"

    @pytest.fixture
    def logger(self, tmp_path):
        """测试日志"""
        return setup_logging(str(tmp_path), "test.log")

    def test_initialization_new_file(self, temp_progress_file, logger):
        """测试初始化新文件"""
        tracker = ProgressTracker(temp_progress_file, logger)

        assert tracker.progress_file.exists()
        assert len(tracker.data["processed"]) == 0
        assert len(tracker.data["failed"]) == 0

    def test_mark_processed(self, temp_progress_file, logger):
        """测试标记已处理"""
        tracker = ProgressTracker(temp_progress_file, logger)

        tracker.mark_processed("BV1xx411c7mD")

        assert tracker.is_processed("BV1xx411c7mD")
        assert not tracker.is_failed("BV1xx411c7mD")

    def test_mark_failed(self, temp_progress_file, logger):
        """测试标记失败"""
        tracker = ProgressTracker(temp_progress_file, logger)

        tracker.mark_failed("BV1xx411c7mD", "下载失败")

        assert tracker.is_failed("BV1xx411c7mD")
        assert "BV1xx411c7mD" in tracker.get_failed_videos()

    def test_filter_unprocessed(self, temp_progress_file, logger):
        """测试过滤未处理"""
        tracker = ProgressTracker(temp_progress_file, logger)

        tracker.mark_processed("vid1")
        tracker.mark_processed("vid2")

        all_videos = ["vid1", "vid2", "vid3", "vid4"]
        unprocessed = tracker.filter_unprocessed(all_videos)

        assert len(unprocessed) == 2
        assert "vid3" in unprocessed
        assert "vid4" in unprocessed

    def test_persistence(self, temp_progress_file, logger):
        """测试持久化"""
        # 第一个 tracker
        tracker1 = ProgressTracker(temp_progress_file, logger)
        tracker1.mark_processed("vid1")
        tracker1.mark_failed("vid2", "错误信息")

        # 第二个 tracker (重新加载)
        tracker2 = ProgressTracker(temp_progress_file, logger)

        assert tracker2.is_processed("vid1")
        assert tracker2.is_failed("vid2")

    def test_reset(self, temp_progress_file, logger):
        """测试重置"""
        tracker = ProgressTracker(temp_progress_file, logger)

        tracker.mark_processed("vid1")
        tracker.mark_failed("vid2", "错误")

        tracker.reset()

        assert len(tracker.data["processed"]) == 0
        assert len(tracker.data["failed"]) == 0

    def test_get_statistics(self, temp_progress_file, logger):
        """测试统计信息"""
        tracker = ProgressTracker(temp_progress_file, logger)

        tracker.mark_processed("vid1")
        tracker.mark_processed("vid2")
        tracker.mark_failed("vid3", "错误")

        stats = tracker.get_statistics()

        assert stats["processed_count"] == 2
        assert stats["failed_count"] == 1
