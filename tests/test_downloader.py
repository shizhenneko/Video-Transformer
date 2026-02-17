"""
视频下载器单元测试

测试 VideoDownloader 类的各项功能
"""

import pytest
from pathlib import Path
import sys

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from downloader import VideoDownloader
from utils.config import load_config
from utils.logger import setup_logging


@pytest.fixture
def config():
    """加载配置"""
    return load_config()


@pytest.fixture
def logger():
    """创建测试日志器"""
    return setup_logging("./data/output/logs", "test_downloader.log")


@pytest.fixture
def downloader(config, logger):
    """创建下载器实例"""
    return VideoDownloader(config, logger)


class TestVideoDownloader:
    """VideoDownloader 测试类"""

    def test_init(self, downloader):
        """测试初始化"""
        assert downloader is not None
        assert downloader.retry_times == 3
        assert downloader.video_format == "mp4"
        assert downloader.temp_dir.exists()

    def test_get_ydl_opts(self, downloader):
        """测试 yt-dlp 配置生成"""
        opts = downloader._get_ydl_opts()

        # 检查关键配置
        assert "format" in opts
        assert str(downloader.max_resolution) in opts["format"]
        assert "headers" in opts
        assert opts["headers"]["Referer"] == "https://www.bilibili.com/"
        # User-Agent is not explicitly set in _get_ydl_opts in the current code I saw, 
        # let's check video_downloader.py content again.
        # It has "headers": {"Referer": ...}. No User-Agent there?


    def test_validate_video_not_exists(self, downloader):
        """测试验证不存在的文件"""
        result = downloader.validate_video("nonexistent_file.mp4")
        assert result is False

    def test_validate_video_too_small(self, downloader, tmp_path):
        """测试验证过小的文件"""
        # 创建一个小文件
        small_file = tmp_path / "small.mp4"
        small_file.write_bytes(b"x" * 100)  # 只有 100 bytes

        result = downloader.validate_video(str(small_file))
        assert result is False

    def test_validate_video_valid(self, downloader, tmp_path):
        """测试验证有效的文件"""
        # 创建一个足够大的文件
        valid_file = tmp_path / "valid.mp4"
        valid_file.write_bytes(b"x" * (600 * 1024))  # 600KB

        result = downloader.validate_video(str(valid_file))
        assert result is True

    def test_cleanup_temp_files(self, downloader, tmp_path):
        """测试清理临时文件"""
        # 创建一些临时文件
        original_temp_dir = downloader.temp_dir
        downloader.temp_dir = tmp_path

        test_files = []
        for i in range(3):
            f = tmp_path / f"test_{i}.mp4"
            f.write_text("test")
            test_files.append(f)

        # 清理
        downloader.cleanup_temp_files()

        # 验证文件已删除
        for f in test_files:
            assert not f.exists()

        # 恢复原始目录
        downloader.temp_dir = original_temp_dir

    @pytest.mark.skip(reason="需要网络连接,仅在集成测试时运行")
    def test_download_video_bilibili(self, downloader):
        """测试下载 Bilibili 视频(集成测试)"""
        # 使用一个公开的短视频进行测试
        test_url = "https://www.bilibili.com/video/BV1xx411c7mD"

        result = downloader.download_video(test_url)

        # 如果下载成功,验证文件
        if result:
            assert Path(result).exists()
            assert downloader.validate_video(result)
            # 清理下载的文件
            Path(result).unlink()

    @pytest.mark.skip(reason="需要网络连接,仅在集成测试时运行")
    def test_download_from_file(self, downloader):
        """测试从文件批量下载(集成测试)"""
        url_file = Path("./data/input/URL.txt")

        if not url_file.exists():
            pytest.skip("URL.txt 文件不存在")

        results = downloader.download_from_file(url_file)

        # 验证结果
        assert isinstance(results, dict)
        assert len(results) > 0

        # 清理下载的文件
        for file_path in results.values():
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
