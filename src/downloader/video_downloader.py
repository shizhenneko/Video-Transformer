"""
视频下载器模块

使用 yt-dlp 下载 Bilibili 视频,支持配置分辨率,包含完整的请求头伪装和反爬虫策略
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

import yt_dlp


class VideoDownloader:
    """Bilibili 视频下载器,支持配置分辨率,带请求头伪装和重试机制"""

    def __init__(self, config: dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        初始化视频下载器

        Args:
            config: 配置字典,包含 downloader 和 system 配置
            logger: 日志记录器,如果为 None 则创建新的
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # 从配置中提取参数
        downloader_config = config.get("downloader", {})
        system_config = config.get("system", {})

        self.retry_times = downloader_config.get("retry_times", 3)
        self.video_format = downloader_config.get("video_format", "mp4")
        self.max_resolution = downloader_config.get("max_resolution", 480)

        # 设置临时目录
        temp_dir = system_config.get("temp_dir", "./data/temp")
        self.temp_dir = Path(temp_dir) / "videos"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"视频下载器初始化完成,临时目录: {self.temp_dir}")

    def _get_ydl_opts(self) -> dict[str, Any]:
        """
        生成 yt-dlp 配置(根据配置文件设置分辨率)

        Returns:
            yt-dlp 配置字典
        """
        return {
            # 格式选择: 根据配置的最大分辨率下载
            "format": f"bestvideo[height<={self.max_resolution}][ext={self.video_format}]+bestaudio[ext=m4a]/best[height<={self.max_resolution}][ext={self.video_format}]/best[height<={self.max_resolution}]",
            # 输出模板
            "outtmpl": str(self.temp_dir / "%(id)s_%(timestamp)s.%(ext)s"),
            # 合并格式
            "merge_output_format": self.video_format,
            # 列表处理
            "noplaylist": True,  # 仅下载当前视频，不下载播放列表
            # 请求设置
            "noproxy": True,  # 禁止使用代理,避免 Bilibili 校验失败
            "headers": {
                "Referer": "https://www.bilibili.com/",
            },
            # 重试配置
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            # 速率限制(避免触发反爬虫)
            "sleep_interval": 2,
            "max_sleep_interval": 4,
            # 日志配置
            "quiet": True,
            "no_warnings": False,
            "logger": self.logger,
        }

    def download_video(self, url: str) -> Optional[str]:
        """
        下载视频,带重试机制

        Args:
            url: 视频 URL(支持 Bilibili 等平台)

        Returns:
            成功返回本地文件路径,失败返回 None
        """
        for attempt in range(1, self.retry_times + 1):
            try:
                self.logger.info(
                    f"开始下载视频 (尝试 {attempt}/{self.retry_times}): {url}"
                )

                # 重试时增加随机延迟,模拟人类行为
                if attempt > 1:
                    delay = random.uniform(3, 6)
                    self.logger.info(f"等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)

                # 获取配置
                ydl_opts = self._get_ydl_opts()

                # 下载视频
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    filename = ydl.prepare_filename(info)

                    # 验证文件
                    if self.validate_video(filename):
                        self.logger.info(f"✅ 视频下载成功: {filename}")
                        return filename
                    else:
                        self.logger.error(f"❌ 视频文件验证失败: {filename}")

            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                self.logger.error(f"下载错误 (尝试 {attempt}): {error_msg}")

                # 特殊错误处理
                if "403" in error_msg or "Forbidden" in error_msg:
                    self.logger.warning("⚠️ 检测到 403 错误,可能需要更新请求头或 Cookie")
                elif "429" in error_msg:
                    self.logger.warning("⚠️ 请求过于频繁,等待更长时间...")
                    time.sleep(10)

            except Exception as e:
                self.logger.error(f"未知错误 (尝试 {attempt}): {e}", exc_info=True)

        self.logger.error(f"❌ 视频下载失败,已达最大重试次数: {url}")
        return None

    def validate_video(self, file_path: str) -> bool:
        """
        验证视频文件完整性

        Args:
            file_path: 视频文件路径

        Returns:
            文件有效返回 True,否则返回 False
        """
        path = Path(file_path)

        # 检查文件是否存在
        if not path.exists():
            self.logger.error(f"文件不存在: {file_path}")
            return False

        # 检查文件大小(720p 视频至少应该有 500KB)
        file_size = path.stat().st_size
        min_size = 500 * 1024  # 500KB

        if file_size < min_size:
            self.logger.warning(
                f"文件过小,可能不完整: {file_size} bytes (最小: {min_size} bytes)"
            )
            return False

        self.logger.info(
            f"文件验证通过: {file_path} ({file_size / (1024 * 1024):.2f} MB)"
        )
        return True

    def cleanup_temp_files(self) -> None:
        """清理临时目录中的所有文件"""
        try:
            deleted_count = 0
            for file in self.temp_dir.glob("*"):
                if file.is_file():
                    file.unlink()
                    deleted_count += 1
                    self.logger.debug(f"已删除临时文件: {file}")

            self.logger.info(f"清理完成,共删除 {deleted_count} 个临时文件")

        except Exception as e:
            self.logger.error(f"清理临时文件失败: {e}", exc_info=True)

    def download_from_file(self, url_file: str | Path) -> dict[str, Optional[str]]:
        """
        从文件中读取 URL 列表并批量下载

        Args:
            url_file: URL 列表文件路径(每行一个 URL)

        Returns:
            字典,键为 URL,值为下载的文件路径(失败为 None)
        """
        url_path = Path(url_file)

        if not url_path.exists():
            self.logger.error(f"URL 文件不存在: {url_file}")
            return {}

        # 读取 URL 列表
        urls = []
        with url_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # 跳过空行和注释
                    urls.append(line)

        self.logger.info(f"从文件读取到 {len(urls)} 个 URL")

        # 批量下载
        results = {}
        for i, url in enumerate(urls, 1):
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"处理第 {i}/{len(urls)} 个视频")
            self.logger.info(f"{'=' * 60}")

            result = self.download_video(url)
            results[url] = result

            # 下载间隔,避免触发限流
            if i < len(urls):
                delay = random.uniform(3, 5)
                self.logger.info(f"等待 {delay:.1f} 秒后处理下一个视频...")
                time.sleep(delay)

        # 统计结果
        success_count = sum(1 for v in results.values() if v is not None)
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"批量下载完成: 成功 {success_count}/{len(urls)}")
        self.logger.info(f"{'=' * 60}")

        return results
