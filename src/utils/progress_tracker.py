"""
进度追踪模块

支持断点续传,记录已处理和失败的视频
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, progress_file: str | Path, logger: logging.Logger):
        """
        初始化进度追踪器

        Args:
            progress_file: 进度文件路径
            logger: 日志记录器
        """
        self.progress_file = Path(progress_file)
        self.logger = logger

        # 确保目录存在
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        # 加载或初始化进度数据
        self.data = self._load_progress()

    def _load_progress(self) -> dict[str, Any]:
        """加载进度数据"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.logger.info(
                        f"已加载进度文件: {len(data.get('processed', []))} 个已处理, "
                        f"{len(data.get('failed', {}))} 个失败"
                    )
                    return data
            except Exception as e:
                self.logger.warning(f"加载进度文件失败: {e}, 使用空进度")

        # 初始化空进度
        data = {"processed": [], "failed": {}, "last_updated": None}
        
        # 立即保存以创建文件
        try:
            data["last_updated"] = datetime.now().isoformat()
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"创建进度文件失败: {e}")
            
        return data

    def _save_progress(self) -> None:
        """保存进度数据"""
        try:
            self.data["last_updated"] = datetime.now().isoformat()
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"进度已保存到 {self.progress_file}")
        except Exception as e:
            self.logger.error(f"保存进度失败: {e}")

    def is_processed(self, video_id: str) -> bool:
        """检查视频是否已处理"""
        return video_id in self.data["processed"]

    def is_failed(self, video_id: str) -> bool:
        """检查视频是否曾失败"""
        return video_id in self.data["failed"]

    def mark_processed(self, video_id: str) -> None:
        """标记视频为已处理"""
        if video_id not in self.data["processed"]:
            self.data["processed"].append(video_id)
            # 从失败列表中移除(如果存在)
            self.data["failed"].pop(video_id, None)
            self._save_progress()
            self.logger.info(f"已标记 {video_id} 为处理完成")

    def mark_failed(self, video_id: str, error_message: str) -> None:
        """标记视频为失败"""
        self.data["failed"][video_id] = {
            "error": error_message,
            "timestamp": datetime.now().isoformat(),
        }
        self._save_progress()
        self.logger.warning(f"已标记 {video_id} 为处理失败: {error_message}")

    def get_failed_videos(self) -> dict[str, dict[str, str]]:
        """获取所有失败的视频"""
        return self.data["failed"]

    def filter_unprocessed(self, video_ids: list[str]) -> list[str]:
        """
        过滤出未处理的视频

        Args:
            video_ids: 视频 ID 列表

        Returns:
            未处理的视频 ID 列表
        """
        unprocessed = [vid for vid in video_ids if not self.is_processed(vid)]

        if len(unprocessed) < len(video_ids):
            skipped = len(video_ids) - len(unprocessed)
            self.logger.info(f"跳过 {skipped} 个已处理视频")

        return unprocessed

    def reset(self) -> None:
        """重置进度(清空所有记录)"""
        self.data = {"processed": [], "failed": {}, "last_updated": None}
        self._save_progress()
        self.logger.info("进度已重置")

    def get_statistics(self) -> dict[str, int]:
        """获取统计信息"""
        return {
            "processed_count": len(self.data["processed"]),
            "failed_count": len(self.data["failed"]),
        }
