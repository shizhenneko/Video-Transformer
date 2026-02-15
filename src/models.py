"""
统一数据模型定义

定义系统中使用的核心数据结构
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessResult:
    """单个视频处理结果"""

    video_id: str
    """视频唯一标识符"""

    url: str
    """视频源 URL"""

    success: bool
    """处理是否成功"""

    document_path: str | None = None
    """生成的知识笔记文档路径"""

    blueprint_path: str | None = None
    """生成的知识蓝图图片路径"""

    api_calls_used: int = 0
    """此视频消耗的 API 调用次数"""

    error_message: str | None = None
    """错误信息(如果失败)"""

    processing_time: float = 0.0
    """处理耗时(秒)"""

    validation_score: float = 0.0
    """校验分数"""

    audit_score: float = 0.0
    """审核分数"""

    def __str__(self) -> str:
        """字符串表示"""
        status = "✅ 成功" if self.success else "❌ 失败"
        return (
            f"{status} | {self.video_id} | "
            f"API调用: {self.api_calls_used} | "
            f"耗时: {self.processing_time:.1f}s"
        )


@dataclass
class BatchResult:
    """批量处理结果"""

    total: int
    """总视频数"""

    successful: int
    """成功处理数"""

    failed: int
    """失败处理数"""

    results: list[ProcessResult] = field(default_factory=list)
    """详细结果列表"""

    total_api_calls: int = 0
    """总 API 调用次数"""

    total_time: float = 0.0
    """总耗时(秒)"""

    def add_result(self, result: ProcessResult) -> None:
        """添加处理结果"""
        self.results.append(result)
        self.total_api_calls += result.api_calls_used
        self.total_time += result.processing_time

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "total_api_calls": self.total_api_calls,
            "total_time": self.total_time,
            "results": [
                {
                    "video_id": r.video_id,
                    "url": r.url,
                    "success": r.success,
                    "document_path": r.document_path,
                    "blueprint_path": r.blueprint_path,
                    "api_calls_used": r.api_calls_used,
                    "error_message": r.error_message,
                    "processing_time": r.processing_time,
                    "validation_score": r.validation_score,
                    "audit_score": r.audit_score,
                }
                for r in self.results
            ],
        }

    def __str__(self) -> str:
        """字符串表示"""
        success_rate = (self.successful / self.total * 100) if self.total > 0 else 0
        return (
            f"批量处理结果: {self.successful}/{self.total} 成功 "
            f"({success_rate:.1f}%) | "
            f"API调用: {self.total_api_calls} | "
            f"总耗时: {self.total_time:.1f}s"
        )
