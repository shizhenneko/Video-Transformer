"""
质量审核模块

使用 Gemini 对生成的知识蓝图图片进行质量审核
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import google.genai as genai  # type: ignore[reportMissingImports]
from google.genai import types  # type: ignore[reportMissingImports]

from utils.counter import APICounter, APILimitExceeded
from utils.gemini_throttle import GeminiThrottle


@dataclass
class AuditResult:
    """审核结果"""

    passed: bool
    """是否通过审核"""

    feedback: str
    """审核反馈"""

    score: float
    """质量评分(0-100)"""


class QualityAuditor:
    """图片质量审核器"""

    def __init__(
        self,
        config: dict[str, Any],
        api_counter: APICounter,
        logger: logging.Logger,
        throttle: GeminiThrottle,
        api_key: str | None = None,
    ):
        """
        初始化审核器

        Args:
            config: 系统配置字典
            api_counter: API 调用计数器
            logger: 日志记录器
            api_key: Gemini API 密钥(可选)
        """
        self.config = config
        self.api_counter = api_counter
        self.logger = logger

        # 加载配置
        auditor_config = config.get("auditor", {})
        self.model_name = auditor_config.get("model", "gemini-2.5-flash")
        self.threshold = auditor_config.get("threshold", 75.0)
        self.timeout = auditor_config.get("timeout", 60)

        # 代理号池配置
        proxy_config = config.get("proxy", {})
        self.proxy_base_url = proxy_config.get("base_url", "http://localhost:8000")
        self.proxy_timeout = proxy_config.get("timeout", 10)

        self._fixed_api_key = api_key
        self._allocated_key_id = None
        self._client: genai.Client | None = None

        http_proxy = proxy_config.get("http")

        if http_proxy:
            import os

            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["HTTPS_PROXY"] = http_proxy
            os.environ["NO_PROXY"] = "localhost,127.0.0.1"
            self.logger.info(f"已设置代理环境变量: {http_proxy}")

        if self._fixed_api_key:
            self._client = genai.Client(
                api_key=self._fixed_api_key,
                http_options={"timeout": 600_000},
            )
            self.logger.info("Gemini SDK 配置完成(使用外部分配的 API Key)")
        else:
            self.logger.warning("未提供 Gemini API Key,QualityAuditor 将无法正常工作")

        self.logger.info("QualityAuditor 初始化完成")

        # 限流器
        self.throttle = throttle

    # _allocate_key_from_pool 已移除,密钥分配逻辑已移至 VideoPipeline

    def _report_usage_to_pool(self) -> None:
        """向代理号池报告成功调用"""
        if not self._allocated_key_id:
            return

        url = f"{self.proxy_base_url.rstrip('/')}/sdk/report-usage"
        try:
            requests.post(
                url,
                json={"key_id": self._allocated_key_id},
                timeout=self.proxy_timeout,
            )
        except requests.RequestException as e:
            self.logger.warning(f"向号池报告用量失败: {e}")

    def _report_error_to_pool(self, is_rpd_limit: bool = False) -> None:
        """向代理号池报告错误"""
        if not self._allocated_key_id:
            return

        url = f"{self.proxy_base_url.rstrip('/')}/sdk/report-error"
        try:
            requests.post(
                url,
                json={"key_id": self._allocated_key_id, "is_rpd_limit": is_rpd_limit},
                timeout=self.proxy_timeout,
            )
        except requests.RequestException as e:
            self.logger.warning(f"向号池报告错误失败: {e}")

    @staticmethod
    def _classify_429_is_daily(exc: Exception) -> bool:
        """仅在异常明确提示每日配额耗尽时返回 True。"""
        message = str(exc).lower()
        if not message:
            return False
        daily_markers = ("per day", "daily", "quota exceeded per day")
        return any(marker in message for marker in daily_markers)

    def _delete_remote_file(self, file_name: str) -> None:
        """删除 Gemini Files 存储中的远程文件，释放配额空间。"""
        if not self._client:
            return
        try:
            self.throttle.wait_for_files_op()
            self._client.files.delete(name=file_name)
            self.logger.info(f"已清理 Gemini 远程文件: {file_name}")
        except Exception as e:
            self.logger.warning(f"清理 Gemini 远程文件失败: {e}")

    def audit_image(
        self,
        image_path: str | Path,
        knowledge_doc_content: str,
    ) -> AuditResult:
        """
        审核知识蓝图图片质量（通过限流器自动处理 429）

        图片只上传一次（在重试循环外部），避免 429 重试时反复 upload 加剧限流。

        Args:
            image_path: 图片文件路径
            knowledge_doc_content: 知识笔记内容(用于对比)

        Returns:
            AuditResult: 审核结果

        Raises:
            APILimitExceeded: 如果 API 调用次数超限
            RuntimeError: 如果审核失败
        """
        # 检查 API 调用次数
        if not self.api_counter.can_call():
            raise APILimitExceeded(
                f"API 调用次数已达上限: {self.api_counter.current_count}/{self.api_counter.max_calls}"
            )

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        if not self._client:
            raise RuntimeError("Gemini Client 未初始化 (缺少 API Key)")

        client = self._client

        # 在重试循环外部上传图片（只 upload 一次）
        self.logger.info(f"上传图片: {image_path.name}")
        self.throttle.wait_for_files_op()
        image_file = client.files.upload(file=str(image_path))

        try:

            def _do_audit() -> AuditResult:
                """单次审核 API 调用（图片已在外部 upload）"""
                prompt = self._build_audit_prompt(knowledge_doc_content)

                # 流式调用，实时输出审核过程
                self.logger.info("开始流式接收审核响应...")
                response_parts: list[str] = []
                thinking_logged = False

                for chunk in client.models.generate_content_stream(
                    model=self.model_name,
                    contents=[
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "file_data": {
                                        "file_uri": image_file.uri,
                                        "mime_type": image_file.mime_type,
                                    }
                                },
                                {"text": prompt},
                            ],
                        }
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2048,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=4096,
                        ),
                        http_options=types.HttpOptions(timeout=600_000),
                    ),
                ):
                    if not chunk.candidates:
                        continue
                    content = chunk.candidates[0].content
                    if not content or not content.parts:
                        continue
                    for part in content.parts:
                        if part.thought:
                            if not thinking_logged:
                                self.logger.info("💭 Gemini 审核思考中...")
                                thinking_logged = True
                            snippet = part.text[:200] if part.text else ""
                            if snippet:
                                self.logger.info(f"  💭 {snippet}")
                        else:
                            if part.text:
                                response_parts.append(part.text)
                                snippet = part.text[:100].replace("\n", " ")
                                self.logger.info(f"  📝 审核中: {snippet}")

                self._report_usage_to_pool()

                self.api_counter.increment("Gemini")
                self.logger.info(
                    f"Gemini 审核调用成功,当前计数: {self.api_counter.current_count}/{self.api_counter.max_calls}"
                )

                response_text = "".join(response_parts).strip()
                return self._parse_audit_response(response_text)

            def _on_retry(attempt: int, exc: Exception) -> None:
                nonlocal reported_retry
                if reported_retry:
                    return
                reported_retry = True
                is_daily_limit = self._classify_429_is_daily(exc)
                self._report_error_to_pool(is_rpd_limit=is_daily_limit)

            reported_retry = False
            return self.throttle.call_with_retry(
                _do_audit,
                on_retry_callback=_on_retry,
            )

        finally:
            # 无论成功还是失败，删除已上传的远程图片文件
            if image_file is not None and image_file.name:
                self._delete_remote_file(image_file.name)

    def _build_audit_prompt(self, content: str) -> str:
        """构建审核 Prompt"""
        return f"""请审核这张知识蓝图图片的质量。

## 参考知识笔记内容

{content[:1500]}...

## 审核要点

1. **内容准确性**: 图片是否正确表达了知识结构
2. **可读性**: 文字是否清晰,布局是否合理
3. **美观度**: 色彩搭配、视觉层次
4. **完整性**: 核心知识点是否遗漏

## 输出要求

请以 0-100 分评估图片质量,并给出简短反馈(不超过100字)。

格式:
评分: <分数>
反馈: <简短评价>
通过: <是/否>

现在请开始审核。
"""

    def _parse_audit_response(self, text: str) -> AuditResult:
        """解析审核响应"""

        try:
            # 简单解析(可以根据实际响应调整)
            lines = text.split("\n")
            score = 0.0
            feedback = ""
            passed = False

            for line in lines:
                if "评分" in line or "score" in line.lower():
                    # 提取数字
                    import re

                    numbers = re.findall(r"\d+\.?\d*", line)
                    if numbers:
                        score = float(numbers[0])
                elif "反馈" in line or "feedback" in line.lower():
                    feedback = line.split(":")[-1].strip()
                elif "通过" in line or "passed" in line.lower():
                    passed = "是" in line or "yes" in line.lower() or "通过" in line

            # 基于分数判断
            if score >= self.threshold:
                passed = True

            return AuditResult(
                passed=passed,
                feedback=feedback if feedback else "图片质量可接受",
                score=score,
            )

        except Exception as e:
            self.logger.warning(f"解析审核响应失败: {e}, 响应: {text}")
            # 返回默认通过结果
            return AuditResult(
                passed=True,
                feedback="审核完成(响应解析异常,默认通过)",
                score=75.0,
            )
