"""
Gemini API 限流器

在每次 Gemini API 调用前主动限速，遇到 429 时耐心等待并自动重试。
通过全局限流 + 智能重试，规避 Google API 的速率/配额限制。
"""

import ast
import json
import logging
import random
import re
import threading
import time
from typing import Any, Callable, Dict, Optional


class GeminiThrottle:
    """
    Gemini API 限流器

    功能:
    - 每次 generate_content 调用前自动等待 min_interval 秒，避免请求过密
    - 文件操作 (upload/get/delete) 前等待 files_interval 秒，与 generate 共享 RPM 窗口
    - 遇到 429 错误时，解析 retryDelay 并耐心等待
    - 最多重试 max_retries 次，总等待不超过 max_total_wait 秒
    - 线程安全
    """

    def __init__(
        self,
        min_interval: float = 10.0,
        files_interval: float = 3.0,
        max_retries: int = 10,
        max_total_wait: float = 600.0,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            min_interval: 两次 generate_content 调用之间的最小间隔(秒)
            files_interval: 两次文件操作 (upload/get/delete) 之间的最小间隔(秒)
            max_retries: 遇到 429 时的最大重试次数
            max_total_wait: 重试总等待时间上限(秒)
            logger: 日志记录器
        """
        self.min_interval = min_interval
        self.files_interval = files_interval
        self.max_retries = max_retries
        self.max_total_wait = max_total_wait
        self.logger = logger or logging.getLogger(__name__)

        self._last_call_time: float = 0.0
        self._lock = threading.Lock()

    def wait_before_call(self) -> None:
        """在 generate_content 调用前等待，确保与上次任意 Gemini 操作的间隔 >= min_interval"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.min_interval:
                wait = self.min_interval - elapsed
                self.logger.info(
                    f"⏳ 限流等待 {wait:.1f}s (最小间隔 {self.min_interval}s)..."
                )
                time.sleep(wait)
            self._last_call_time = time.time()

    def wait_for_files_op(self) -> None:
        """在文件操作 (upload/get/delete) 前等待，确保与上次任意 Gemini 操作的间隔 >= files_interval"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call_time
            if elapsed < self.files_interval:
                wait = self.files_interval - elapsed
                self.logger.info(
                    f"⏳ 文件操作限速等待 {wait:.1f}s (间隔 {self.files_interval}s)..."
                )
                time.sleep(wait)
            self._last_call_time = time.time()

    def call_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        on_retry_callback: Optional[Callable[[int, Exception], None]] = None,
        log_context: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        带限流和 429 自动重试的调用包装

        Args:
            func: 要调用的函数
            *args: 函数参数
            on_retry_callback: 每次重试前的回调, 接收 (attempt, exception)
            **kwargs: 函数关键字参数

        Returns:
            func 的返回值

        Raises:
            最后一次失败的异常 (如果重试耗尽)
        """
        total_waited = 0.0
        last_error: Optional[Exception] = None
        context = log_context or {}
        endpoint = context.get("endpoint", "unknown")
        model = context.get("model", "unknown")
        key_id = context.get("key_id", "unknown")

        for attempt in range(1, self.max_retries + 1):
            # 主动限速
            self.wait_before_call()

            try:
                self.logger.info(
                    f"尝试调用 Gemini API (第 {attempt}/{self.max_retries} 次)"
                )
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                error_str = str(e)

                # 判断是否为 429 / 配额耗尽
                if not self._is_rate_limit_error(error_str):
                    # 非限流错误，不在此层重试，直接抛出
                    raise

                wait_time = self._extract_retry_delay(
                    error_str, attempt=attempt, base_delay=30.0
                )
                timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                error_type = type(e).__name__
                self.logger.warning(
                    "⚠️ 429 detected | "
                    f"timestamp={timestamp} | "
                    f"endpoint={endpoint} | model={model} | key_id={key_id} | "
                    f"attempt={attempt} | status_code=429 | "
                    f"retry_after={wait_time:.1f} | delay={wait_time:.1f}s | "
                    f"error_type={error_type} | error={error_str[:200]}"
                )

                self.logger.warning(
                    f"⚠️ 429 Rate Limit | attempt={attempt}/{self.max_retries} | "
                    + f"wait={wait_time:.1f}s | error={error_str[:300]}"
                )

                # 回调通知
                if on_retry_callback:
                    on_retry_callback(attempt, e)

                # 已达最大重试次数
                if attempt >= self.max_retries:
                    break

                # 检查是否超过总等待上限
                if total_waited + wait_time > self.max_total_wait:
                    remaining = self.max_total_wait - total_waited
                    if remaining <= 0:
                        self.logger.error(
                            f"❌ 已超过总等待时间上限 ({self.max_total_wait}s)，放弃重试"
                        )
                        break
                    wait_time = remaining
                    self.logger.warning(
                        f"⏳ 等待时间截断为 {wait_time:.1f}s (即将达到总上限 {self.max_total_wait}s)"
                    )

                self.logger.info(
                    f"⏳ 等待 {wait_time:.1f}s 后重试... "
                    f"(已累计等待 {total_waited:.0f}s / 上限 {self.max_total_wait:.0f}s)"
                )
                time.sleep(wait_time)
                total_waited += wait_time

        raise RuntimeError(
            f"Gemini API 调用失败，已重试 {self.max_retries} 次 "
            f"(累计等待 {total_waited:.0f}s): {last_error}"
        )

    @staticmethod
    def _is_rate_limit_error(error_str: str) -> bool:
        """判断是否为速率/配额限制错误"""
        lower = error_str.lower()
        indicators = [
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
        ]
        return any(ind in lower for ind in indicators)

    @staticmethod
    def _extract_retry_delay(
        error_str: str,
        default: float = 60.0,
        attempt: int = 1,
        base_delay: float = 30.0,
    ) -> float:
        """
        从错误信息中提取 API 建议的重试等待时间

        支持格式:
        - "retry in 42.631938741s"
        - "retryDelay": "42s"
        """
        lower = error_str.lower()

        def coerce_seconds(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                match = re.match(r"^(\d+(?:\.\d+)?)s$", stripped)
                if match:
                    return float(match.group(1))
                if stripped.replace(".", "", 1).isdigit():
                    return float(stripped)
            return None

        def parse_retry_delay_value(value: Any) -> Optional[float]:
            if isinstance(value, dict):
                seconds = value.get("seconds")
                nanos = value.get("nanos", 0)
                if seconds is None:
                    return None
                return float(seconds) + float(nanos) / 1_000_000_000
            return coerce_seconds(value)

        def find_retry_after(obj: Any) -> Optional[float]:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_lower = str(key).lower()
                    if key_lower in {"retry-after", "retry_after", "retryafter"}:
                        retry_after = coerce_seconds(value)
                        if retry_after is not None:
                            return retry_after
                    if key_lower == "retrydelay":
                        retry_delay = parse_retry_delay_value(value)
                        if retry_delay is not None:
                            return retry_delay
                    nested = find_retry_after(value)
                    if nested is not None:
                        return nested
            elif isinstance(obj, list):
                for item in obj:
                    nested = find_retry_after(item)
                    if nested is not None:
                        return nested
            return None

        retry_after = None
        trimmed = error_str.strip()
        if trimmed.startswith("{") or trimmed.startswith("["):
            try:
                retry_after = find_retry_after(json.loads(trimmed))
            except json.JSONDecodeError:
                try:
                    retry_after = find_retry_after(ast.literal_eval(trimmed))
                except (ValueError, SyntaxError):
                    retry_after = None
        if retry_after is None:
            brace_start = error_str.find("{")
            brace_end = error_str.rfind("}")
            if brace_start != -1 and brace_end > brace_start:
                candidate = error_str[brace_start : brace_end + 1]
                try:
                    retry_after = find_retry_after(json.loads(candidate))
                except json.JSONDecodeError:
                    try:
                        retry_after = find_retry_after(ast.literal_eval(candidate))
                    except (ValueError, SyntaxError):
                        retry_after = None

        if retry_after is None:
            match = re.search(r"retry-after\s*[:=]\s*(\d+(?:\.\d+)?)s?", lower)
            if match:
                retry_after = float(match.group(1))

        if retry_after is None:
            match = re.search(r"retry in (\d+(?:\.\d+)?)s", lower)
            if match:
                retry_after = float(match.group(1))

        if retry_after is None:
            match = re.search(r"retrydelay.*?(\d+(?:\.\d+)?)s", lower)
            if match:
                retry_after = float(match.group(1))

        wait_time = retry_after
        if wait_time is None:
            if attempt < 1 or base_delay <= 0:
                wait_time = default
            else:
                wait_time = base_delay * (2**attempt)

        # Add jitter to prevent thundering herd
        wait_time = wait_time * (0.9 + random.random() * 0.2)

        return max(wait_time, 0.0)
