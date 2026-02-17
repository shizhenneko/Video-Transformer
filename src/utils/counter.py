from __future__ import annotations

from dataclasses import dataclass


class APILimitExceeded(RuntimeError):
    pass


@dataclass
class APICounter:
    max_calls: int = 20  # Default to 20 for Gemini
    current_count: int = 0
    hard_max_calls: int | None = None

    def _effective_max_calls(self) -> int:
        if self.hard_max_calls is None:
            return self.max_calls
        return min(self.max_calls, self.hard_max_calls)

    def set_max_calls(self, max_calls: int, hard_max_calls: int | None = None) -> int:
        if hard_max_calls is not None:
            self.hard_max_calls = hard_max_calls
        effective_hard = self.hard_max_calls
        if effective_hard is None:
            effective_hard = max_calls
        self.max_calls = min(max_calls, effective_hard)
        return self.max_calls

    def increase_max_calls(
        self, additional_calls: int, hard_max_calls: int | None = None
    ) -> int:
        return self.set_max_calls(self.max_calls + additional_calls, hard_max_calls)

    def increment(self, service: str) -> bool:
        """
        增加 API 调用计数

        Args:
            service: 服务名称 (如 'Gemini', 'Kimi', 'NanoBanana')

        Returns:
            bool: 是否成功增加计数

        Raises:
            APILimitExceeded: 如果调用次数超限 (仅针对 Gemini)
        """
        # 仅对 Gemini 服务进行计数限制
        if service.lower() == "gemini":
            limit = self._effective_max_calls()
            if self.current_count >= limit:
                raise APILimitExceeded(
                    f"Gemini API call limit reached: {self.current_count}/{limit}"
                )
            self.current_count += 1
            return True

        # 其他服务不计入限制
        return True

    def can_call(self) -> bool:
        # 总是返回 True，具体的限制检查在 increment 中进行
        # 或者保留原逻辑但仅对 Gemini 有效？
        # 鉴于 pipeline.py 中有 usage checking: if not self.api_counter.can_call(): ...
        # 我们需要保持 can_call 的语义，即 "能否继续调用 Gemini"。
        # 但如果 pipeline 询问 can_call() 仅仅是为了检查全局状态，那么它应该反映 Gemini 的状态。
        return self.current_count < self._effective_max_calls()

    def remaining(self) -> int:
        return max(self._effective_max_calls() - self.current_count, 0)

    def reset(self) -> None:
        self.current_count = 0
