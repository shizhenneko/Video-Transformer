"""
一致性校验模块

使用 Kimi K2 API 对知识蓝图结构进行质量评估
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from utils.counter import APICounter, APILimitExceeded


@dataclass
class ValidationResult:
    """校验结果数据类"""

    total_score: float
    """总分(0-100)"""

    accuracy: float
    """准确性得分(满分40)"""

    completeness: float
    """完整性得分(满分30)"""

    visualization: float
    """可视化适配性得分(满分20)"""

    logic: float
    """逻辑性得分(满分10)"""

    feedback: str
    """改进建议"""

    passed: bool
    """是否通过(>= 阈值)"""


class ConsistencyValidator:
    """
    知识蓝图一致性校验器

    使用 Kimi K2 API 评估蓝图结构的质量
    """

    def __init__(
        self,
        config: dict[str, Any],
        api_counter: APICounter,
        logger: logging.Logger,
        api_key: str | None = None,
    ):
        """
        初始化校验器

        Args:
            config: 系统配置字典
            api_counter: API 调用计数器
            logger: 日志记录器
            api_key: Kimi API 密钥(可选,若为 None 则从配置读取)
        """
        self.config = config
        self.api_counter = api_counter
        self.logger = logger

        # 加载校验器配置
        validator_config = config.get("validator", {})
        self.threshold = validator_config.get("threshold", 75.0)
        self.max_rounds = validator_config.get("max_rounds", 3)
        self.timeout = validator_config.get("timeout", 30)

        # Kimi API 配置
        self.kimi_api_key = api_key or config.get("api_keys", {}).get("kimi", "")
        if not self.kimi_api_key:
            self.logger.warning("Kimi API Key 未配置,校验功能将不可用")

        self.kimi_base_url = "https://api.moonshot.cn/v1"
        self.kimi_model = "kimi-k2-0905-preview"

        self.logger.info("ConsistencyValidator 初始化完成")

    def validate(
        self,
        mind_map_structure: list[str] | str,
        knowledge_doc_content: str,
    ) -> ValidationResult:
        """
        执行一致性校验

        Args:
            mind_map_structure: 知识蓝图结构(Markdown列表)
            knowledge_doc_content: 原始知识笔记内容

        Returns:
            ValidationResult: 校验结果对象

        Raises:
            APILimitExceeded: 如果 API 调用次数超限
            RuntimeError: 如果 API 调用失败
        """
        # 检查 API 调用次数
        if not self.api_counter.can_call():
            raise APILimitExceeded(
                f"API 调用次数已达上限: {self.api_counter.current_count}/{self.api_counter.max_calls}"
            )

        # 构建校验 Prompt
        structure_text = (
            "\n".join(mind_map_structure)
            if isinstance(mind_map_structure, list)
            else mind_map_structure
        )

        prompt = self._build_validation_prompt(structure_text, knowledge_doc_content)

        # 调用 Kimi API
        try:
            response_data = self._call_kimi_api(prompt)

            # 增加 API 计数
            self.api_counter.increment("Kimi")
            self.logger.info(
                f"Kimi API 调用成功,当前计数: {self.api_counter.current_count}/{self.api_counter.max_calls}"
            )

            # 解析响应
            result = self._parse_kimi_response(response_data)
            return result

        except Exception as e:
            self.logger.error(f"校验失败: {e}")
            raise

    def _build_validation_prompt(self, structure: str, content: str) -> str:
        """构建校验 Prompt"""
        return f"""请评估以下"知识蓝图结构"与"原始内容"的吻合度。

## 评分标准(总分100分)

1. **准确性(40分)**: 结构是否准确反映原内容的核心知识点
2. **完整性(30分)**: 是否涵盖所有核心知识点,无重要遗漏
3. **可视化适配性(20分)**: 结构是否清晰,适合转化为可视化图表
4. **逻辑性(10分)**: 知识点之间的层级关系是否合理

## 原始知识笔记内容

{content[:2000]}...

## 知识蓝图结构

```markdown
{structure}
```

## 输出要求

请以 JSON 格式输出评估结果(不要包含其他文本):

```json
{{
  "total_score": <总分(0-100)>,
  "accuracy": <准确性得分(0-40)>,
  "completeness": <完整性得分(0-30)>,
  "visualization": <可视化得分(0-20)>,
  "logic": <逻辑性得分(0-10)>,
  "feedback": "<具体改进建议,如果得分>=75则填'结构准确完整'>"
}}
```

现在请开始评估。
"""

    def _call_kimi_api(self, prompt: str) -> dict[str, Any]:
        """调用 Kimi API 进行校验"""

        url = f"{self.kimi_base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.kimi_api_key}",
        }

        payload = {
            "model": self.kimi_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一位专业的知识结构评审员,擅长评估知识蓝图的准确性和完整性。",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,  # 降低随机性,提高评分一致性
        }

        try:
            self.logger.info("调用 Kimi API 进行校验...")
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as e:
            raise RuntimeError(f"Kimi API 调用失败: {e}")

    def _parse_kimi_response(self, response: dict[str, Any]) -> ValidationResult:
        """解析 Kimi API 响应"""

        try:
            # 提取响应文本
            content = response["choices"][0]["message"]["content"]

            # 尝试提取 JSON(可能包含在代码块中)
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            # 解析 JSON
            data = json.loads(content)

            total_score = float(data["total_score"])
            passed = total_score >= self.threshold

            return ValidationResult(
                total_score=total_score,
                accuracy=float(data["accuracy"]),
                completeness=float(data["completeness"]),
                visualization=float(data["visualization"]),
                logic=float(data["logic"]),
                feedback=data["feedback"],
                passed=passed,
            )

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.error(f"解析 Kimi 响应失败: {e}, 响应内容: {response}")
            # 返回一个默认的失败结果
            return ValidationResult(
                total_score=0,
                accuracy=0,
                completeness=0,
                visualization=0,
                logic=0,
                feedback="API 响应解析失败",
                passed=False,
            )
