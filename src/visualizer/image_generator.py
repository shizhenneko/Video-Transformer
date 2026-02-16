"""
图像生成模块

使用 grsai 中转站调用 Nano Banana Pro API 生成知识蓝图可视化图片。
采用异步提交 + 轮询获取结果的模式。
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image


from analyzer.prompt_loader import (  # type: ignore[reportImplicitRelativeImport]
    load_prompts,
    render_prompt,
)


@dataclass
class ImageGenerationConfig:
    """图像生成配置"""

    style: str = "paper"
    model: str = "nano-banana-pro"
    aspect_ratio: str = "16:9"
    image_size: str = "1K"
    format: str = "png"
    quality: int = 95
    poll_interval: int = 3
    poll_timeout: int = 180


class ImageGenerator:
    """知识蓝图图像生成器（通过 grsai 中转站调用 Nano Banana）"""

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger,
        api_key: str | None = None,
    ):
        self.config = config
        self.logger = logger

        img_config = config.get("image_generator", {})
        self.img_config = ImageGenerationConfig(
            style=img_config.get("style", "paper"),
            model=img_config.get("model", "nano-banana-pro"),
            aspect_ratio=img_config.get("aspect_ratio", "16:9"),
            image_size=img_config.get("image_size", "1K"),
            format=img_config.get("format", "png"),
            quality=img_config.get("quality", 95),
            poll_interval=img_config.get("poll_interval", 3),
            poll_timeout=img_config.get("poll_timeout", 180),
        )

        self.api_key = api_key or config.get("api_keys", {}).get("nano_banana", "")
        if not self.api_key:
            self.logger.warning("Nano Banana API Key 未配置,图像生成功能将不可用")

        grsai_config = config.get("grsai", {})
        self.base_url = grsai_config.get("base_url", "https://grsai.dakka.com.cn")
        self.timeout = 120

        # 加载提示词模板
        self.prompts = load_prompts()

        self.logger.info("ImageGenerator 初始化完成 (grsai 中转站)")

    def generate_blueprint(
        self,
        visual_schema: str,
    ) -> bytes:
        """
        生成知识蓝图图片

        Args:
            visual_schema: 知识蓝图 Visual Schema 描述字符串

        Returns:
            图片二进制数据

        Raises:
            RuntimeError: 如果生成失败
        """
        if not visual_schema:
            raise ValueError("Visual Schema 为空，无法生成图片")

        prompt = self._build_generation_prompt(visual_schema)

        try:
            image_data = self._call_grsai_draw_api(prompt)

            if self._validate_image(image_data):
                self.logger.info("知识蓝图图片生成成功")
                return image_data
            else:
                raise RuntimeError("生成的图片验证失败")

        except Exception as e:
            self.logger.error(f"图像生成失败: {e}")
            raise

    def save_image(self, image_data: bytes, output_path: str | Path) -> str:
        """
        保存图片到本地

        Args:
            image_data: 图片二进制数据
            output_path: 输出路径

        Returns:
            保存后的文件完整路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:
            f.write(image_data)

        self.logger.info(f"图片已保存: {output_path}")
        return str(output_path.absolute())

    def _build_generation_prompt(self, schema: str) -> str:
        """构建图像生成 Prompt"""

        renderer_config = self.prompts.get("nano_banana_renderer", {})
        prompt_template = renderer_config.get("prompt", "")

        if not prompt_template:
            self.logger.warning("未找到 nano_banana_renderer 提示词模板，使用默认模板")
            return f"Draw a diagram based on:\n{schema}"

        return render_prompt(prompt_template, visual_schema=schema)

    def _call_grsai_draw_api(self, prompt: str) -> bytes:
        """通过 grsai 中转站调用 Nano Banana 绘画接口（webHook="-1" 轮询模式）"""

        draw_url = f"{self.base_url}/v1/draw/nano-banana"
        result_url = f"{self.base_url}/v1/draw/result"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.img_config.model,
            "prompt": prompt,
            "aspectRatio": self.img_config.aspect_ratio,
            "imageSize": self.img_config.image_size,
            "webHook": "-1",
            "shutProgress": True,
        }

        try:
            self.logger.info(
                f"提交绘画任务到 grsai 中转站 (model={self.img_config.model})..."
            )
            response = requests.post(
                draw_url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()

            submit_result = response.json()
            if submit_result.get("code") != 0:
                raise RuntimeError(
                    f"提交绘画任务失败: {submit_result.get('msg', '未知错误')}"
                )

            task_id = submit_result["data"]["id"]
            self.logger.info(f"绘画任务已提交, task_id={task_id}")

            image_url = self._poll_draw_result(result_url, headers, task_id)

            self.logger.info(f"下载图片: {image_url}")

            # 使用 Session 和 Retry 机制下载图片，增强稳定性
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            try:
                img_response = session.get(image_url, timeout=60)
                img_response.raise_for_status()
                return img_response.content
            except requests.exceptions.SSLError as e:
                self.logger.warning(
                    f"下载图片 SSL 验证失败: {e}，尝试禁用 SSL 验证重试..."
                )
                # 最后的兜底：如果 SSL 失败，尝试禁用验证 (虽然不安全，但在这种场景下通常可以接受)
                img_response = session.get(image_url, timeout=60, verify=False)
                img_response.raise_for_status()
                return img_response.content
            finally:
                session.close()

        except requests.RequestException as e:
            raise RuntimeError(f"grsai Nano Banana API 调用或者图片下载失败: {e}")

    def _poll_draw_result(
        self, result_url: str, headers: dict[str, str], task_id: str
    ) -> str:
        """轮询获取绘画结果，返回图片 URL"""
        start_time = time.time()

        while True:
            time.sleep(self.img_config.poll_interval)

            try:
                resp = requests.post(
                    result_url,
                    json={"id": task_id},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") == -22:
                    raise RuntimeError(f"绘画任务不存在: {task_id}")

                task_data = data.get("data", {})
                status = task_data.get("status", "")
                progress = task_data.get("progress", 0)

                elapsed = time.time() - start_time
                self.logger.info(
                    "event=grsai_poll "
                    f"elapsed={elapsed:.1f}s status={status} "
                    f"progress={progress}% task_id={task_id}"
                )

                if elapsed > self.img_config.poll_timeout:
                    self.logger.warning(f"event=grsai_timeout task_id={task_id}")
                    raise RuntimeError("grsai polling timeout")

                if status == "succeeded":
                    results = task_data.get("results", [])
                    if not results or not results[0].get("url"):
                        raise RuntimeError("绘画成功但未返回图片 URL")
                    return results[0]["url"]

                if status == "failed":
                    reason = task_data.get("failure_reason", "")
                    error = task_data.get("error", "")
                    raise RuntimeError(f"绘画任务失败: reason={reason}, error={error}")

            except requests.RequestException as e:
                self.logger.warning(f"轮询请求失败: {e}")

    def _validate_image(self, image_data: bytes) -> bool:
        """验证图片有效性"""

        try:
            img = Image.open(io.BytesIO(image_data))
            width, height = img.size

            if width < 100 or height < 100:
                self.logger.warning(f"图片分辨率过低: {width}x{height}")
                return False

            self.logger.info(f"图片验证通过: {width}x{height}, {img.format}")
            return True

        except Exception as e:
            self.logger.error(f"图片验证失败: {e}")
            return False
