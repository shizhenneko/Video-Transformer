"""
内容分析与文档生成模块

使用 Gemini 2.5 Flash 对视频进行多模态分析，生成中文精英知识笔记。
通过代理号池服务的 /sdk/allocate-key API 获取真实 API Key，
直接使用 google-genai SDK 调用 Gemini API。
"""

from __future__ import annotations

import re
import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
import requests

from utils.counter import APICounter, APILimitExceeded
from utils.gemini_throttle import GeminiThrottle
from analyzer.models import AnalysisResult
from analyzer.prompt_loader import load_prompts, render_prompt


class ContentAnalyzer:
    """
    视频内容分析器

    负责使用 Gemini 2.5 Flash 分析视频内容，生成精英知识笔记、术语表和知识蓝图结构。
    通过代理号池服务分配真实 API Key，SDK 直连 Google API。
    """

    def __init__(
        self,
        config: dict[str, Any],
        api_counter: APICounter,
        logger: logging.Logger,
        api_key: str | None = None,
        throttle: GeminiThrottle | None = None,
    ):
        """
        初始化内容分析器

        Args:
            config: 系统配置字典
            api_counter: API 调用计数器
            logger: 日志记录器
            api_key: Gemini API 密钥（可选，若提供则直接使用，不走代理号池）
        """
        self.config = config
        self.api_counter = api_counter
        self.logger = logger

        # 加载分析器配置
        self.analyzer_config = config.get("analyzer", {})
        self.model_name = self.analyzer_config.get("model", "gemini-2.5-flash")
        self.temperature = self.analyzer_config.get("temperature", 0.7)
        self.max_output_tokens = self.analyzer_config.get("max_output_tokens", 65536)
        self.retry_times = self.analyzer_config.get("retry_times", 10)
        self.timeout = self.analyzer_config.get("timeout", 120)
        self.max_continuations = self.analyzer_config.get("max_continuations", 3)

        # 限流器
        min_interval = self.analyzer_config.get("min_call_interval", 4.0)
        max_retry_wait = self.analyzer_config.get("max_retry_wait", 600.0)
        self.throttle = throttle or GeminiThrottle(
            min_interval=min_interval,
            max_retries=self.retry_times,
            max_total_wait=max_retry_wait,
            logger=logger,
        )

        proxy_config = config.get("proxy", {})
        self.proxy_base_url = proxy_config.get("base_url", "http://localhost:8000")
        self.proxy_timeout = proxy_config.get("timeout", 10)

        self._fixed_api_key = api_key
        self._allocated_key_id = None  # 如果是从外部传入且已知 ID,可以扩展接口传输 ID
        self._allocated_api_key = api_key
        self._client: genai.Client | None = None
        self._llm_repair_used = False

        http_proxy = proxy_config.get("http")

        if http_proxy:
            import os

            os.environ["HTTP_PROXY"] = http_proxy
            os.environ["HTTPS_PROXY"] = http_proxy
            # 确保本地服务不走代理
            os.environ["NO_PROXY"] = "localhost,127.0.0.1"
            self.logger.info(f"已设置代理环境变量: {http_proxy}")

        if self._allocated_api_key:
            self._client = genai.Client(
                api_key=self._allocated_api_key,
                http_options={"timeout": 600_000},
            )
            self.logger.info("Gemini SDK 配置完成(使用外部分配的 API Key)")
        else:
            self.logger.warning("未提供 Gemini API Key,ContentAnalyzer 将无法正常工作")

        # 加载 Prompt 模板
        self.prompts = load_prompts()
        self.logger.info("ContentAnalyzer 初始化完成")

    # _allocate_key_from_pool 已移除,密钥分配逻辑已移至 VideoPipeline

    def _report_usage_to_pool(self) -> None:
        """向代理号池报告一次成功的 API 调用。"""
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
        """向代理号池报告 API 调用错误。"""
        if not self._allocated_key_id:
            return
        url = f"{self.proxy_base_url.rstrip('/')}/sdk/report-error"
        try:
            requests.post(
                url,
                json={
                    "key_id": self._allocated_key_id,
                    "is_rpd_limit": is_rpd_limit,
                },
                timeout=self.proxy_timeout,
            )
        except requests.RequestException as e:
            self.logger.warning(f"向号池报告错误失败: {e}")

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

    def _compress_video_for_upload(self, video_path: Path) -> Path:
        """用 ffmpeg 压缩视频以减小上传体积，返回压缩后的临时文件路径。

        压缩策略: 360p + CRF 28，对 Gemini 内容分析足够，体积可降至原来的 1/5~1/10。
        如果 ffmpeg 不可用或压缩失败，返回原始路径。
        """
        max_size_mb = 30
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        if file_size_mb <= max_size_mb:
            self.logger.info(
                f"视频体积 {file_size_mb:.1f}MB <= {max_size_mb}MB，跳过压缩"
            )
            return video_path

        self.logger.info(
            f"视频体积 {file_size_mb:.1f}MB > {max_size_mb}MB，开始 ffmpeg 压缩..."
        )

        compressed_path = video_path.parent / f"compressed_{video_path.name}"

        # 检查是否已存在压缩文件
        if compressed_path.exists() and compressed_path.stat().st_size > 0:
            self.logger.info(f"发现已存在的压缩文件: {compressed_path.name}，跳过压缩")
            return compressed_path

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                "scale=-2:360",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-preset",
                "fast",
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(compressed_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                self.logger.warning(f"ffmpeg 压缩失败: {(result.stderr or '')[:200]}")
                return video_path

            new_size_mb = compressed_path.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"压缩完成: {file_size_mb:.1f}MB -> {new_size_mb:.1f}MB "
                f"(压缩率 {new_size_mb / file_size_mb * 100:.0f}%)"
            )
            return compressed_path

        except FileNotFoundError:
            self.logger.warning("ffmpeg 未安装，跳过压缩")
            return video_path
        except subprocess.TimeoutExpired:
            self.logger.warning("ffmpeg 压缩超时(5分钟)，跳过压缩")
            if compressed_path.exists():
                compressed_path.unlink()
            return video_path

    def _upload_video(self, video_path: Path) -> Any:
        """
        处理视频压缩和上传

        Returns:
             Gemini File 对象
        """
        upload_path = self._compress_video_for_upload(video_path)
        video_file = None

        try:
            self.logger.info(f"上传视频文件: {upload_path.name}")
            self.throttle.wait_for_files_op()
            video_file = self._client.files.upload(file=str(upload_path))

            while video_file.state.name == "PROCESSING":
                self.logger.info("等待视频处理...")
                self.throttle.wait_for_files_op()
                video_file = self._client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"视频文件处理失败: {video_file.state.name}")

            self.logger.info(f"视频文件上传成功: {video_file.name}")
            return video_file

        except Exception as e:
            self.logger.error(f"视频上传失败: {e}")
            # 如果上传失败，清理远程文件（如果已创建）
            if video_file:
                self._delete_remote_file(video_file.name)
            raise

    def analyze_video(self, video_path: str | Path) -> AnalysisResult:
        """
        分析视频内容，生成完整的分析结果
        """
        self._llm_repair_used = False
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self.logger.info(f"开始分析视频: {video_path.name}")

        # 检查 API 调用次数 (预留 2 次: 1次内容分析, 1次 Schema 生成)
        if self.api_counter.current_count + 2 > self.api_counter.max_calls:
            raise APILimitExceeded(
                f"API 调用次数不足以完成全流程: {self.api_counter.current_count}/{self.api_counter.max_calls}"
            )

        # Step 0: 准备视频文件 (压缩 + 上传)
        # 注意: 这里不进行重试，如果上传失败通常是网络或文件问题，重试意义不大或由外部控制
        if not self._client:
            raise RuntimeError("Gemini Client 未初始化 (缺少 API Key)")

        video_file = self._upload_video(video_path)

        try:
            # Step 1: 视频内容分析 (通过限流器自动重试 429)
            self.logger.info("Step 1: 执行视频内容分析...")
            system_role = self.prompts.get("gemini_analysis", {}).get("system_role", "")
            main_prompt = self.prompts.get("gemini_analysis", {}).get("main_prompt", "")

            def _on_retry(attempt: int, exc: Exception) -> None:
                """429 重试回调: 向号池报告错误"""
                self._report_error_to_pool(is_rpd_limit=True)

            # 内层重试: JSON 解析失败时重新请求 API (最多 2 次额外尝试)
            json_parse_max_retries = 2
            response_data = None
            for json_attempt in range(
                1, json_parse_max_retries + 2
            ):  # 1 次正常 + 2 次重试
                try:
                    response_data = self.throttle.call_with_retry(
                        self._generate_content,
                        video_file,
                        system_role,
                        main_prompt,
                        on_retry_callback=_on_retry,
                    )
                    break  # 成功则跳出
                except ValueError as ve:
                    if json_attempt <= json_parse_max_retries:
                        self.logger.warning(
                            f"⚠️ JSON 解析失败 (第 {json_attempt} 次)，重新请求 API: {ve}"
                        )
                    else:
                        raise  # 最后一次仍失败则抛出

            # 增加 API 计数
            self.api_counter.increment("Gemini")

            # Step 2: 生成 Visual Schema
            # 如果 Step 1 已经生成了 Visual Schema，则跳过 Step 2
            # 否则尝试单独生成 (Fallback)

            raw_schemas = response_data.get("visual_schemas", [])
            has_valid_schema = any(
                isinstance(s, dict) and "---BEGIN PROMPT---" in s.get("schema", "")
                for s in (raw_schemas if isinstance(raw_schemas, list) else [])
            )

            if has_valid_schema:
                self.logger.info("Visual Schema 已在 Step 1 中生成，跳过独立生成步骤")
            else:
                self.logger.info(
                    "Step 1 未生成有效 Visual Schema，尝试独立生成 (Step 2)..."
                )
                if self.api_counter.current_count + 1 <= self.api_counter.max_calls:
                    try:
                        self.logger.info("Step 2: 生成知识蓝图 Visual Schema...")
                        deep_dive_content = json.dumps(
                            response_data.get("deep_dive", []),
                            ensure_ascii=False,
                            indent=2,
                        )
                        fallback_schema = self._generate_visual_schema(
                            deep_dive_content
                        )
                        self.logger.info("Visual Schema 生成成功")
                        response_data["visual_schemas"] = [
                            {
                                "type": "overview",
                                "description": "总览知识导图",
                                "schema": fallback_schema,
                            }
                        ]
                    except Exception as e:
                        self.logger.error(
                            f"Visual Schema 生成失败: {e}，将生成不带图的报告"
                        )
                else:
                    self.logger.warning(
                        "API 配额不足以执行 Step 2，跳过 Visual Schema 生成"
                    )

            metadata = {
                "video_name": video_path.name,
                "video_size": video_path.stat().st_size,
            }

            try:
                result = AnalysisResult.from_api_response(
                    video_path=video_path,
                    response_data=response_data,
                    metadata=metadata,
                )
                self.logger.info(f"视频分析全流程完成: {result.title}")
                return result
            except ValueError as e:
                self.logger.error(f"API 响应格式错误: {e}")
                raise

        finally:
            # 清理资源
            if video_file:
                self._delete_remote_file(video_file.name)

            # 清理压缩文件 (可选: 如果希望下次运行复用，可以注释掉这行，或者保留以节省空间)
            # 根据用户需求 "断点处继续"，我们应该保留压缩文件，或者在成功后才清理？
            # 现在的逻辑是 `_compress_video_for_upload` 会检查文件是否存在。
            # 如果我们在这里删除了，那么下次又要重新压缩。
            # 为了支持"断点压缩"，我们应该仅在完全成功后删除，或者保留在 temp 目录由系统清理？
            # 之前的逻辑是 `finally` 中删除。
            # 用户抱怨的是 "失败后...重新下载并压缩"。
            # 如果我们删除了，下次确实要重新压缩。
            # 所以这里我们应该移除删除压缩文件的逻辑，或者只在成功时删除？
            # 实际上，`VideoDownloader` 有个 temp 目录，可能更适合放那里。
            # 这里我们还是先保留删除逻辑，但是因为 `_compress_video_for_upload` 现在检查了存在性，
            # 只要在"本次元操作"中不删除，下次重试（同一进程内）还是会通过 path 传递。
            # 但如果是进程崩溃重启，`upload_path` 是临时生成的吗？
            # `video_path.parent / f"compressed_{video_path.name}"` 是在原目录下。
            # 如果我删除了，下次进程启动还是要压缩。
            # 鉴于用户的需求，我将注释掉删除压缩文件的代码，让它保留，
            # 或者将其移动到 VideoPipeline 的清理阶段？
            # 简单起见，我先不删除压缩文件，让它变成"缓存"。
            # 为了避免垃圾堆积，可以在 AnalysisResult 成功返回前删除？
            # 还是说，用户希望的是"失败重试"时不重新压缩。
            # 当前的 `finally` 块是在 `analyze_video` 结束时执行。
            # 如果 `analyze_video` 失败抛出异常，`finally` 执行，文件被删。
            # 下次调用 `analyze_video` (比如外部重试)，文件没了，又要压缩。
            # 所以必须 **不删除** 压缩文件，或者只在 **成功** 后删除。

            compressed_file = video_path.parent / f"compressed_{video_path.name}"
            if compressed_file.exists():
                # 只有在成功生成结果后才删除？或者干脆不删，留给用户手动清理/定期清理？
                # 为了防止磁盘爆满，我们还是尝试删除，但是前提是必须区分"完全失败"和"成功"。
                # 由于 `finally` 不区分，我们很难做。
                # 最好的方式是：不在这里删除。让上层 `VideoPipeline` 或者 `cleanup` 方法来处理。
                # 或者，仅仅记录日志说"保留压缩文件以便重试"。
                self.logger.info(f"保留压缩文件以备重试: {compressed_file.name}")
                pass

    def _generate_visual_schema(self, deep_dive_content: str) -> str:
        """根据深度解析内容生成 Visual Schema"""

        schema_config = self.prompts.get("gemini_visual_schema", {})
        system_role = schema_config.get("system_role", "")
        main_prompt_template = schema_config.get("main_prompt", "")

        full_prompt = render_prompt(
            main_prompt_template, deep_dive_content=deep_dive_content
        )

        # 使用通用文本生成接口
        response_text = self._call_gemini_text_api(
            system_role=system_role, user_prompt=full_prompt, temperature=0.7
        )

        # 提取 Markdown 代码块
        if "```markdown" in response_text:
            start = response_text.find("```markdown") + 11
            end = response_text.find("```", start)
            return response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            return response_text[start:end].strip()

        return response_text.strip()

    def _stream_response(
        self,
        contents: list[dict[str, Any]],
        gen_config: types.GenerateContentConfig,
    ) -> tuple[str, str]:
        """流式接收 Gemini 响应，返回 (response_text, finish_reason_name)。

        这是 _generate_content 和 _call_gemini_text_api 共用的底层流式接收方法。

        Args:
            contents: 多轮对话内容列表
            gen_config: 生成配置

        Returns:
            (response_text, finish_reason_name) 元组。
            finish_reason_name 为 "STOP"、"MAX_TOKENS" 等字符串，
            如果未获取到 finish_reason 则返回 "UNKNOWN"。
        """
        response_text_parts: list[str] = []
        thinking_logged = False
        finish_reason_name = "UNKNOWN"

        for chunk in self._client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=gen_config,
        ):
            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]

            if candidate.finish_reason:
                finish_reason_name = candidate.finish_reason.name or "UNKNOWN"

            if not candidate.content or not candidate.content.parts:
                continue

            for part in candidate.content.parts:
                if part.thought:
                    if not thinking_logged:
                        self.logger.info("💭 Gemini 思考中...")
                        thinking_logged = True
                    snippet = (part.text or "")[:200] if part.text else ""
                    if snippet:
                        self.logger.info(f"  💭 {snippet}")
                else:
                    if part.text:
                        response_text_parts.append(part.text)
                        snippet = (part.text or "")[:100].replace("\n", " ")
                        self.logger.info(f"  📝 生成中: {snippet}...")

        response_text = "".join(response_text_parts)
        return response_text, finish_reason_name

    def _stream_with_continuation(
        self,
        contents: list[dict[str, Any]],
        gen_config: types.GenerateContentConfig,
        continuation_prompt: str,
    ) -> str:
        """流式接收 + MAX_TOKENS 自动续传。

        当 finish_reason 为 MAX_TOKENS 时，将已有内容作为 model 回复追加到对话历史，
        再发送续传指令，让 Gemini 从截断处继续输出。最多续传 max_continuations 轮。

        Args:
            contents: 初始对话内容列表（会被原地修改以追加续传轮次）
            gen_config: 生成配置
            continuation_prompt: 续传时发送给 Gemini 的用户指令

        Returns:
            拼接后的完整响应文本
        """
        all_text_parts: list[str] = []

        for round_idx in range(self.max_continuations + 1):
            round_label = f"第 {round_idx + 1} 轮" if round_idx > 0 else "首次请求"
            self.logger.info(f"开始流式接收 Gemini 响应 ({round_label})...")

            response_text, finish_reason = self._stream_response(contents, gen_config)

            self.logger.info(
                f"流式接收完成 ({round_label})，"
                f"本轮长度: {len(response_text)} 字符，"
                f"finish_reason: {finish_reason}"
            )

            all_text_parts.append(response_text)
            self._report_usage_to_pool()

            if finish_reason != "MAX_TOKENS":
                if finish_reason == "STOP":
                    self.logger.info("Gemini 生成正常结束 (STOP)")
                else:
                    self.logger.warning(
                        f"Gemini 生成结束原因非 STOP: {finish_reason} (可能发生截断)"
                    )
                break

            if round_idx >= self.max_continuations:
                self.logger.warning(
                    f"已达最大续传轮数 ({self.max_continuations})，停止续传"
                )
                break

            self.logger.info(
                f"⚠️ 检测到 MAX_TOKENS 截断，发起续传 "
                f"(第 {round_idx + 2}/{self.max_continuations + 1} 轮)..."
            )

            contents.append({"role": "model", "parts": [{"text": response_text}]})
            contents.append({"role": "user", "parts": [{"text": continuation_prompt}]})

            self.throttle.wait_before_call()

        total_text = "".join(all_text_parts)
        self.logger.info(
            f"响应总长度: {len(total_text)} 字符 (共 {len(all_text_parts)} 轮)"
        )
        return total_text

    def _generate_content(
        self,
        video_file: Any,
        system_role: str,
        main_prompt: str,
    ) -> dict[str, Any]:
        """
        使用已上传的视频文件生成内容
        """
        # Disable thinking for JSON mode to avoid empty-response edge case
        gen_config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="application/json",
            system_instruction=system_role,
            thinking_config=None,
        )

        contents: list[dict[str, Any]] = [
            {
                "role": "user",
                "parts": [
                    {
                        "file_data": {
                            "file_uri": video_file.uri,
                            "mime_type": video_file.mime_type,
                        }
                    },
                    {"text": main_prompt},
                ],
            }
        ]

        continuation_prompt = (
            "你的上一次输出因长度限制被截断了。"
            "请从上次截断处继续输出，直接续写 JSON 内容，"
            "不要重复已输出的部分，不要添加任何前缀说明。"
        )

        response_text = self._stream_with_continuation(
            contents, gen_config, continuation_prompt
        )

        response_text = response_text.strip()

        # 空响应检测：Gemini 可能只返回 thinking 内容而无实际文本
        if not response_text:
            raise ValueError("Gemini 返回了空响应（0 字符），可能仅包含 thinking 内容")

        self.logger.debug(f"API 响应: {response_text[:200]}...")

        # 解析 JSON 响应
        # 1. 优先尝试标准的 ```json ... ``` 块
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # 2. 尝试不带 json 标签的 ``` ... ``` 块，但排除非 JSON 的代码块
            code_block_match = re.search(
                r"```\s*(\{.*?\})\s*```", response_text, re.DOTALL
            )
            if code_block_match:
                response_text = code_block_match.group(1)
            else:
                # 3. 兜底：寻找最外层的 {}
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx : end_idx + 1]

        # 多轮 JSON 修复
        cleaned_text, stripped_count = self._strip_stray_token_prefixes(response_text)
        if stripped_count > 0:
            self.logger.debug(f"event=json_stray_token_strip count={stripped_count}")
        repaired = self._try_repair_json(cleaned_text)
        if repaired is not None:
            self.logger.info("API 响应解析成功")
            response_data = repaired
        else:
            if not self._llm_repair_used:
                llm_repaired = self._llm_repair_json(cleaned_text)
                self._llm_repair_used = True
                if llm_repaired is not None:
                    response_data = llm_repaired
                else:
                    self.logger.error(
                        "event=json_parse_failed reason=llm_repair_exhausted"
                    )
                    self._dump_failed_json(response_text)
                    raise ValueError("JSON parse failed after LLM repair")
            else:
                self.logger.error(
                    "event=json_parse_failed reason=llm_repair_already_used"
                )
                self._dump_failed_json(response_text)
                raise ValueError("LLM repair already used for this video")

        # 必需字段：缺失则无法构建有意义的文档
        required_fields = {
            "title",
            "one_sentence_summary",
            "key_takeaways",
            "deep_dive",
            "glossary",
        }
        # 可选字段：visual_schemas 缺失时由 Step 2 fallback 单独生成
        missing = required_fields - response_data.keys()
        if missing:
            self.logger.error(
                f"event=validation_failed reason=missing_required_fields fields={','.join(sorted(missing))}"
            )
            self.logger.warning(
                f"API 响应 JSON 不完整，缺少必需字段: {', '.join(sorted(missing))}，触发重试"
            )
            raise ValueError(
                f"API 响应 JSON 不完整，缺少必需字段: {', '.join(sorted(missing))}"
            )

        return response_data

    def _llm_repair_json(self, invalid_json: str) -> dict[str, Any] | None:
        prompt = (
            "The following is invalid JSON. Output ONLY the corrected JSON "
            "with no markdown fences, explanations, or other text. "
            "Preserve all content and meaning exactly.\n\n"
            f"{invalid_json}"
        )

        self.logger.info("event=llm_json_repair_attempt")
        try:
            response_text = self._call_gemini_text_api(
                system_role="",
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=8192,
            )
            parsed = json.loads(response_text)
        except Exception:
            self.logger.warning("event=llm_json_repair_failed")
            return None

        if isinstance(parsed, dict):
            self.logger.info("event=llm_json_repair_success")
            return parsed

        self.logger.warning("event=llm_json_repair_failed")
        return None

    @staticmethod
    def _dump_failed_json(response_text: str) -> None:
        # 修复失败：保存原始响应到磁盘以便事后调试
        try:
            dump_path = Path("data/output/logs") / f"failed_json_{int(time.time())}.txt"
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            dump_path.write_text(response_text, encoding="utf-8")
            logging.getLogger(__name__).error(
                f"JSON 修复失败，已保存原始响应到: {dump_path}"
            )
        except Exception:
            logging.getLogger(__name__).error("JSON 修复失败，且无法保存原始响应到磁盘")

    def _call_gemini_text_api(
        self,
        system_role: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 8192,
    ) -> str:
        """调用 Gemini 纯文本生成接口（通过限流器自动处理 429）"""
        if not self._client:
            raise RuntimeError("Gemini Client 未初始化 (缺少 API Key)")

        def _do_text_call() -> str:
            if not self._client:
                if self._allocated_api_key:
                    self._client = genai.Client(
                        api_key=self._allocated_api_key,
                        http_options={"timeout": 600_000},
                    )
                else:
                    raise RuntimeError("Gemini Client 未初始化且无可用 Key")

            gen_config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                system_instruction=system_role if system_role else None,
                thinking_config=types.ThinkingConfig(
                    thinking_budget=4096,
                ),
            )

            contents: list[dict[str, Any]] = [
                {"role": "user", "parts": [{"text": user_prompt}]}
            ]

            continuation_prompt = (
                "你的上一次输出因长度限制被截断了。"
                "请从上次截断处继续输出，直接续写内容，"
                "不要重复已输出的部分，不要添加任何前缀说明。"
            )

            result = self._stream_with_continuation(
                contents, gen_config, continuation_prompt
            )

            self.api_counter.increment("Gemini")
            return result.strip()

        def _on_retry(attempt: int, exc: Exception) -> None:
            self._report_error_to_pool(is_rpd_limit=True)

        return self.throttle.call_with_retry(
            _do_text_call,
            on_retry_callback=_on_retry,
        )

    @staticmethod
    def _strip_stray_token_prefixes(text: str) -> tuple[str, int]:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return text, 0

        prefix = text[: start_idx + 1]
        body = text[start_idx + 1 : end_idx]
        suffix = text[end_idx:]

        cleaned_body, count = re.subn(
            r'(?<=\n)\s*[A-Za-z0-9]\s+(?=")',
            "",
            body,
        )
        if count == 0:
            return text, 0
        return f"{prefix}{cleaned_body}{suffix}", count

    @staticmethod
    def _sanitize_json_escapes(text: str) -> str:
        """修复 JSON 字符串值中的非法转义序列。

        常见场景: Gemini 返回的 JSON 中包含 LaTeX 公式 (\frac, \sum, \ln 等),
        这些在 JSON 规范中是非法的转义序列。

        策略: 仅在字符串值内部, 将 `\X` (X 非 JSON 合法转义字符) 替换为 `\\X`。
        JSON 合法转义字符: " \\ / b f n r t u
        """
        legal_escapes = set('"\\/ b f n r t u'.split() + ['"', "\\", "/"])
        # 更精确: JSON 允许 \" \\\\ \/ \b \f \n \r \t \uXXXX
        legal_escape_chars = set('"\\/ bfnrtu')

        result: list[str] = []
        i = 0
        in_string = False
        length = len(text)

        while i < length:
            ch = text[i]

            if not in_string:
                result.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
            else:
                # 在字符串内部
                if ch == "\\":
                    # 检查下一个字符
                    if i + 1 < length:
                        next_ch = text[i + 1]
                        if next_ch in legal_escape_chars:
                            # 合法转义, 原样保留
                            result.append(ch)
                            result.append(next_ch)
                            i += 2
                        else:
                            # 非法转义 (如 \frac, \sum): 替换为 \\\\
                            result.append("\\\\")
                            # 不跳过 next_ch, 让它作为普通字符处理
                            i += 1
                    else:
                        # 反斜杠在末尾, 转义它
                        result.append("\\\\")
                        i += 1
                elif ch == '"':
                    result.append(ch)
                    in_string = False
                    i += 1
                else:
                    result.append(ch)
                    i += 1

        return "".join(result)

    @staticmethod
    def _close_truncated_json(text: str) -> str:
        """闭合被截断的 JSON: 未闭合的字符串、尾部逗号、未闭合的括号。"""
        text = text.rstrip()

        # 1. 闭合未闭合的字符串
        in_string = False
        escape_next = False
        for ch in text:
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string

        if in_string:
            text += '"'

        # 2. 去除尾部逗号
        text = text.rstrip().rstrip(",")

        # 3. 统计未闭合的括号并逆序闭合
        bracket_stack: list[str] = []
        in_str = False
        esc = False
        for ch in text:
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in ("{", "["):
                bracket_stack.append(ch)
            elif ch == "}" and bracket_stack and bracket_stack[-1] == "{":
                bracket_stack.pop()
            elif ch == "]" and bracket_stack and bracket_stack[-1] == "[":
                bracket_stack.pop()

        closing_map = {"[": "]", "{": "}"}
        suffix = "".join(closing_map[b] for b in reversed(bracket_stack))
        return text + suffix

    @staticmethod
    def _truncate_to_last_complete_item(text: str) -> str | None:
        """截断到最后一个逗号处, 然后重新闭合括号。用于丢弃被截断的最后一个元素。"""
        last_comma = text.rfind(",")
        if last_comma <= 0:
            return None

        truncated = text[:last_comma]

        # 重新扫描并闭合
        bracket_stack: list[str] = []
        in_str = False
        esc = False
        for ch in truncated:
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in ("{", "["):
                bracket_stack.append(ch)
            elif ch == "}" and bracket_stack and bracket_stack[-1] == "{":
                bracket_stack.pop()
            elif ch == "]" and bracket_stack and bracket_stack[-1] == "[":
                bracket_stack.pop()

        closing_map = {"[": "]", "{": "}"}
        suffix = "".join(closing_map[b] for b in reversed(bracket_stack))
        return truncated + suffix

    @staticmethod
    def _fix_unquoted_keys(text: str) -> str:
        """给未加引号的 JSON key 补上双引号。
        例: MINDMAP: "..." → "MINDMAP": "..."
        仅处理字符串外部、看起来像 JSON key 的裸单词。"""
        result: list[str] = []
        i = 0
        in_string = False
        esc = False
        length = len(text)

        while i < length:
            ch = text[i]
            if esc:
                result.append(ch)
                esc = False
                i += 1
                continue
            if in_string:
                result.append(ch)
                if ch == "\\":
                    esc = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue

            if ch == '"':
                result.append(ch)
                in_string = True
                i += 1
                continue

            # 字符串外部：检测裸 key（字母/下划线开头，后跟冒号）
            if ch.isalpha() or ch == "_":
                j = i + 1
                while j < length and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                # 跳过空白后检查是否紧跟冒号
                k = j
                while k < length and text[k] in (" ", "\t"):
                    k += 1
                if k < length and text[k] == ":":
                    bare_key = text[i:j]
                    result.append(f'"{bare_key}"')
                    i = j
                    continue
            result.append(ch)
            i += 1

        return "".join(result)

    @staticmethod
    def _fix_backtick_as_quote(text: str) -> str:
        """修复反引号被误用为双引号的情况。
        例: "explanation`: "..." → "explanation": "..."
        仅替换紧邻冒号的反引号（key 尾部）和紧邻冒号后的反引号（value 头部）。"""
        # key 尾部: `": → "":   (反引号替代了 key 的闭合引号)
        text = re.sub(r"`(\s*:)", r'"\1', text)
        # value 头部: : `  → : "  (反引号替代了 value 的开启引号)
        text = re.sub(r"(:\s*)`", r'\1"', text)
        return text

    @classmethod
    def _try_repair_json(cls, text: str) -> dict[str, Any] | None:
        """多轮尝试修复 JSON 响应。

        修复策略 (逐轮尝试, 成功即返回):
          第 0 轮: 直接解析 (快速路径)
          第 1 轮: 修复非法转义序列 (LaTeX 公式等)
          第 1.5 轮: 修复反引号误用 + 未加引号的 key
          第 2 轮: 修复转义 + 闭合截断
          第 3 轮: 修复转义 + 截断到最后完整项
          第 4 轮: 移除无法修复的控制字符后重试
        """

        # --- 第 0 轮: 直接解析 ---
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # --- 第 1 轮: 仅修复非法转义 ---
        sanitized = cls._sanitize_json_escapes(text)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            pass

        # --- 第 1.5 轮: 修复反引号误用 + 未加引号的 key ---
        patched = cls._fix_backtick_as_quote(sanitized)
        patched = cls._fix_unquoted_keys(patched)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

        patched_closed = cls._close_truncated_json(patched)
        try:
            return json.loads(patched_closed)
        except json.JSONDecodeError:
            pass

        # --- 第 2 轮: 修复转义 + 闭合截断 ---
        closed = cls._close_truncated_json(sanitized)
        try:
            return json.loads(closed)
        except json.JSONDecodeError:
            pass

        # --- 第 3 轮: 修复转义 + 截断到最后完整项 ---
        truncated = cls._truncate_to_last_complete_item(sanitized)
        if truncated:
            try:
                return json.loads(truncated)
            except json.JSONDecodeError:
                pass

            # 闭合截断后的结果
            closed_truncated = cls._close_truncated_json(truncated)
            try:
                return json.loads(closed_truncated)
            except json.JSONDecodeError:
                pass

        # --- 第 4 轮: 清理控制字符 ---
        # 移除 JSON 字符串值中的裸控制字符 (\x00-\x1f 除 \t \n \r)
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)
        closed_cleaned = cls._close_truncated_json(cleaned)
        try:
            return json.loads(closed_cleaned)
        except json.JSONDecodeError:
            pass

        return None

    def generate_report(
        self,
        analysis: AnalysisResult,
        image_relative_path: str | None = None,
        self_check_mode: str = "static",
    ) -> str:
        """
        生成知识笔记报告的 Markdown 格式

        Args:
            analysis: 分析结果对象
            image_relative_path: 知识蓝图图片的相对路径（用于嵌入报告）
            self_check_mode: 自测题渲染模式(static/interactive/questions_only)

        Returns:
            Markdown 格式的知识笔记报告
        """
        return analysis.to_markdown(
            image_paths=[image_relative_path] if image_relative_path else None,
            self_check_mode=self_check_mode,
        )

    def rewrite_visual_schema(
        self,
        original_structure: str,
        feedback: str,
    ) -> str:
        """
        根据反馈改写 Visual Schema
        """
        rewrite_prompt_template = self.prompts.get("gemini_rewrite", {}).get(
            "prompt", ""
        )

        full_prompt = render_prompt(
            rewrite_prompt_template,
            original_structure=original_structure,
            feedback=feedback,
        )

        return self._call_gemini_text_api(
            system_role="",
            user_prompt=full_prompt,
        )
