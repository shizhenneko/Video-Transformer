"""
主流程编排器

负责协调视频下载、分析、校验、图像生成和审核的完整流程
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import requests

from analyzer.content_analyzer import ContentAnalyzer
from auditor.quality_auditor import QualityAuditor
from downloader.video_downloader import VideoDownloader
from models import BatchResult, ProcessResult
from utils.counter import APICounter, APILimitExceeded
from utils.gemini_throttle import GeminiThrottle
from utils.progress_tracker import ProgressTracker
from validator.consistency_validator import ConsistencyValidator
from visualizer.image_generator import ImageGenerator


class VideoPipeline:
    """视频处理流程编排器"""

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger,
        api_counter: APICounter,
        progress_tracker: ProgressTracker | None = None,
    ):
        """
        初始化流程编排器

        Args:
            config: 系统配置
            logger: 日志记录器
            api_counter: API 调用计数器
            progress_tracker: 进度追踪器(可选)
        """
        self.config = config
        self.logger = logger
        self.api_counter = api_counter
        self.progress_tracker = progress_tracker

        self.progress_tracker = progress_tracker

        # 初始化各模块
        self.downloader = VideoDownloader(config, logger)

        self.validator = ConsistencyValidator(
            config=config,
            api_counter=api_counter,
            logger=logger,
        )

        self.generator = ImageGenerator(
            config=config,
            logger=logger,
        )

        self.validator = ConsistencyValidator(
            config=config,
            api_counter=api_counter,
            logger=logger,
        )

        self.generator = ImageGenerator(
            config=config,
            logger=logger,
        )



        # 输出目录
        self.output_dir = Path(config["system"]["output_dir"])
        self.doc_dir = self.output_dir / "documents"
        self.blueprint_dir = self.output_dir / "blueprints"

        # 创建输出目录
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.blueprint_dir.mkdir(parents=True, exist_ok=True)

        # 校验配置
        validator_config = config.get("validator", {})
        self.validation_threshold = validator_config.get("threshold", 75.0)
        self.max_validation_rounds = validator_config.get("max_rounds", 3)

        self.logger.info("VideoPipeline 初始化完成")

    def process_single_video(self, url: str) -> ProcessResult:
        """
        处理单个视频

        Args:
            url: 视频 URL

        Returns:
            ProcessResult: 处理结果
        """
        start_time = time.time()
        video_id = self._extract_video_id(url)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"开始处理视频: {video_id}")
        self.logger.info(f"URL: {url}")
        self.logger.info(f"{'='*60}")

        # 检查是否已处理
        if self.progress_tracker and self.progress_tracker.is_processed(video_id):
            self.logger.info(f"视频 {video_id} 已处理,跳过")
            return ProcessResult(
                video_id=video_id,
                url=url,
                success=True,
                error_message="已处理(跳过)",
                processing_time=0.0,
            )

        api_calls_start = self.api_counter.current_count

        # 1. 分配当前视频专用 Key
        current_api_key = self._allocate_gemini_key()
        
        # 2. 创建共享限流器 (同一个视频任务内所有 Gemini 调用共享限速)
        analyzer_config = self.config.get("analyzer", {})
        throttle = GeminiThrottle(
            min_interval=analyzer_config.get("min_call_interval", 4.0),
            max_retries=analyzer_config.get("retry_times", 10),
            max_total_wait=analyzer_config.get("max_retry_wait", 600.0),
            logger=self.logger,
        )
        
        # 3. 实例化组件 (使用当前分配的 Key + 共享限流器)
        analyzer = ContentAnalyzer(
            config=self.config,
            api_counter=self.api_counter,
            logger=self.logger,
            api_key=current_api_key,
            throttle=throttle,
        )
        auditor = QualityAuditor(
            config=self.config,
            api_counter=self.api_counter,
            logger=self.logger,
            api_key=current_api_key,
            throttle=throttle,
        )

        try:
            # 步骤 1: 下载视频
            self.logger.info("\n[1/5] 下载视频...")
            video_path = self.downloader.download_video(url)
            if not video_path:
                raise RuntimeError("视频下载失败")
            self.logger.info(f"✅ 视频已下载: {video_path}")

            # 步骤 2: 内容分析
            self.logger.info("\n[2/5] 分析视频内容...")
            analysis_result = analyzer.analyze_video(video_path)
            self.logger.info(
                f"✅ 内容分析完成 (知识点: {len(analysis_result.knowledge_doc.deep_dive)})"
            )

            # 步骤 3: 校验与改写循环
            self.logger.info("\n[3/5] 校验知识蓝图 Visual Schema...")
            final_structure = self._validation_loop(
                analysis_result.knowledge_doc.visual_schema, analysis_result.knowledge_doc.to_markdown()
            )

            # 步骤 4: 生成图片
            image_data = None
            audit_result = None
            
            if final_structure:
                self.logger.info("\n[4/5] 生成知识蓝图图片...")
                try:
                    image_data = self.generator.generate_blueprint(final_structure)
                    if image_data:
                        self.logger.info(f"✅ 图片生成完成 ({len(image_data)} bytes)")

                        # 步骤 5: 审核图片
                        self.logger.info("\n[5/5] 审核图片质量...")
                        blueprint_path_temp = self.output_dir / "temp" / f"{video_id}_temp.png"
                        blueprint_path_temp.parent.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            self.generator.save_image(image_data, blueprint_path_temp)

                            audit_result = auditor.audit_image(
                                image_path=blueprint_path_temp,
                                knowledge_doc_content=analysis_result.knowledge_doc.to_markdown(),
                            )
                            
                            if audit_result.passed:
                                self.logger.info(f"✅ 审核通过 (分数: {audit_result.score:.1f})")
                            else:
                                self.logger.warning(
                                    f"❌ 审核未通过 (分数: {audit_result.score:.1f} < {auditor.threshold})\n"
                                    f"反馈: {audit_result.feedback}"
                                )
                                self.logger.info("丢弃质量不佳的图片")
                                image_data = None
                                audit_result = None # Clear result so it doesn't show up as success in stats? or keep it?
                                # Keep explicit audit result for logging/stats if needed, but here we just need to ensure image is not saved.
                                
                        except Exception as e:
                            self.logger.warning(f"⚠️ 图片审核过程出错 (已保留原图)，跳过审核: {e}") 
                            # If audit fails due to error (not quality), we currently keep the image.
                            # Is this desired? "Kimi 看到质量不佳的图应该直接拒绝".
                            # If audit crashes, we might want to keep it or discard it. The prompt implies "quality poor".
                            # So exception means we don't know the quality. Defaulting to keep is safer for "errors", 
                            # but for "quality failure" we discard.
                        
                        # 清理临时文件
                        if blueprint_path_temp.exists():
                           blueprint_path_temp.unlink()

                    else:
                        self.logger.warning("❌ 图片生成返回空数据")
                
                except Exception as e:
                     self.logger.error(f"❌ 图片生成失败: {e}", exc_info=True)
                     image_data = None
            else:
                self.logger.warning("⚠️ Visual Schema 为空，跳过图片生成与审核")

            # 保存最终输出
            image_relative_path = f"../blueprints/{video_id}_mind_map.png" if image_data else None
            
            doc_path, blueprint_path = self._save_outputs(
                video_id=video_id,
                document_content=analyzer.generate_report(analysis_result, image_relative_path),
                image_data=image_data,
            )

            # 计算 API 调用次数
            api_calls_used = self.api_counter.current_count - api_calls_start
            processing_time = time.time() - start_time

            # 标记为已处理
            if self.progress_tracker:
                self.progress_tracker.mark_processed(video_id)

            result = ProcessResult(
                video_id=video_id,
                url=url,
                success=True,
                document_path=str(doc_path),
                blueprint_path=str(blueprint_path) if blueprint_path else None,
                api_calls_used=api_calls_used,
                processing_time=processing_time,
                audit_score=audit_result.score if audit_result else 0.0,
            )

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"✅ 处理成功: {video_id}")
            if not blueprint_path:
                self.logger.info("⚠️ 注意: 未生成知识蓝图图片")
            self.logger.info(f"API 调用: {api_calls_used}")
            self.logger.info(f"耗时: {processing_time:.1f}s")
            self.logger.info(f"{'='*60}\n")

            return result

        except APILimitExceeded as e:
            self.logger.error(f"❌ API 调用次数超限: {e}")
            processing_time = time.time() - start_time
            return ProcessResult(
                video_id=video_id,
                url=url,
                success=False,
                error_message=f"API 调用超限: {e}",
                processing_time=processing_time,
            )

        except Exception as e:
            self.logger.error(f"❌ 处理失败: {e}", exc_info=True)
            processing_time = time.time() - start_time

            # 标记为失败
            if self.progress_tracker:
                self.progress_tracker.mark_failed(video_id, str(e))

            return ProcessResult(
                video_id=video_id,
                url=url,
                success=False,
                error_message=str(e),
                processing_time=processing_time,
            )

    def process_batch(self, urls: list[str]) -> BatchResult:
        """
        批量处理视频

        Args:
            urls: 视频 URL 列表

        Returns:
            BatchResult: 批量处理结果
        """
        total = len(urls)
        self.logger.info(f"\n开始批量处理 {total} 个视频")

        result = BatchResult(total=total, successful=0, failed=0)

        for idx, url in enumerate(urls, 1):
            self.logger.info(f"\n处理进度: {idx}/{total}")

            # 检查 API 调用次数
            if not self.api_counter.can_call():
                self.logger.warning(
                    f"API 调用次数已达上限,终止批量处理 (已处理 {idx-1}/{total})"
                )
                break

            # 处理单个视频
            video_result = self.process_single_video(url)
            result.add_result(video_result)

            if video_result.success:
                result.successful += 1
            else:
                result.failed += 1

        self.logger.info(f"\n批量处理完成: {result}")
        return result

    def _validation_loop(
        self, initial_structure: str, knowledge_content: str
    ) -> str:
        """
        校验-改写循环

        Args:
            initial_structure: 初始 Visual Schema
            knowledge_content: 知识笔记内容

        Returns:
            最终的 Visual Schema
        """
        current_structure = initial_structure
        if not current_structure:
             self.logger.warning("Visual Schema 为空，跳过校验")
             return ""

        for round_num in range(1, self.max_validation_rounds + 1):
            self.logger.info(f"  第 {round_num} 轮校验...")

            try:
                validation_result = self.validator.validate(
                    mind_map_structure=current_structure,
                    knowledge_doc_content=knowledge_content,
                )

                self.logger.info(
                    f"  校验得分: {validation_result.total_score:.1f}/100"
                )

                if validation_result.passed:
                    self.logger.info(f"  ✅ 校验通过!")
                    return current_structure

                else:
                    self.logger.warning(
                        f"  ⚠️ 校验未通过 (阈值: {self.validation_threshold})"
                    )
                    self.logger.info(f"  反馈: {validation_result.feedback}")

                    if round_num < self.max_validation_rounds:
                        self.logger.info(f"  尝试改写...")
                        current_structure = analyzer.rewrite_visual_schema(
                            original_structure=current_structure,
                            feedback=validation_result.feedback,
                        )
                        self.logger.info(f"  改写完成,进入下一轮校验")
                    else:
                        self.logger.warning(
                            f"  已达最大校验轮次 ({self.max_validation_rounds}),使用当前结构"
                        )

            except Exception as e:
                self.logger.error(f"  校验失败: {e}")
                break

        return current_structure

    def _save_outputs(
        self, video_id: str, document_content: str, image_data: bytes | None
    ) -> tuple[Path, Path | None]:
        """
        保存输出文件

        Args:
            video_id: 视频 ID
            document_content: 文档内容
            image_data: 图片数据 (可选)

        Returns:
            (文档路径, 图片路径)
        """
        # 文档路径: {video_id}_knowledge_note.md
        doc_path = self.doc_dir / f"{video_id}_knowledge_note.md"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(document_content)
        self.logger.info(f"📄 文档已保存: {doc_path}")

        blueprint_path = None
        if image_data:
            # 图片路径: {video_id}_mind_map.png
            blueprint_path = self.blueprint_dir / f"{video_id}_mind_map.png"
            self.generator.save_image(image_data, blueprint_path)
            self.logger.info(f"🖼️  图片已保存: {blueprint_path}")

        return doc_path, blueprint_path

    def _extract_video_id(self, url: str) -> str:
        """
        从 URL 提取视频 ID (支持分集)

        Args:
            url: 视频 URL

        Returns:
            视频 ID (如果包含分集 p 参数,会附加 _p{N})
        """
        video_id = None
        
        # Bilibili BV 号匹配
        bv_match = re.search(r"BV[a-zA-Z0-9]+", url)
        if bv_match:
            video_id = bv_match.group(0)
            
            # 检查是否有分集参数 (p=X)
            p_match = re.search(r"[?&]p=(\d+)", url)
            if p_match:
                p_num = p_match.group(1)
                video_id = f"{video_id}_p{p_num}"

        # YouTube 视频 ID 匹配
        if not video_id:
            yt_match = re.search(r"(?:v=|/)([a-zA-Z0-9_-]{11})", url)
            if yt_match:
                video_id = yt_match.group(1)

        # 其他情况,使用 URL 的哈希值
        if not video_id:
            import hashlib
            video_id = hashlib.md5(url.encode()).hexdigest()[:12]
            
        return video_id

    def _allocate_gemini_key(self) -> str | None:
        """
        为整个 Pipeline 分配一个统一的 Gemini API Key

        Returns:
            str | None: 分配到的 API Key,如果未配置且号池不可用则返回 None
        """
        # 1. 优先从配置读取
        api_keys = self.config.get("api_keys", {})
        fixed_key = api_keys.get("gemini")
        if fixed_key:
            self.logger.info("从配置文件中使用固定 Gemini API Key")
            return fixed_key

        # 2. 从代理号池分配
        proxy_config = self.config.get("proxy", {})
        base_url = proxy_config.get("base_url", "http://localhost:8000")
        timeout = proxy_config.get("timeout", 10)

        self.logger.info(f"尝试从代理号池分配统一 Gemini API Key ({base_url})...")
        url = f"{base_url.rstrip('/')}/sdk/allocate-key"

        try:
            resp = requests.post(url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                key_id = data.get("key_id", "unknown")
                api_key = data.get("api_key")
                self.logger.info(f"✅ 成功从号池分配统一 Key: {key_id}")
                return api_key
            elif resp.status_code == 503:
                self.logger.warning("⚠️ 号池所有 Key 已耗尽")
            else:
                self.logger.warning(
                    f"⚠️ 号池分配失败 (HTTP {resp.status_code}): {resp.text}"
                )
        except Exception as e:
            self.logger.warning(f"⚠️ 无法连接号池进行统一分发: {e}")

        return None
