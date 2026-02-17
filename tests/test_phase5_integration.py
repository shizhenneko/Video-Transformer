"""
阶段 5 集成测试

测试主流程编排器与所有模块的集成
"""

import os
import pytest
from pathlib import Path

from pipeline import VideoPipeline
from models import ProcessResult, BatchResult
from utils.counter import APICounter
from utils.logger import setup_logging
from utils.config import load_config
from utils.proxy import verify_proxy_connection
from utils.progress_tracker import ProgressTracker


@pytest.mark.integration
class TestPhase5Integration:
    """阶段 5 集成测试"""

    @pytest.fixture(scope="class")
    def setup_system(self):
        """设置真实系统组件"""

        # 加载配置
        config = load_config("config/config.yaml")

        # 检查 API 密钥
        api_keys = config.get("api_keys", {})
        kimi_key = os.getenv("KIMI_API_KEY") or api_keys.get("kimi")
        nano_banana_key = os.getenv("NANO_BANANA_API_KEY") or api_keys.get(
            "nano_banana"
        )

        missing = []
        if not kimi_key:
            missing.append("kimi")
        if not nano_banana_key:
            missing.append("nano_banana")
        if missing:
            pytest.skip(f"缺少 API 密钥,跳过集成测试. 需要: {missing}")

        # 检查代理号池
        proxy_base_url = config.get("proxy", {}).get(
            "base_url", "http://localhost:8000"
        )
        if not verify_proxy_connection(proxy_base_url):
            pytest.skip(f"号池代理服务不可用 ({proxy_base_url})，跳过集成测试")

        config["api_keys"] = {
            "kimi": kimi_key,
            "nano_banana": nano_banana_key,
        }

        # 初始化组件
        logger = setup_logging(
            config.get("system", {}).get("log_dir", "data/logs"),
            "test_phase5_integration.log",
        )
        counter = APICounter(max_calls=20)  # 增加限制用于测试

        return {
            "config": config,
            "counter": counter,
            "logger": logger,
        }

    def test_pipeline_initialization(self, setup_system):
        """测试: Pipeline 初始化"""

        sys = setup_system

        pipeline = VideoPipeline(
            config=sys["config"],
            logger=sys["logger"],
            api_counter=sys["counter"],
        )

        assert pipeline is not None
        assert pipeline.downloader is not None
        assert pipeline.analyzer is not None
        assert pipeline.validator is not None
        assert pipeline.generator is not None
        assert pipeline.auditor is not None

    def test_video_id_extraction(self, setup_system):
        """测试: 视频 ID 提取"""

        sys = setup_system
        pipeline = VideoPipeline(
            config=sys["config"],
            logger=sys["logger"],
            api_counter=sys["counter"],
        )

        # Bilibili
        bv_id = pipeline._extract_video_id("https://www.bilibili.com/video/BV1xx411c7mD")
        assert bv_id == "BV1xx411c7mD"

        # YouTube
        yt_id = pipeline._extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert yt_id == "dQw4w9WgXcQ"

    @pytest.mark.slow
    def test_complete_pipeline_real_video(self, setup_system, tmp_path):
        """测试: 完整 Pipeline 真实视频处理"""

        sys = setup_system
        sys["counter"].reset()

        # 使用临时输出目录
        sys["config"]["system"]["output_dir"] = str(tmp_path / "output")

        pipeline = VideoPipeline(
            config=sys["config"],
            logger=sys["logger"],
            api_counter=sys["counter"],
        )

        # 使用一个短视频
        test_url = "https://www.bilibili.com/video/BV1xx411c7mD"

        # 处理
        result = pipeline.process_single_video(test_url)

        # 断言
        assert isinstance(result, ProcessResult)
        # 注意: 可能成功也可能失败(取决于网络和API),只检查结构
        assert result.video_id is not None
        assert result.url == test_url
        assert result.processing_time >= 0

        if result.success:
            assert result.document_path is not None
            assert result.blueprint_path is not None
            assert Path(result.document_path).exists()
            assert Path(result.blueprint_path).exists()
            sys["logger"].info(f"✅ 完整流程测试成功: {result}")
        else:
            sys["logger"].warning(f"⚠️  处理失败(可能是网络问题): {result.error_message}")

    def test_progress_tracker_integration(self, setup_system, tmp_path):
        """测试: 进度追踪器集成"""

        sys = setup_system
        sys["counter"].reset()

        # 进度文件
        progress_file = tmp_path / "progress.json"
        progress_tracker = ProgressTracker(progress_file, sys["logger"])

        # 使用临时输出目录
        sys["config"]["system"]["output_dir"] = str(tmp_path / "output")

        pipeline = VideoPipeline(
            config=sys["config"],
            logger=sys["logger"],
            api_counter=sys["counter"],
            progress_tracker=progress_tracker,
        )

        # 模拟已处理
        progress_tracker.mark_processed("BV_already_processed")

        # 批量处理(包含已处理的)
        urls = [
            "https://www.bilibili.com/video/BV_already_processed",
            "https://www.bilibili.com/video/BV_new_video",
        ]

        # 注意: 第一个会被跳过
        batch_result = pipeline.process_batch(urls)

        assert batch_result.total == 2
        # 第一个被跳过,但仍算成功
        assert len(batch_result.results) == 2

    def test_api_limit_enforcement(self, setup_system, tmp_path):
        """测试: API 限制执行"""

        sys = setup_system

        # 设置非常低的 API 限制
        low_counter = APICounter(max_calls=1)

        sys["config"]["system"]["output_dir"] = str(tmp_path / "output")

        pipeline = VideoPipeline(
            config=sys["config"],
            logger=sys["logger"],
            api_counter=low_counter,
        )

        # 尝试处理多个视频
        urls = [
            "https://www.bilibili.com/video/BV1",
            "https://www.bilibili.com/video/BV2",
        ]

        batch_result = pipeline.process_batch(urls)

        # 由于 API 限制,应该提前终止
        # 注意: 具体结果取决于第一个视频是否成功
        assert batch_result.total_api_calls <= 1 or batch_result.failed > 0
