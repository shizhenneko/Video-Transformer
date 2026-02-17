"""
阶段四集成测试

测试校验、图像生成和审核模块的集成工作流
"""

import os
import pytest
from pathlib import Path

from analyzer.content_analyzer import ContentAnalyzer
from validator.consistency_validator import ConsistencyValidator
from visualizer.image_generator import ImageGenerator
from auditor.quality_auditor import QualityAuditor
from utils.counter import APICounter
from utils.logger import setup_logging
from utils.config import load_config
from utils.proxy import verify_proxy_connection


@pytest.mark.integration
class TestPhase4Integration:
    """阶段四集成测试"""

    @pytest.fixture(scope="class")
    def setup_system(self):
        """设置真实系统组件"""

        # 加载配置
        config = load_config("config/config.yaml")

        # 从配置文件读取 API 密钥（环境变量可覆盖）
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

        # Gemini key 通过号池代理动态分配，检查号池连通性
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
            config.get("system", {}).get("log_dir", "data/logs"), "test_integration.log"
        )
        counter = APICounter(max_calls=10)

        # Gemini key 不传固定值，由号池动态分配
        analyzer = ContentAnalyzer(
            config=config,
            api_counter=counter,
            logger=logger,
        )

        validator = ConsistencyValidator(
            config=config,
            api_counter=counter,
            logger=logger,
        )

        generator = ImageGenerator(
            config=config,
            logger=logger,
        )

        auditor = QualityAuditor(
            config=config,
            api_counter=counter,
            logger=logger,
        )

        return {
            "config": config,
            "counter": counter,
            "analyzer": analyzer,
            "validator": validator,
            "generator": generator,
            "auditor": auditor,
            "logger": logger,
        }

    def test_validation_workflow(self, setup_system):
        """测试:校验工作流"""

        sys = setup_system

        # 模拟蓝图结构
        mind_map_structure = [
            "root: 深度学习基础",
            "  - 核心组件",
            "    - 神经元",
            "    - 权重与偏置",
            "    - 激活函数",
            "  - 训练机制",
            "    - 前向传播",
            "    - 损失函数",
            "    - 反向传播",
        ]

        knowledge_content = """# 深度学习基础

深度学习通过模拟人脑神经元...

## 核心知识点

### 1. 神经元
神经网络的基本单元...

### 2. 激活函数
引入非线性因素...
"""

        # 执行校验
        result = sys["validator"].validate(
            mind_map_structure=mind_map_structure,
            knowledge_doc_content=knowledge_content,
        )

        # 断言
        assert result is not None
        assert result.total_score >= 0
        assert sys["counter"].current_count == 1  # Kimi 调用1次

        sys["logger"].info(f"校验结果: {result}")

    def test_rewrite_workflow(self, setup_system):
        """测试:改写工作流"""

        sys = setup_system
        sys["counter"].reset()  # 重置计数器

        original_structure = [
            "root: 测试主题",
            "  - 节点1",
        ]

        feedback = "需要扩展节点1的下级内容"

        # 执行改写
        rewritten = sys["analyzer"].rewrite_mind_map(
            original_structure=original_structure,
            feedback=feedback,
        )

        # 断言
        assert isinstance(rewritten, list)
        assert len(rewritten) > 0
        assert sys["counter"].current_count == 1  # Gemini 调用1次

        sys["logger"].info(f"改写后结构: {rewritten}")

    @pytest.mark.slow
    def test_image_generation_workflow(self, setup_system, tmp_path):
        """测试:图像生成工作流"""

        sys = setup_system

        mind_map_structure = [
            "root: 机器学习",
            "  - 监督学习",
            "  - 无监督学习",
            "  - 强化学习",
        ]

        # 执行生成
        image_data = sys["generator"].generate_blueprint(
            mind_map_structure=mind_map_structure
        )

        # 断言
        assert image_data is not None
        assert len(image_data) > 0

        # 保存并验证
        output_path = tmp_path / "test_blueprint.png"
        saved_path = sys["generator"].save_image(image_data, output_path)

        assert Path(saved_path).exists()
        sys["logger"].info(f"图片已保存: {saved_path}")

    @pytest.mark.slow
    def test_audit_workflow(self, setup_system, tmp_path):
        """测试:审核工作流"""

        sys = setup_system
        sys["counter"].reset()

        # 先生成一张图片
        mind_map_structure = [
            "root: Python基础",
            "  - 变量类型",
            "  - 控制流",
        ]

        image_data = sys["generator"].generate_blueprint(mind_map_structure)
        image_path = tmp_path / "test_audit.png"
        sys["generator"].save_image(image_data, image_path)

        knowledge_content = """# Python基础

Python是一门流行的编程语言...

## 核心概念

### 变量类型
Python支持多种变量类型...

### 控制流
if/else、for、while等...
"""

        # 执行审核
        result = sys["auditor"].audit_image(
            image_path=image_path,
            knowledge_doc_content=knowledge_content,
        )

        # 断言
        assert isinstance(
            result,
            __import__("auditor.quality_auditor", fromlist=["AuditResult"]).AuditResult,
        )
        assert result.score >= 0
        assert sys["counter"].current_count == 1  # Gemini 调用1次

        sys["logger"].info(f"审核结果: {result}")

    @pytest.mark.slow
    def test_complete_workflow(self, setup_system, tmp_path):
        """测试:完整工作流(校验 -> 改写 -> 图像生成 -> 审核)"""

        sys = setup_system
        sys["counter"].reset()

        # 步骤1: 初始蓝图
        initial_structure = [
            "root: AI概念",
            "  - 机器学习",
            "  - 深度学习",
        ]

        knowledge_content = """# AI 概念

人工智能包含多个领域...
"""

        # 步骤2: 校验
        validation_result = sys["validator"].validate(
            mind_map_structure=initial_structure,
            knowledge_doc_content=knowledge_content,
        )

        sys["logger"].info(f"初始校验: {validation_result.total_score}")

        # 步骤3: 如果未通过,改写
        final_structure = initial_structure
        if not validation_result.passed:
            final_structure = sys["analyzer"].rewrite_mind_map(
                original_structure=initial_structure,
                feedback=validation_result.feedback,
            )
            sys["logger"].info(f"改写后结构: {final_structure}")

        # 步骤4: 生成图片
        image_data = sys["generator"].generate_blueprint(final_structure)
        image_path = tmp_path / "complete_workflow.png"
        sys["generator"].save_image(image_data, image_path)

        # 步骤5: 审核图片
        audit_result = sys["auditor"].audit_image(
            image_path=image_path,
            knowledge_doc_content=knowledge_content,
        )

        # 断言
        assert audit_result.score >= 0
        assert sys["counter"].current_count <= 10  # 确保未超过限制

        sys["logger"].info(
            f"完整流程完成. API调用: {sys['counter'].current_count}/10, "
            f"审核分数: {audit_result.score}"
        )
