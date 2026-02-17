"""
图像生成器单元测试
"""

import pytest
import requests
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from visualizer.image_generator import ImageGenerator, ImageGenerationConfig


class TestImageGenerator:
    """测试图像生成器"""

    @pytest.fixture
    def mock_config(self):
        return {
            "image_generator": {
                "style": "paper",
                "size": "1920x1080",
                "format": "png",
                "quality": 95,
            },
            "api_keys": {
                "nano_banana": "test-nb-key",
            },
        }

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def fake_image_data(self):
        # PNG 文件头
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    def test_init(self, mock_config, mock_logger):
        """测试:初始化"""
        generator = ImageGenerator(
            config=mock_config,
            logger=mock_logger,
        )

        assert generator.img_config.format == "png"
        assert generator.img_config.quality == 95
        assert generator.img_config.style == "paper"

    @patch("visualizer.image_generator.time.sleep", return_value=None)
    @patch("visualizer.image_generator.requests.get")
    @patch("visualizer.image_generator.requests.post")
    def test_generate_blueprint_success(
        self,
        mock_post,
        mock_get,
        mock_sleep,
        mock_config,
        mock_logger,
        fake_image_data,
    ):
        """测试:成功生成图片（grsai 提交+轮询模式）"""
        generator = ImageGenerator(
            config=mock_config,
            logger=mock_logger,
        )

        # 第一次 post: 提交绘画任务
        submit_resp = MagicMock()
        submit_resp.status_code = 200
        submit_resp.json.return_value = {
            "code": 0,
            "data": {"id": "task-123"},
        }

        # 第二次 post: 轮询获取结果
        poll_resp = MagicMock()
        poll_resp.status_code = 200
        poll_resp.json.return_value = {
            "code": 0,
            "data": {
                "status": "succeeded",
                "results": [{"url": "https://example.com/image.png"}],
            },
        }

        mock_post.side_effect = [submit_resp, poll_resp]

        # 模拟图片下载
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = fake_image_data

        # Mock PIL Image
        with patch("visualizer.image_generator.Image.open") as mock_img_open:
            mock_img = MagicMock()
            mock_img.size = (1920, 1080)
            mock_img.format = "PNG"
            mock_img_open.return_value = mock_img

            # 执行
            result = generator.generate_blueprint(
                mind_map_structure=["root: 测试", "  - 节点1"]
            )

            # 断言
            assert result == fake_image_data
            assert mock_post.call_count == 2
            mock_get.assert_called_once()

    @patch("visualizer.image_generator.requests.post")
    def test_generate_blueprint_api_error(
        self,
        mock_post,
        mock_config,
        mock_logger,
    ):
        """测试:API 调用失败"""
        generator = ImageGenerator(
            config=mock_config,
            logger=mock_logger,
        )

        # 模拟 API 错误（使用 requests.RequestException 以匹配源码的 except 子句）
        mock_post.side_effect = requests.RequestException("API error")

        with pytest.raises(RuntimeError):
            generator.generate_blueprint(mind_map_structure=["root: 测试"])

    def test_save_image(self, mock_config, mock_logger, fake_image_data, tmp_path):
        """测试:保存图片"""
        generator = ImageGenerator(
            config=mock_config,
            logger=mock_logger,
        )

        output_path = tmp_path / "test_image.png"

        # 执行
        saved_path = generator.save_image(fake_image_data, output_path)

        # 断言
        assert Path(saved_path).exists()
        assert Path(saved_path).read_bytes() == fake_image_data

    @patch("visualizer.image_generator.Image.open")
    def test_validate_image_wrong_size(
        self,
        mock_img_open,
        mock_config,
        mock_logger,
        fake_image_data,
    ):
        """测试:图片尺寸不符合要求"""
        generator = ImageGenerator(
            config=mock_config,
            logger=mock_logger,
        )

        # 模拟过小尺寸（低于源码 100px 阈值）
        mock_img = MagicMock()
        mock_img.size = (50, 50)
        mock_img.format = "PNG"
        mock_img_open.return_value = mock_img

        # 执行
        result = generator._validate_image(fake_image_data)

        # 断言
        assert result is False
