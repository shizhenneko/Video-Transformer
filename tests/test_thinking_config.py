import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from analyzer.content_analyzer import ContentAnalyzer
from utils.counter import APICounter


@pytest.fixture
def mock_config():
    return {
        "proxy": {
            "base_url": "http://localhost:8000",
            "timeout": 60,
        },
        "analyzer": {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "retry_times": 3,
            "timeout": 120,
        },
    }


@pytest.fixture
def mock_api_counter():
    return APICounter(max_calls=10, current_count=0)


@pytest.fixture
def mock_logger():
    return MagicMock()


def _make_analyzer(mock_config, mock_api_counter, mock_logger):
    return ContentAnalyzer(
        config=mock_config,
        api_counter=mock_api_counter,
        logger=mock_logger,
        api_key="test-key",
    )


def test_generate_content_no_thinking_config(
    mock_config, mock_api_counter, mock_logger
):
    """Verify that _generate_content() disables thinking_config for JSON mode."""
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)

    captured_config = None

    def mock_stream(*args, **kwargs):
        nonlocal captured_config
        captured_config = kwargs.get("config")

        mock_chunk = MagicMock()
        mock_chunk.candidates = [
            MagicMock(
                finish_reason=MagicMock(name="STOP"),
                content=MagicMock(
                    parts=[MagicMock(thought=False, text='{"title": "Test"}')]
                ),
            )
        ]
        return [mock_chunk]

    analyzer._client.models.generate_content_stream = mock_stream

    video_file = SimpleNamespace(uri="file://video", mime_type="video/mp4")

    try:
        analyzer._generate_content(video_file, system_role="", main_prompt="prompt")
    except Exception:
        pass

    assert captured_config is not None, "Config was not captured"
    assert captured_config.thinking_config is None, (
        f"Expected thinking_config=None, got {captured_config.thinking_config}"
    )


def test_analyze_video_no_empty_response(mock_config, mock_api_counter, mock_logger):
    """Verify that empty responses (only thinking, no text) trigger retry."""
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)

    def mock_stream_thinking_only(*args, **kwargs):
        mock_chunk = MagicMock()
        mock_chunk.candidates = [
            MagicMock(
                finish_reason=MagicMock(name="STOP"),
                content=MagicMock(
                    parts=[MagicMock(thought=True, text="thinking content")]
                ),
            )
        ]
        return [mock_chunk]

    analyzer._client.models.generate_content_stream = mock_stream_thinking_only

    video_file = SimpleNamespace(uri="file://video", mime_type="video/mp4")

    with pytest.raises(
        ValueError, match="返回了空响应.*0 字符.*可能仅包含 thinking 内容"
    ):
        analyzer._generate_content(video_file, system_role="", main_prompt="prompt")
