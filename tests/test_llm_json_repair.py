import json
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _valid_response_payload():
    return {
        "title": "Test Title",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["Key"],
        "deep_dive": [],
        "glossary": {},
        "visual_schemas": [{"schema": "---BEGIN PROMPT---"}],
    }


def _make_analyzer(mock_config, mock_api_counter, mock_logger):
    return ContentAnalyzer(
        config=mock_config,
        api_counter=mock_api_counter,
        logger=mock_logger,
    )


def test_llm_repair_json_success(mock_config, mock_api_counter, mock_logger):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._call_gemini_text_api = MagicMock(return_value='{"title": "ok"}')

    result = analyzer._llm_repair_json("{bad")

    assert result == {"title": "ok"}
    analyzer._call_gemini_text_api.assert_called_once()


def test_llm_repair_json_failure(mock_config, mock_api_counter, mock_logger):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._call_gemini_text_api = MagicMock(return_value="{bad")

    result = analyzer._llm_repair_json("{bad")

    assert result is None


def test_llm_repair_json_called_once(mock_config, mock_api_counter, mock_logger):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._stream_with_continuation = MagicMock(return_value="{bad")
    analyzer._try_repair_json = MagicMock(return_value=None)
    analyzer._call_gemini_text_api = MagicMock(
        return_value=json.dumps(_valid_response_payload())
    )

    video_file = SimpleNamespace(uri="file://video", mime_type="video/mp4")
    result = analyzer._generate_content(
        video_file, system_role="", main_prompt="prompt"
    )

    assert result["title"] == "Test Title"
    analyzer._call_gemini_text_api.assert_called_once()


def test_llm_repair_increments_counter(mock_config, mock_api_counter, mock_logger):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._client = MagicMock()
    analyzer._stream_with_continuation = MagicMock(
        return_value=json.dumps(_valid_response_payload())
    )
    analyzer.throttle = MagicMock()
    analyzer.throttle.call_with_retry.side_effect = lambda func, *args, **kwargs: func()

    assert analyzer.api_counter.current_count == 0
    result = analyzer._llm_repair_json("{bad")

    assert result == _valid_response_payload()
    assert analyzer.api_counter.current_count == 1


def test_llm_repair_flag_prevents_second_attempt(
    mock_config, mock_api_counter, mock_logger
):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._llm_repair_used = True
    analyzer._stream_with_continuation = MagicMock(return_value="{bad")
    analyzer._try_repair_json = MagicMock(return_value=None)
    analyzer._call_gemini_text_api = MagicMock()

    video_file = SimpleNamespace(uri="file://video", mime_type="video/mp4")

    with pytest.raises(ValueError, match="LLM repair already used for this video"):
        analyzer._generate_content(video_file, system_role="", main_prompt="prompt")

    analyzer._call_gemini_text_api.assert_not_called()


def test_llm_repair_flag_reset_on_analyze_video(
    mock_config, mock_api_counter, mock_logger, tmp_path
):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._client = MagicMock()
    analyzer._llm_repair_used = True
    analyzer._upload_video = MagicMock(
        return_value=SimpleNamespace(
            uri="file://video", mime_type="video/mp4", name="v"
        )
    )
    analyzer._delete_remote_file = MagicMock()
    analyzer.throttle = MagicMock()
    analyzer.throttle.call_with_retry.side_effect = lambda func, *args, **kwargs: func()

    response_data = _valid_response_payload()

    def _fake_generate_content(*args, **kwargs):
        assert analyzer._llm_repair_used is False
        return response_data

    analyzer._generate_content = MagicMock(side_effect=_fake_generate_content)

    video_path = tmp_path / "video.mp4"
    video_path.write_text("data", encoding="utf-8")

    analyzer.analyze_video(video_path)

    assert analyzer._llm_repair_used is False


def test_generate_content_with_llm_repair_fallback(
    mock_config, mock_api_counter, mock_logger
):
    analyzer = _make_analyzer(mock_config, mock_api_counter, mock_logger)
    analyzer._stream_with_continuation = MagicMock(return_value="{bad")
    analyzer._try_repair_json = MagicMock(return_value=None)
    analyzer._call_gemini_text_api = MagicMock(
        return_value=json.dumps(_valid_response_payload())
    )

    video_file = SimpleNamespace(uri="file://video", mime_type="video/mp4")
    result = analyzer._generate_content(
        video_file, system_role="", main_prompt="prompt"
    )

    assert result["title"] == "Test Title"
    assert analyzer._llm_repair_used is True
    analyzer._call_gemini_text_api.assert_called_once()
