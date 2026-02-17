"""
Test text-only continuation to avoid re-sending video file_data
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
import pytest
from analyzer.content_analyzer import ContentAnalyzer
from utils.counter import APICounter
from utils.gemini_throttle import GeminiThrottle


class TestTextOnlyContinuation:
    """Test that continuation rounds use text-only contents (no file_data)"""

    @pytest.fixture
    def mock_config(self):
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
                "max_continuations": 3,
            },
        }

    @pytest.fixture
    def mock_api_counter(self):
        return APICounter(max_calls=10, current_count=0)

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_throttle(self):
        throttle = Mock(spec=GeminiThrottle)
        throttle.wait_before_call = Mock()
        return throttle

    def test_continuation_removes_file_data(
        self, mock_config, mock_api_counter, mock_logger, mock_throttle
    ):
        """Test that round 2+ uses text-only contents without file_data"""

        with patch("analyzer.content_analyzer.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            captured_contents = []

            def mock_generate_stream(*args, **kwargs):
                captured_contents.append(kwargs.get("contents"))

                if len(captured_contents) == 1:
                    mock_chunk = MagicMock()
                    mock_chunk.candidates = [MagicMock()]
                    mock_chunk.candidates[0].finish_reason = SimpleNamespace(
                        name="MAX_TOKENS"
                    )
                    mock_chunk.candidates[0].content = MagicMock()
                    mock_chunk.candidates[0].content.parts = [MagicMock()]
                    mock_chunk.candidates[0].content.parts[0].thought = False
                    mock_chunk.candidates[0].content.parts[0].text = '{"partial": "json'
                    mock_chunk.usage_metadata = None
                    return [mock_chunk]
                else:
                    mock_chunk = MagicMock()
                    mock_chunk.candidates = [MagicMock()]
                    mock_chunk.candidates[0].finish_reason = SimpleNamespace(
                        name="STOP"
                    )
                    mock_chunk.candidates[0].content = MagicMock()
                    mock_chunk.candidates[0].content.parts = [MagicMock()]
                    mock_chunk.candidates[0].content.parts[0].thought = False
                    mock_chunk.candidates[0].content.parts[0].text = 'content"}'
                    mock_chunk.usage_metadata = None
                    return [mock_chunk]

            mock_client.models.generate_content_stream = mock_generate_stream

            analyzer = ContentAnalyzer(
                config=mock_config,
                api_counter=mock_api_counter,
                logger=mock_logger,
                throttle=mock_throttle,
                api_key="test_key",
            )

            initial_contents = [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "file_uri": "gs://test/video.mp4",
                                "mime_type": "video/mp4",
                            }
                        },
                        {"text": "Analyze this video"},
                    ],
                }
            ]

            from google.genai import types

            gen_config = types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=8192,
            )

            result = analyzer._stream_with_continuation(
                initial_contents, gen_config, "Continue from where you left off"
            )

            assert len(captured_contents) == 2

            round1_contents = captured_contents[0]
            assert any(
                "file_data" in part
                for msg in round1_contents
                for part in msg.get("parts", [])
            )

            round2_contents = captured_contents[1]
            assert not any(
                "file_data" in part
                for msg in round2_contents
                for part in msg.get("parts", [])
            )

            assert any(
                part.get("text") == "Analyze this video"
                for msg in round2_contents
                for part in msg.get("parts", [])
            )

            assert result == '{"partial": "jsoncontent"}'

    def test_extract_text_only_prompt(self):
        """Test _extract_text_only_prompt removes file_data"""

        contents = [
            {
                "role": "user",
                "parts": [
                    {
                        "file_data": {
                            "file_uri": "gs://test/video.mp4",
                            "mime_type": "video/mp4",
                        }
                    },
                    {"text": "Analyze this video"},
                ],
            }
        ]

        text_only = ContentAnalyzer._extract_text_only_prompt(contents)

        assert len(text_only) == 1
        assert text_only[0]["role"] == "user"
        assert len(text_only[0]["parts"]) == 1
        assert "text" in text_only[0]["parts"][0]
        assert text_only[0]["parts"][0]["text"] == "Analyze this video"
        assert not any("file_data" in part for part in text_only[0]["parts"])

    def test_is_input_token_overflow_error(self):
        """Test error detection for input token overflow"""

        overflow_error = Exception(
            "400 INVALID_ARGUMENT: input token count exceeds maximum of 1048576"
        )
        assert ContentAnalyzer._is_input_token_overflow_error(overflow_error)

        other_error = Exception("500 Internal Server Error")
        assert not ContentAnalyzer._is_input_token_overflow_error(other_error)

        output_error = Exception("400 INVALID_ARGUMENT: output token limit exceeded")
        assert not ContentAnalyzer._is_input_token_overflow_error(output_error)
