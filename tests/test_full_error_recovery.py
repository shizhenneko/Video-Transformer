# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import json
import logging
import sys
import types as py_types
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from utils.counter import APICounter
from utils.logger import setup_logging


class FakeThrottle:
    def __init__(self, *args, **kwargs):
        pass

    def call_with_retry(self, func, *args, **kwargs):
        kwargs.pop("on_retry_callback", None)
        return func(*args, **kwargs)

    def wait_before_call(self):
        return None

    def wait_for_files_op(self):
        return None


class FakeDownloader:
    video_path: Path | None = None

    def __init__(self, *args, **kwargs):
        pass

    def download_video(self, _url: str) -> str:
        if not FakeDownloader.video_path:
            raise RuntimeError("FakeDownloader missing video_path")
        return str(FakeDownloader.video_path)


class FakeGenerator:
    last_structure: str | None = None

    def __init__(self, *args, **kwargs):
        pass

    def generate_blueprint(self, structure: str) -> bytes:
        FakeGenerator.last_structure = structure
        raise RuntimeError("grsai polling timeout")

    def save_image(self, _image_data: bytes, _output_path: Path) -> None:
        return None


class FakeValidator:
    call_count: int = 0

    def __init__(self, *args, **kwargs):
        pass

    def validate(self, mind_map_structure: str, knowledge_doc_content: str):
        _ = mind_map_structure
        _ = knowledge_doc_content
        FakeValidator.call_count += 1
        if FakeValidator.call_count == 1:
            return SimpleNamespace(
                total_score=35.0,
                accuracy=14.0,
                completeness=10.5,
                visualization=7.0,
                logic=3.5,
                feedback="Needs improvement",
                passed=False,
            )
        return SimpleNamespace(
            total_score=80.0,
            accuracy=32.0,
            completeness=24.0,
            visualization=16.0,
            logic=8.0,
            feedback="Looks good",
            passed=True,
        )


class FakeAuditor:
    def __init__(self, *args, **kwargs):
        pass

    def audit_image(self, image_path: Path, knowledge_doc_content: str):
        _ = image_path
        _ = knowledge_doc_content
        return SimpleNamespace(passed=True, score=90.0, feedback="Good")


@pytest.fixture
def mock_config(tmp_path):
    return {
        "system": {
            "max_api_calls": 10,
            "temp_dir": str(tmp_path / "temp"),
            "output_dir": str(tmp_path / "output"),
        },
        "validator": {"threshold": 75.0, "max_rounds": 3},
        "api_keys": {"gemini": "test-key"},
        "proxy": {"base_url": "http://localhost:8000"},
        "analyzer": {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_output_tokens": 2048,
            "retry_times": 3,
            "timeout": 120,
            "min_call_interval": 0.0,
            "max_retry_wait": 1.0,
        },
    }


@pytest.fixture
def logger(tmp_path):
    return setup_logging(str(tmp_path / "logs"), "test.log")


@pytest.fixture
def api_counter():
    return APICounter(max_calls=10)


def _valid_response_payload():
    return {
        "title": "Recovered Title",
        "one_sentence_summary": "Recovered summary",
        "key_takeaways": ["Point A"],
        "deep_dive": [],
        "glossary": {"Term": "Definition"},
        "visual_schemas": [
            {
                "type": "overview",
                "description": "Recovered schema",
                "schema": "---BEGIN PROMPT---\nA-->B",
            }
        ],
    }


def _fake_google_genai_modules():
    fake_google = py_types.ModuleType("google")
    fake_genai = py_types.ModuleType("google.genai")
    fake_types = py_types.ModuleType("google.genai.types")

    class DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    setattr(fake_types, "GenerateContentConfig", DummyConfig)
    setattr(fake_types, "ThinkingConfig", DummyConfig)
    setattr(fake_genai, "types", fake_types)
    setattr(fake_genai, "Client", object)
    setattr(fake_google, "genai", fake_genai)

    return {
        "google": fake_google,
        "google.genai": fake_genai,
        "google.genai.types": fake_types,
    }


def test_full_error_recovery_flow(tmp_path, mock_config, logger, api_counter):
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"data")
    FakeDownloader.video_path = video_path
    FakeValidator.call_count = 0
    FakeGenerator.last_structure = None

    stray_response = (
        "{\n"
        'G          "title": "Broken"\n'
        '1          "one_sentence_summary": "Broken",\n'
        '2          "key_takeaways": ["A"],\n'
        '3          "deep_dive": [],\n'
        '4          "glossary": {},\n'
        '5          "visual_schemas": [{"schema": "---BEGIN PROMPT---"}],\n'
        '6          "invalid": ...\n'
        "}"
    )

    log_stream = StringIO()
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setLevel(logging.INFO)
    logger.addHandler(log_handler)

    with patch.dict(sys.modules, _fake_google_genai_modules(), clear=False):
        from analyzer.content_analyzer import ContentAnalyzer
        from pipeline import VideoPipeline

        with (
            patch("pipeline.VideoDownloader", FakeDownloader),
            patch("pipeline.ImageGenerator", FakeGenerator),
            patch("pipeline.ConsistencyValidator", FakeValidator),
            patch("pipeline.QualityAuditor", FakeAuditor),
            patch("pipeline.GeminiThrottle", FakeThrottle),
            patch(
                "pipeline.VideoPipeline._allocate_gemini_key",
                return_value="test-key",
            ),
            patch(
                "analyzer.content_analyzer.load_prompts",
                return_value={
                    "gemini_analysis": {"system_role": "", "main_prompt": ""}
                },
            ),
            patch("analyzer.content_analyzer.genai.Client") as MockClient,
            patch.object(ContentAnalyzer, "_stream_response") as mock_stream,
            patch.object(ContentAnalyzer, "_call_gemini_text_api") as mock_repair,
            patch.object(ContentAnalyzer, "_upload_video") as mock_upload,
            patch.object(ContentAnalyzer, "_delete_remote_file") as mock_delete,
            patch.object(
                ContentAnalyzer,
                "rewrite_visual_schema",
                return_value="updated schema string",
            ) as mock_rewrite,
        ):
            MockClient.return_value = SimpleNamespace()

            mock_stream.side_effect = [
                ("", "UNKNOWN"),
                (stray_response, "STOP"),
            ]
            mock_repair.return_value = json.dumps(_valid_response_payload())

            mock_upload.return_value = SimpleNamespace(
                uri="file://video", mime_type="video/mp4", name="video"
            )
            mock_delete.return_value = None

            pipeline = VideoPipeline(
                config=mock_config, logger=logger, api_counter=api_counter
            )

            result = pipeline.process_single_video("https://example.com/video")

    logger.removeHandler(log_handler)

    assert result.success is True
    assert result.blueprint_path is None
    assert result.document_path is not None
    assert Path(result.document_path).exists()

    assert mock_stream.call_count == 2
    mock_repair.assert_called_once()
    mock_rewrite.assert_called_once_with(
        original_structure="---BEGIN PROMPT---\nA-->B",
        feedback="Needs improvement",
    )
    assert FakeGenerator.last_structure == "updated schema string"
    assert FakeValidator.call_count == 2

    combined = log_stream.getvalue()
    assert "event=video_start" in combined
    assert "event=video_complete" in combined
