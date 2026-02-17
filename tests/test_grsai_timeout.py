from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import visualizer.image_generator as image_generator_module  # type: ignore[reportMissingImports]
from pipeline import VideoPipeline  # type: ignore[reportMissingImports]
from utils.counter import APICounter  # type: ignore[reportMissingImports]
from visualizer.image_generator import (  # type: ignore[reportMissingImports]
    ImageGenerator,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _make_image_generator(mock_logger, poll_timeout=2, poll_interval=0):
    config = {
        "image_generator": {
            "poll_interval": poll_interval,
            "poll_timeout": poll_timeout,
        },
        "api_keys": {},
        "grsai": {"base_url": "http://example.com"},
    }
    return ImageGenerator(config=config, logger=mock_logger)


def _make_pipeline_config(tmp_path):
    return {
        "system": {
            "output_dir": str(tmp_path),
            "temp_dir": str(tmp_path / "temp"),
            "log_dir": str(tmp_path / "logs"),
            "self_check_mode": "static",
        },
        "downloader": {
            "retry_times": 1,
            "video_format": "mp4",
            "max_resolution": 360,
        },
        "analyzer": {},
        "validator": {
            "threshold": 75,
            "max_rounds": 1,
        },
        "image_generator": {
            "poll_interval": 0,
            "poll_timeout": 1,
        },
        "proxy": {
            "base_url": "http://localhost:8000",
            "timeout": 60,
        },
        "api_keys": {},
        "grsai": {"base_url": "http://example.com"},
    }


class FakeAnalyzer:
    def __init__(self, config, api_counter, logger, api_key=None, throttle=None):
        self.config = config
        self.api_counter = api_counter
        self.logger = logger

    def analyze_video(self, video_path):
        knowledge_doc = SimpleNamespace(
            deep_dive=[],
            visual_schemas=[SimpleNamespace(schema="schema")],
        )
        knowledge_doc.to_markdown = MagicMock(return_value="markdown")
        return SimpleNamespace(knowledge_doc=knowledge_doc)

    def generate_report(self, analysis_result, image_relative_path, self_check_mode):
        return "report content"


class FakeAuditor:
    def __init__(self, config, api_counter, logger, api_key=None, throttle=None):
        self.threshold = 75.0

    def audit_image(self, image_path, knowledge_doc_content):
        return SimpleNamespace(passed=True, score=85.0, feedback="")


def test_grsai_poll_timeout(monkeypatch):
    mock_logger = MagicMock()
    generator = _make_image_generator(mock_logger, poll_timeout=2, poll_interval=0)

    response = FakeResponse({"code": 0, "data": {"status": "running", "progress": 10}})
    monkeypatch.setattr(
        image_generator_module.requests, "post", lambda *args, **kwargs: response
    )
    monkeypatch.setattr(image_generator_module.time, "sleep", lambda _: None)

    times = iter([0, 1, 3.1])
    monkeypatch.setattr(image_generator_module.time, "time", lambda: next(times))

    with pytest.raises(RuntimeError, match="grsai polling timeout"):
        generator._poll_draw_result("http://example.com", {}, "task-1")

    assert any(
        "event=grsai_timeout" in call.args[0]
        for call in mock_logger.warning.call_args_list
    )


def test_grsai_poll_logs_progress(monkeypatch):
    mock_logger = MagicMock()
    generator = _make_image_generator(mock_logger, poll_timeout=10, poll_interval=0)

    responses = iter(
        [
            FakeResponse({"code": 0, "data": {"status": "running", "progress": 10}}),
            FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "status": "succeeded",
                        "progress": 100,
                        "results": [{"url": "http://image"}],
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr(
        image_generator_module.requests, "post", lambda *args, **kwargs: next(responses)
    )
    monkeypatch.setattr(image_generator_module.time, "sleep", lambda _: None)

    times = iter([0, 0.5, 0.6, 0.7, 0.8])
    monkeypatch.setattr(image_generator_module.time, "time", lambda: next(times))

    result = generator._poll_draw_result("http://example.com", {}, "task-2")
    assert result == "http://image"

    poll_logs = [
        call.args[0]
        for call in mock_logger.info.call_args_list
        if "event=grsai_poll" in call.args[0]
    ]
    assert len(poll_logs) == 2
    assert any("status=running" in msg and "progress=10%" in msg for msg in poll_logs)


def test_pipeline_continues_on_image_timeout(monkeypatch, tmp_path):
    config = _make_pipeline_config(tmp_path)
    logger = MagicMock()
    api_counter = APICounter(max_calls=10, current_count=0)

    monkeypatch.setattr("pipeline.ContentAnalyzer", FakeAnalyzer)
    monkeypatch.setattr("pipeline.QualityAuditor", FakeAuditor)

    pipeline = VideoPipeline(config=config, logger=logger, api_counter=api_counter)
    pipeline.downloader.download_video = MagicMock(return_value="/tmp/video.mp4")
    pipeline.validator = SimpleNamespace(
        validate=MagicMock(
            return_value=SimpleNamespace(passed=True, total_score=100.0, feedback="")
        )
    )
    pipeline.generator = MagicMock()
    pipeline.generator.generate_blueprint = MagicMock(
        side_effect=RuntimeError("grsai polling timeout")
    )

    result = pipeline.process_single_video("https://example.com/video")

    assert result.success is True
    assert result.blueprint_path is None


def test_output_markdown_only_on_timeout(monkeypatch, tmp_path):
    config = _make_pipeline_config(tmp_path)
    logger = MagicMock()
    api_counter = APICounter(max_calls=10, current_count=0)

    monkeypatch.setattr("pipeline.ContentAnalyzer", FakeAnalyzer)
    monkeypatch.setattr("pipeline.QualityAuditor", FakeAuditor)

    pipeline = VideoPipeline(config=config, logger=logger, api_counter=api_counter)
    pipeline.downloader.download_video = MagicMock(return_value="/tmp/video.mp4")
    pipeline.validator = SimpleNamespace(
        validate=MagicMock(
            return_value=SimpleNamespace(passed=True, total_score=100.0, feedback="")
        )
    )
    pipeline.generator = MagicMock()
    pipeline.generator.generate_blueprint = MagicMock(
        side_effect=RuntimeError("grsai polling timeout")
    )

    result = pipeline.process_single_video("https://example.com/video")

    assert result.document_path
    doc_content = (
        tmp_path / "documents" / f"{result.video_id}_knowledge_note.md"
    ).read_text(encoding="utf-8")
    assert "⚠️ Image generation timed out, Markdown-only output" in doc_content
