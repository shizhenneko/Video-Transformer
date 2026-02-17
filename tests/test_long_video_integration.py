# pyright: reportPrivateUsage=false
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast
from unittest.mock import MagicMock, Mock, patch

from analyzer.content_analyzer import ContentAnalyzer
from utils.counter import APICounter
from utils.gemini_throttle import GeminiThrottle


def _base_config(temp_dir: Path) -> dict[str, object]:
    return {
        "system": {
            "temp_dir": str(temp_dir),
        },
        "proxy": {
            "base_url": "http://localhost:8000",
            "timeout": 60,
        },
        "analyzer": {
            "model": "gemini-2.5-flash",
            "temperature": 0.7,
            "max_output_tokens": 8192,
            "retry_times": 1,
            "timeout": 120,
            "max_continuations": 1,
            "long_video": {
                "enabled": True,
                "default_segment_seconds": 60,
                "overlap_seconds": 0,
                "min_segment_seconds": 30,
                "hard_max_api_calls": 50,
                "duration_threshold_seconds": 120,
            },
        },
    }


def _make_analyzer(config: dict[str, object], counter: APICounter) -> ContentAnalyzer:
    throttle = Mock(spec=GeminiThrottle)

    def _call_with_retry(
        func: Callable[..., object], *args: object, **kwargs: object
    ) -> object:
        return func(*args, **kwargs)

    throttle.call_with_retry = Mock(side_effect=_call_with_retry)
    throttle.wait_before_call = Mock()

    with patch("analyzer.content_analyzer.genai.Client") as mock_client:
        mock_client.return_value = MagicMock()
    analyzer = ContentAnalyzer(
        config=config,
        api_counter=counter,
        logger=MagicMock(),
        throttle=throttle,
        api_key="test-key",
    )
    analyzer._delete_remote_file = Mock()
    return analyzer


def _write_dummy_video(path: Path, size_bytes: int = 1024) -> None:
    _ = path.write_bytes(b"\x00" * size_bytes)


def _fake_video_file() -> SimpleNamespace:
    return SimpleNamespace(
        uri="gs://test/video.mp4",
        mime_type="video/mp4",
        name="files/123",
    )


def _extract_segment(
    _input_path: Path | str,
    _start: float,
    _end: float,
    output_path: Path,
    _stream_copy: bool = True,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ = output_path.write_bytes(b"segment")
    return True


def test_full_segmentation_pipeline_end_to_end(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "long_video.mp4"
    _write_dummy_video(video_path)

    responses: list[dict[str, object]] = [
        {
            "title": "Segment 1",
            "one_sentence_summary": "Summary 1",
            "key_takeaways": ["A"],
            "deep_dive": [
                {
                    "chapter_title": "00:00:00-00:01:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t1",
                            "explanation": "e1",
                            "timestamp": "00:00:10-00:00:20",
                        }
                    ],
                }
            ],
            "glossary": {"Foo": "Def1"},
        },
        {
            "title": "Segment 2",
            "one_sentence_summary": "Summary 2",
            "key_takeaways": ["B"],
            "deep_dive": [
                {
                    "chapter_title": "00:01:00-00:02:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t2",
                            "explanation": "e2",
                            "timestamp": "00:01:10-00:01:20",
                        }
                    ],
                }
            ],
            "glossary": {"Bar": "Def2"},
        },
        {
            "title": "Segment 3",
            "one_sentence_summary": "Summary 3",
            "key_takeaways": ["C"],
            "deep_dive": [
                {
                    "chapter_title": "00:02:00-00:03:10",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t3",
                            "explanation": "e3",
                            "timestamp": "00:02:10-00:02:20",
                        }
                    ],
                }
            ],
            "glossary": {},
        },
    ]

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._analyze_segment_range = Mock(
        side_effect=[
            [{"start": 0.0, "end": 60.0, "data": responses[0]}],
            [{"start": 60.0, "end": 120.0, "data": responses[1]}],
            [{"start": 120.0, "end": 180.0, "data": responses[2]}],
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=180.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ) as extract_mock,
    ):
        result = analyzer.analyze_video(video_path)

    assert analyzer._analyze_segment_range.call_count == 3
    assert extract_mock.call_count == 0
    assert result.title == "Segment 1"
    assert result.knowledge_doc.key_takeaways == ["A", "B", "C"]
    assert result.knowledge_doc.glossary == {"Foo": "Def1", "Bar": "Def2"}

    sections: list[dict[str, object]] = []
    for chapter in result.knowledge_doc.deep_dive:
        raw_sections = cast(list[object], chapter.get("sections", []))
        for section in raw_sections:
            if isinstance(section, dict):
                sections.append(cast(dict[str, object], section))
    assert [section.get("topic") for section in sections] == ["t1", "t2", "t3"]
    assert sections[0].get("timestamp") == "00:00:10-00:00:20"
    assert "关键结论" in result.to_markdown()


def test_segmentation_triggered_by_duration_threshold(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 60

    counter = APICounter(max_calls=5, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "threshold_video.mp4"
    _write_dummy_video(video_path)

    responses: list[dict[str, object]] = [
        {
            "title": "Segment 1",
            "one_sentence_summary": "Summary",
            "key_takeaways": ["A"],
            "deep_dive": [
                {
                    "chapter_title": "00:00:00-00:01:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t1",
                            "explanation": "e1",
                            "timestamp": "00:00:10-00:00:20",
                        }
                    ],
                }
            ],
            "glossary": {},
        },
        {
            "title": "Segment 2",
            "one_sentence_summary": "Summary",
            "key_takeaways": ["B"],
            "deep_dive": [
                {
                    "chapter_title": "00:01:00-00:02:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t2",
                            "explanation": "e2",
                            "timestamp": "00:01:10-00:01:20",
                        }
                    ],
                }
            ],
            "glossary": {},
        },
    ]

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._analyze_segment_range = Mock(
        side_effect=[
            [{"start": 0.0, "end": 60.0, "data": responses[0]}],
            [{"start": 60.0, "end": 61.0, "data": responses[1]}],
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=61.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert analyzer._analyze_segment_range.call_count == 2
    assert result.knowledge_doc.key_takeaways == ["A", "B"]


def test_short_video_skips_segmentation(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    counter = APICounter(max_calls=5, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "short_video.mp4"
    _write_dummy_video(video_path)

    response: dict[str, object] = {
        "title": "Short",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "topic": "t1",
                "explanation": "e1",
                "timestamp": "00:00:05-00:00:10",
            }
        ],
        "glossary": {},
    }

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._call_analysis_json = Mock(return_value=response)

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=30.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ) as extract_mock,
    ):
        result = analyzer.analyze_video(video_path)

    assert extract_mock.call_count == 0
    analyzer._upload_video.assert_called_once_with(video_path)
    assert result.title == "Short"
