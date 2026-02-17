# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from analyzer.content_analyzer import ContentAnalyzer
from utils.counter import APICounter, APILimitExceeded
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


def test_small_file_long_duration_triggers_segmentation(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 3600
    long_video["default_segment_seconds"] = 3600
    long_video["min_segment_seconds"] = 3600

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "tiny_long.mp4"
    _write_dummy_video(video_path, size_bytes=1024 * 1024)

    responses: list[dict[str, object]] = [
        {
            "title": "Segment 1",
            "one_sentence_summary": "Summary 1",
            "key_takeaways": ["A"],
            "deep_dive": [
                {
                    "chapter_title": "00:00:00-01:00:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t1",
                            "explanation": "e1",
                            "timestamp": "00:10:00-00:20:00",
                        }
                    ],
                }
            ],
            "glossary": {},
        },
        {
            "title": "Segment 2",
            "one_sentence_summary": "Summary 2",
            "key_takeaways": ["B"],
            "deep_dive": [
                {
                    "chapter_title": "01:00:00-02:00:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t2",
                            "explanation": "e2",
                            "timestamp": "01:10:00-01:20:00",
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
            [{"start": 0.0, "end": 3600.0, "data": responses[0]}],
            [{"start": 3600.0, "end": 7200.0, "data": responses[1]}],
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=7200.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert analyzer._analyze_segment_range.call_count == 2
    assert result.knowledge_doc.key_takeaways == ["A", "B"]


def test_vfr_timestamps_preserved(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 0

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "vfr_video.mp4"
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
                            "timestamp": {"start": 0.5, "end": 1.25},
                        }
                    ],
                }
            ],
            "glossary": {},
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
                            "timestamp": {"start": 61.75, "end": 62.4},
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
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    sections: list[dict[str, object]] = []
    for chapter in result.knowledge_doc.deep_dive:
        raw_sections = cast(list[object], chapter.get("sections", []))
        for section in raw_sections:
            if isinstance(section, dict):
                sections.append(cast(dict[str, object], section))
    assert sections[0].get("timestamp") == {"start": 0.5, "end": 1.25}


def test_non_mp4_container_handles_segmentation(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 60

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "container_video.mkv"
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
            "glossary": {},
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
            "glossary": {},
        },
    ]

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._analyze_segment_range = Mock(
        side_effect=[
            [{"start": 0.0, "end": 60.0, "data": responses[0]}],
            [{"start": 60.0, "end": 120.0, "data": responses[1]}],
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert result.title == "Segment 1"
    manifest_path = Path(tmp_path) / "segments" / video_path.stem / "manifest.json"
    manifest = cast(
        dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    segments = cast(list[dict[str, object]], manifest.get("segments", []))
    assert all(Path(str(entry["file_path"])).suffix == ".mp4" for entry in segments)


def test_no_audio_track_short_video(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    counter = APICounter(max_calls=5, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "no_audio.mp4"
    _write_dummy_video(video_path)

    response: dict[str, object] = {
        "title": "No Audio",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "topic": "t1",
                "explanation": "e1",
                "timestamp": "00:00:01-00:00:02",
            }
        ],
        "glossary": {},
    }

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._call_analysis_json = Mock(return_value=response)

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=20.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ) as extract_mock,
    ):
        result = analyzer.analyze_video(video_path)

    assert extract_mock.call_count == 0
    assert result.title == "No Audio"


def test_multi_language_video_handles_segments(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 60

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "multi_lang.mp4"
    _write_dummy_video(video_path)

    responses: list[dict[str, object]] = [
        {
            "title": "Segment 1",
            "one_sentence_summary": "Summary 1",
            "key_takeaways": ["EN: point"],
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
            "glossary": {"EN": "English"},
        },
        {
            "title": "Segment 2",
            "one_sentence_summary": "Summary 2",
            "key_takeaways": ["ES: punto"],
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
            "glossary": {"ES": "Espanol"},
        },
    ]

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._analyze_segment_range = Mock(
        side_effect=[
            [{"start": 0.0, "end": 60.0, "data": responses[0]}],
            [{"start": 60.0, "end": 120.0, "data": responses[1]}],
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert result.knowledge_doc.key_takeaways == ["EN: point", "ES: punto"]
    assert result.knowledge_doc.glossary == {"EN": "English", "ES": "Espanol"}


def test_budget_exhaustion_produces_best_effort_output(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 60

    counter = APICounter(max_calls=50, current_count=49)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "budget.mp4"
    _write_dummy_video(video_path)

    response: dict[str, object] = {
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
        "glossary": {},
    }

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._analyze_segment_range = Mock(
        side_effect=[
            [{"start": 0.0, "end": 60.0, "data": response}],
            APILimitExceeded("API 调用次数不足"),
        ]
    )

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=190.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert any("未覆盖" in item for item in result.knowledge_doc.key_takeaways)
    assert result.metadata.get("segment_gaps")


def test_segment_overflow_min_segment_reached_raises(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = cast(dict[str, object], analyzer_config["long_video"])
    long_video["duration_threshold_seconds"] = 0
    long_video["min_segment_seconds"] = 90
    long_video["default_segment_seconds"] = 60

    counter = APICounter(max_calls=5, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "overflow.mp4"
    _write_dummy_video(video_path)

    overflow = Exception(
        "400 INVALID_ARGUMENT: input token count exceeds maximum of 1048576"
    )

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._call_analysis_json = Mock(side_effect=overflow)

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=60.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
        pytest.raises(RuntimeError, match="分段分析失败"),
    ):
        _ = analyzer.analyze_video(video_path)
