from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
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
                "consolidate": True,
                "duration_threshold_seconds": None,
            },
        },
    }


def _make_analyzer(config: dict[str, object], counter: APICounter) -> ContentAnalyzer:
    throttle = Mock(spec=GeminiThrottle)
    throttle.call_with_retry = Mock(side_effect=lambda func, *args, **_: func(*args))
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


def _write_dummy_video(path: Path) -> None:
    path.write_bytes(b"\x00\x00\x00\x00")


def _fake_video_file() -> SimpleNamespace:
    return SimpleNamespace(
        uri="gs://test/video.mp4",
        mime_type="video/mp4",
        name="files/123",
    )


def test_segment_analysis_merges_and_dedupes(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = analyzer_config["long_video"]
    assert isinstance(long_video, dict)
    long_video["consolidate"] = False
    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    responses = [
        {
            "title": "Segment 1",
            "one_sentence_summary": "Summary 1",
            "key_takeaways": ["A", "B"],
            "deep_dive": [
                {
                    "chapter_title": "00:00:00-00:01:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t1",
                            "explanation": "e1",
                            "timestamp": "00:00:10-00:00:20",
                        },
                        {
                            "topic": "t2",
                            "explanation": "e2",
                            "timestamp": "00:00:30-00:00:40",
                        },
                    ],
                }
            ],
            "glossary": {"Foo": "Def1"},
        },
        {
            "title": "Segment 2",
            "one_sentence_summary": "Summary 2",
            "key_takeaways": ["B", "C"],
            "deep_dive": [
                {
                    "chapter_title": "00:01:00-00:02:00",
                    "chapter_summary": "",
                    "sections": [
                        {
                            "topic": "t2",
                            "explanation": "e2",
                            "timestamp": "00:00:35-00:00:45",
                        },
                        {
                            "topic": "t3",
                            "explanation": "e3",
                            "timestamp": "00:01:10-00:01:20",
                        },
                    ],
                }
            ],
            "glossary": {"foo": "Def2", "Bar": "Def3"},
        },
    ]

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._generate_content = Mock(side_effect=responses)

    def _extract_segment(*_, output_path: Path, **__) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")
        return True

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert result.title == "Segment 1"
    assert result.knowledge_doc.one_sentence_summary == "Summary 1"
    assert result.knowledge_doc.key_takeaways == ["A", "B", "C"]
    assert result.knowledge_doc.glossary == {"Foo": "Def1", "Bar": "Def3"}

    chapters = result.knowledge_doc.deep_dive
    sections = [
        section
        for chapter in chapters
        for section in chapter.get("sections", [])
        if isinstance(section, dict)
    ]
    topics = [section.get("topic") for section in sections]
    assert topics == ["t1", "t2", "t3"]


def test_segment_overflow_triggers_binary_split(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = analyzer_config["long_video"]
    assert isinstance(long_video, dict)
    long_video["duration_threshold_seconds"] = 0
    long_video["default_segment_seconds"] = 200
    long_video["consolidate"] = False

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    overflow = Exception(
        "400 INVALID_ARGUMENT: input token count exceeds maximum of 1048576"
    )

    responses = [
        overflow,
        {
            "title": "Split 1",
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
            "title": "Split 2",
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
    analyzer._generate_content = Mock(side_effect=responses)

    def _extract_segment(*_, output_path: Path, **__) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")
        return True

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert analyzer._generate_content.call_count == 3
    sections = [
        section
        for chapter in result.knowledge_doc.deep_dive
        for section in chapter.get("sections", [])
        if isinstance(section, dict)
    ]
    assert [section.get("topic") for section in sections] == ["t1", "t2"]


def test_segment_budget_exhaustion_adds_gap_note(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = analyzer_config["long_video"]
    assert isinstance(long_video, dict)
    long_video["consolidate"] = False
    counter = APICounter(max_calls=1, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._generate_content = Mock(
        return_value={
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
        }
    )

    def _extract_segment(*_, output_path: Path, **__) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")
        return True

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    assert any("未覆盖" in item for item in result.knowledge_doc.key_takeaways)


def test_consolidate_segments_enforces_bounds(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    merged = {
        "title": "Merged",
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
            },
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
            },
        ],
        "glossary": {},
    }

    consolidated = {
        "title": "Merged",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "chapter_title": "Concepts 1",
                "chapter_summary": "",
                "sections": [
                    {
                        "topic": "t1",
                        "explanation": "e1",
                        "timestamp": "00:00:10-00:00:20",
                    }
                ],
            },
            {
                "chapter_title": "Concepts 2",
                "chapter_summary": "",
                "sections": [
                    {
                        "topic": "t2",
                        "explanation": "e2",
                        "timestamp": "00:01:10-00:01:20",
                    }
                ],
            },
        ],
        "glossary": {},
    }

    analyzer._call_gemini_text_api = Mock(return_value=json.dumps(consolidated))

    result = analyzer._consolidate_segments(merged)
    assert result is not None
    assert 2 <= len(result.get("deep_dive", [])) <= 6


def test_single_pass_consolidation_runs_once(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    system_config = config["system"]
    assert isinstance(system_config, dict)
    system_config["quality_gates"] = {
        "enabled": True,
        "max_extra_llm_calls": 1,
    }
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = analyzer_config["long_video"]
    assert isinstance(long_video, dict)
    long_video["enabled"] = False

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    response = {
        "title": "Single",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "chapter_title": "Ch1",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t1", "explanation": "e1", "timestamp": "00:00:10"}
                ],
            },
            {
                "chapter_title": "Ch2",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t2", "explanation": "e2", "timestamp": "00:00:20"}
                ],
            },
            {
                "chapter_title": "Ch3",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t3", "explanation": "e3", "timestamp": "00:00:30"}
                ],
            },
        ],
        "glossary": {},
        "visual_schemas": [
            {
                "type": "overview",
                "description": "",
                "schema": "---BEGIN PROMPT---\nX\n---END PROMPT---",
            }
        ],
    }
    consolidated = {
        "title": "Single",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "chapter_title": "Concepts",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t1", "explanation": "e1", "timestamp": "00:00:10"}
                ],
            },
            {
                "chapter_title": "Practice",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t2", "explanation": "e2", "timestamp": "00:00:20"},
                    {"topic": "t3", "explanation": "e3", "timestamp": "00:00:30"},
                ],
            },
        ],
        "glossary": {},
    }

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._generate_content = Mock(return_value=response)
    analyzer._call_gemini_text_api = Mock(return_value=json.dumps(consolidated))

    with patch("analyzer.content_analyzer.probe_duration", return_value=30.0):
        result = analyzer.analyze_video(video_path)

    analyzer._call_gemini_text_api.assert_called_once()
    assert len(result.knowledge_doc.deep_dive) == 2


def test_single_pass_consolidation_out_of_range_falls_back(
    tmp_path: Path,
) -> None:
    config = _base_config(tmp_path)
    system_config = config["system"]
    assert isinstance(system_config, dict)
    system_config["quality_gates"] = {
        "enabled": True,
        "max_extra_llm_calls": 1,
    }
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = analyzer_config["long_video"]
    assert isinstance(long_video, dict)
    long_video["enabled"] = False

    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    response = {
        "title": "Single",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "chapter_title": "Ch1",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t1", "explanation": "e1", "timestamp": "00:00:10"}
                ],
            },
            {
                "chapter_title": "Ch2",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t2", "explanation": "e2", "timestamp": "00:00:20"}
                ],
            },
            {
                "chapter_title": "Ch3",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t3", "explanation": "e3", "timestamp": "00:00:30"}
                ],
            },
        ],
        "glossary": {},
        "visual_schemas": [
            {
                "type": "overview",
                "description": "",
                "schema": "---BEGIN PROMPT---\nX\n---END PROMPT---",
            }
        ],
    }
    consolidated = {
        "title": "Single",
        "one_sentence_summary": "Summary",
        "key_takeaways": ["A"],
        "deep_dive": [
            {
                "chapter_title": "Only",
                "chapter_summary": "",
                "sections": [
                    {"topic": "t1", "explanation": "e1", "timestamp": "00:00:10"}
                ],
            }
        ],
        "glossary": {},
    }

    analyzer._upload_video = Mock(return_value=_fake_video_file())
    analyzer._generate_content = Mock(return_value=response)
    analyzer._call_gemini_text_api = Mock(return_value=json.dumps(consolidated))

    with patch("analyzer.content_analyzer.probe_duration", return_value=30.0):
        result = analyzer.analyze_video(video_path)

    analyzer._call_gemini_text_api.assert_called_once()
    assert len(result.knowledge_doc.deep_dive) == 3


def test_consolidation_failure_falls_back(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    responses = [
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
    analyzer._generate_content = Mock(side_effect=responses)
    analyzer._call_gemini_text_api = Mock(return_value="not-json")

    def _extract_segment(*_, output_path: Path, **__) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")
        return True

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        result = analyzer.analyze_video(video_path)

    sections = [
        section
        for chapter in result.knowledge_doc.deep_dive
        for section in chapter.get("sections", [])
        if isinstance(section, dict)
    ]
    assert [section.get("topic") for section in sections] == ["t1", "t2"]


def test_consolidation_disabled_skips_call(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    analyzer_config = config["analyzer"]
    assert isinstance(analyzer_config, dict)
    long_video = analyzer_config["long_video"]
    assert isinstance(long_video, dict)
    long_video["consolidate"] = False
    counter = APICounter(max_calls=10, current_count=0)
    analyzer = _make_analyzer(config, counter)

    video_path = tmp_path / "video.mp4"
    _write_dummy_video(video_path)

    responses = [
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
    analyzer._generate_content = Mock(side_effect=responses)
    analyzer._call_gemini_text_api = Mock()

    def _extract_segment(*_, output_path: Path, **__) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"segment")
        return True

    with (
        patch("analyzer.content_analyzer.probe_duration", return_value=120.0),
        patch(
            "analyzer.content_analyzer.extract_segment", side_effect=_extract_segment
        ),
    ):
        analyzer.analyze_video(video_path)

    analyzer._call_gemini_text_api.assert_not_called()
