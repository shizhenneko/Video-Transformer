from __future__ import annotations

from typing import cast

from utils.budget_planner import plan_segments_with_budget


def _base_config() -> dict[str, object]:
    return {
        "analyzer": {
            "max_continuations": 3,
            "retry_times": 5,
            "long_video": {
                "enabled": True,
                "default_segment_seconds": 480,
                "overlap_seconds": 20,
                "min_segment_seconds": 90,
                "hard_max_api_calls": 50,
                "consolidate": True,
            },
        }
    }


def test_long_video_caps_calls() -> None:
    config = _base_config()
    duration = 3 * 60 * 60
    plan = plan_segments_with_budget(duration, config, current_api_count=0)
    assert plan.num_segments >= 1
    assert plan.estimated_calls <= plan.hard_max_calls


def test_short_video_under_threshold_single_segment() -> None:
    config = _base_config()
    analyzer_config = cast(dict[str, object], config["analyzer"])
    long_video_config = cast(dict[str, object], analyzer_config["long_video"])
    long_video_config["duration_threshold_seconds"] = 600
    duration = 9 * 60
    plan = plan_segments_with_budget(duration, config, current_api_count=0)
    assert plan.num_segments == 1
    assert plan.overlap == 0


def test_budget_exact_limit() -> None:
    config: dict[str, object] = {
        "analyzer": {
            "max_continuations": 2,
            "retry_times": 0,
            "long_video": {
                "enabled": True,
                "default_segment_seconds": 400,
                "overlap_seconds": 0,
                "min_segment_seconds": 90,
                "hard_max_api_calls": 8,
                "consolidate": True,
            },
        }
    }
    duration = 1200
    plan = plan_segments_with_budget(duration, config, current_api_count=0)
    assert plan.estimated_calls == plan.hard_max_calls
