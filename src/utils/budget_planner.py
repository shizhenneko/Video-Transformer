from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import math
from typing import cast


@dataclass(frozen=True)
class SegmentPlan:
    segment_duration: int
    overlap: int
    num_segments: int
    estimated_calls: int
    available_calls: int
    hard_max_calls: int
    fits_budget: bool


def _coerce_int(value: object, default: int) -> int:
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _estimate_segments(duration: float, segment_duration: int, overlap: int) -> int:
    if duration <= 0:
        return 0
    segment_duration = max(segment_duration, 1)
    overlap = max(min(overlap, segment_duration - 1), 0)
    if duration <= segment_duration:
        return 1
    stride = segment_duration - overlap
    if stride <= 0:
        stride = 1
    return int(math.ceil((duration - segment_duration) / stride)) + 1


def _estimate_calls(
    num_segments: int,
    max_continuations: int,
    retry_buffer: int,
    extra_calls: int = 0,
) -> int:
    if num_segments <= 0:
        return 0
    return (
        num_segments
        + 1
        + extra_calls
        + (num_segments * max_continuations)
        + retry_buffer
    )


def plan_segments_with_budget(
    duration: float,
    config: Mapping[str, object],
    current_api_count: int,
) -> SegmentPlan:
    analyzer_config_raw = config.get("analyzer")
    analyzer_config = (
        cast(dict[str, object], analyzer_config_raw)
        if isinstance(analyzer_config_raw, dict)
        else {}
    )
    long_video_config_raw = analyzer_config.get("long_video")
    long_video_config = (
        cast(dict[str, object], long_video_config_raw)
        if isinstance(long_video_config_raw, dict)
        else {}
    )

    default_segment = _coerce_int(long_video_config.get("default_segment_seconds"), 480)
    overlap = _coerce_int(long_video_config.get("overlap_seconds"), 20)
    min_segment = _coerce_int(long_video_config.get("min_segment_seconds"), 90)
    hard_max_calls = _coerce_int(long_video_config.get("hard_max_api_calls"), 50)
    max_continuations = _coerce_int(analyzer_config.get("max_continuations"), 3)
    retry_buffer = _coerce_int(analyzer_config.get("retry_times"), 0)
    duration_threshold = long_video_config.get("duration_threshold_seconds")
    consolidate_enabled = _coerce_bool(long_video_config.get("consolidate"), True)

    duration = max(float(duration), 0.0)
    available_calls = max(hard_max_calls - int(current_api_count), 0)

    if duration <= 0 or available_calls == 0:
        return SegmentPlan(
            segment_duration=0,
            overlap=0,
            num_segments=0,
            estimated_calls=0,
            available_calls=available_calls,
            hard_max_calls=hard_max_calls,
            fits_budget=False,
        )

    threshold_value = None
    if isinstance(duration_threshold, (int, float, str)):
        try:
            threshold_value = float(duration_threshold)
        except ValueError:
            threshold_value = None

    if threshold_value is not None and duration < threshold_value:
        segment_duration = max(int(math.ceil(duration)), 1)
        overlap = 0
    else:
        segment_duration = max(default_segment, min_segment, 1)
        overlap = max(min(overlap, segment_duration - 1), 0)

    num_segments = _estimate_segments(duration, segment_duration, overlap)
    extra_calls = 1 if consolidate_enabled else 0
    estimated_calls = _estimate_calls(
        num_segments, max_continuations, retry_buffer, extra_calls
    )

    if estimated_calls > available_calls:
        overlap = 0
        num_segments = _estimate_segments(duration, segment_duration, overlap)
        estimated_calls = _estimate_calls(
            num_segments, max_continuations, retry_buffer, extra_calls
        )

    if estimated_calls > available_calls and available_calls > 0:
        per_segment_calls = 1 + max_continuations
        overhead_calls = 1 + extra_calls + retry_buffer
        max_segments = (available_calls - overhead_calls) // per_segment_calls
        if max_segments < 1:
            return SegmentPlan(
                segment_duration=0,
                overlap=0,
                num_segments=0,
                estimated_calls=0,
                available_calls=available_calls,
                hard_max_calls=hard_max_calls,
                fits_budget=False,
            )

        max_segments = max(int(max_segments), 1)
        segment_duration = max(int(math.ceil(duration / max_segments)), min_segment, 1)
        overlap = 0
        num_segments = _estimate_segments(duration, segment_duration, overlap)
        estimated_calls = _estimate_calls(
            num_segments, max_continuations, retry_buffer, extra_calls
        )

        while estimated_calls > available_calls and max_segments > 1:
            max_segments -= 1
            segment_duration = max(
                int(math.ceil(duration / max_segments)), min_segment, 1
            )
            num_segments = _estimate_segments(duration, segment_duration, overlap)
            estimated_calls = _estimate_calls(
                num_segments, max_continuations, retry_buffer, extra_calls
            )

        if estimated_calls > available_calls:
            return SegmentPlan(
                segment_duration=0,
                overlap=0,
                num_segments=0,
                estimated_calls=0,
                available_calls=available_calls,
                hard_max_calls=hard_max_calls,
                fits_budget=False,
            )

    fits_budget = estimated_calls <= available_calls
    return SegmentPlan(
        segment_duration=segment_duration,
        overlap=overlap,
        num_segments=num_segments,
        estimated_calls=estimated_calls,
        available_calls=available_calls,
        hard_max_calls=hard_max_calls,
        fits_budget=fits_budget,
    )
