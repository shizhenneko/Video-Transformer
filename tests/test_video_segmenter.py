from __future__ import annotations

import importlib
import math
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, TypedDict, cast

import pytest


class SegmentInfo(Protocol):
    segment_id: int
    start: float
    end: float
    effective_start: float
    effective_end: float


class SegmentEntry(TypedDict):
    id: int
    start: float
    end: float
    effective_start: float
    effective_end: float
    file_path: str
    status: str
    attempts: int
    error: str | None


class SegmentManifest(TypedDict):
    segments: list[SegmentEntry]


class VideoSegmenterModule(Protocol):
    def plan_segments(
        self, duration: float, segment_seconds: float, overlap_seconds: float
    ) -> list[SegmentInfo]: ...

    def create_manifest(
        self,
        *,
        video_id: str,
        duration: float,
        segment_seconds: float,
        overlap_seconds: float,
        temp_dir: Path,
    ) -> SegmentManifest: ...

    def get_manifest_path(self, video_id: str, temp_dir: Path) -> Path: ...

    def save_manifest(self, manifest_path: Path, manifest: SegmentManifest) -> None: ...

    def load_or_create_manifest(
        self,
        *,
        video_id: str,
        duration: float,
        segment_seconds: float,
        overlap_seconds: float,
        temp_dir: Path,
    ) -> SegmentManifest: ...

    def pending_segments(self, manifest: SegmentManifest) -> list[SegmentEntry]: ...

    def extract_segment(
        self,
        *,
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
        stream_copy: bool = True,
    ) -> bool: ...


video_segmenter = cast(
    VideoSegmenterModule,
    cast(object, importlib.import_module("utils.video_segmenter")),
)


def _assert_close(value: float, expected: float) -> None:
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-6)


def test_plan_segments_with_overlap() -> None:
    segments = video_segmenter.plan_segments(
        duration=100.0, segment_seconds=30.0, overlap_seconds=5.0
    )
    assert len(segments) == 4

    _assert_close(segments[0].start, 0.0)
    _assert_close(segments[0].end, 35.0)
    _assert_close(segments[0].effective_start, 0.0)
    _assert_close(segments[0].effective_end, 30.0)

    _assert_close(segments[1].start, 25.0)
    _assert_close(segments[1].end, 65.0)
    _assert_close(segments[1].effective_start, 30.0)
    _assert_close(segments[1].effective_end, 60.0)

    _assert_close(segments[3].start, 85.0)
    _assert_close(segments[3].end, 100.0)
    _assert_close(segments[3].effective_start, 90.0)
    _assert_close(segments[3].effective_end, 100.0)


def test_plan_segments_no_overlap() -> None:
    segments = video_segmenter.plan_segments(
        duration=50.0, segment_seconds=20.0, overlap_seconds=-3.0
    )
    assert len(segments) == 3
    _assert_close(segments[1].start, 20.0)
    _assert_close(segments[1].end, 40.0)


def test_manifest_create_and_resume(tmp_path: Path) -> None:
    manifest = video_segmenter.create_manifest(
        video_id="video123",
        duration=65.0,
        segment_seconds=30.0,
        overlap_seconds=5.0,
        temp_dir=tmp_path,
    )
    manifest_path = video_segmenter.get_manifest_path("video123", tmp_path)
    assert manifest_path.exists()
    assert manifest["segments"][0]["status"] == "pending"

    manifest["segments"][0]["status"] = "completed"
    video_segmenter.save_manifest(manifest_path, manifest)

    loaded = video_segmenter.load_or_create_manifest(
        video_id="video123",
        duration=65.0,
        segment_seconds=30.0,
        overlap_seconds=5.0,
        temp_dir=tmp_path,
    )
    assert loaded["segments"][0]["status"] == "completed"
    pending = video_segmenter.pending_segments(loaded)
    assert all(segment["id"] != 0 for segment in pending)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
def test_extract_segment_integration(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "segment.mp4"

    create_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:d=1",
        "-c:v",
        "libx264",
        str(input_path),
    ]
    result = subprocess.run(create_cmd, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0

    success = video_segmenter.extract_segment(
        input_path=input_path,
        start=0.0,
        end=0.5,
        output_path=output_path,
        stream_copy=True,
    )
    assert success
    assert output_path.exists()
    assert output_path.stat().st_size > 0
