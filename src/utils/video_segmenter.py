from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast


@dataclass(frozen=True)
class SegmentInfo:
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
    version: int
    video_id: str
    created_at: str
    segment_seconds: float
    overlap_seconds: float
    segments: list[SegmentEntry]


def plan_segments(
    duration: float, segment_seconds: float, overlap_seconds: float
) -> list[SegmentInfo]:
    if duration <= 0 or segment_seconds <= 0:
        return []

    overlap = max(0.0, overlap_seconds)
    segments: list[SegmentInfo] = []
    cursor = 0.0
    segment_id = 0

    while cursor < duration:
        core_start = cursor
        core_end = min(cursor + segment_seconds, duration)

        if core_start == 0:
            extract_start = 0.0
        else:
            extract_start = max(0.0, core_start - overlap)

        if core_end >= duration:
            extract_end = duration
        else:
            extract_end = min(duration, core_end + overlap)

        if extract_end <= extract_start:
            break

        segments.append(
            SegmentInfo(
                segment_id=segment_id,
                start=extract_start,
                end=extract_end,
                effective_start=core_start,
                effective_end=core_end,
            )
        )

        segment_id += 1
        cursor = core_end

    return segments


def extract_segment(
    input_path: str | Path,
    start: float,
    end: float,
    output_path: str | Path,
    stream_copy: bool = True,
) -> bool:
    duration = end - start
    if duration <= 0:
        return False

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def run_ffmpeg(args: list[str]) -> bool:
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return False

        if result.returncode != 0:
            return False
        if not output_path.exists() or output_path.stat().st_size <= 0:
            return False
        return True

    base_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-movflags",
        "+faststart",
    ]

    if stream_copy:
        copy_cmd = base_cmd + ["-c", "copy", str(output_path)]
        if run_ffmpeg(copy_cmd):
            return True
        if output_path.exists():
            output_path.unlink()

    encode_cmd = base_cmd + [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(output_path),
    ]
    return run_ffmpeg(encode_cmd)


def snap_to_keyframe(video_path: str | Path, timestamp: float) -> float:
    _ = video_path
    return max(0.0, float(timestamp))


def get_segment_dir(video_id: str, temp_dir: str | Path) -> Path:
    return Path(temp_dir) / "segments" / video_id


def get_manifest_path(video_id: str, temp_dir: str | Path) -> Path:
    return get_segment_dir(video_id, temp_dir) / "manifest.json"


def create_manifest(
    *,
    video_id: str,
    duration: float,
    segment_seconds: float,
    overlap_seconds: float,
    temp_dir: str | Path,
) -> SegmentManifest:
    segment_dir = get_segment_dir(video_id, temp_dir)
    segment_dir.mkdir(parents=True, exist_ok=True)

    segments = plan_segments(duration, segment_seconds, overlap_seconds)
    manifest: SegmentManifest = {
        "version": 1,
        "video_id": video_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segment_seconds": segment_seconds,
        "overlap_seconds": overlap_seconds,
        "segments": [
            {
                "id": segment.segment_id,
                "start": segment.start,
                "end": segment.end,
                "effective_start": segment.effective_start,
                "effective_end": segment.effective_end,
                "file_path": str(segment_dir / f"segment_{segment.segment_id:04d}.mp4"),
                "status": "pending",
                "attempts": 0,
                "error": None,
            }
            for segment in segments
        ],
    }

    save_manifest(get_manifest_path(video_id, temp_dir), manifest)
    return manifest


def load_manifest(manifest_path: str | Path) -> SegmentManifest:
    path = Path(manifest_path)
    return cast(SegmentManifest, json.loads(path.read_text(encoding="utf-8")))


def save_manifest(manifest_path: str | Path, manifest: SegmentManifest) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def load_or_create_manifest(
    *,
    video_id: str,
    duration: float,
    segment_seconds: float,
    overlap_seconds: float,
    temp_dir: str | Path,
) -> SegmentManifest:
    manifest_path = get_manifest_path(video_id, temp_dir)
    if manifest_path.exists():
        return load_manifest(manifest_path)
    return create_manifest(
        video_id=video_id,
        duration=duration,
        segment_seconds=segment_seconds,
        overlap_seconds=overlap_seconds,
        temp_dir=temp_dir,
    )


def pending_segments(manifest: SegmentManifest) -> list[SegmentEntry]:
    return [
        segment for segment in manifest["segments"] if segment["status"] != "completed"
    ]


def update_segment_status(
    manifest: SegmentManifest,
    segment_id: int,
    status: str,
    *,
    error: str | None = None,
    increment_attempts: bool = False,
) -> None:
    for segment in manifest["segments"]:
        if segment["id"] == segment_id:
            segment["status"] = status
            if error is not None:
                segment["error"] = error
            if increment_attempts:
                segment["attempts"] = segment["attempts"] + 1
            return

    logging.getLogger(__name__).warning(
        "Segment id %s not found in manifest", segment_id
    )
