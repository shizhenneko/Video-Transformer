from __future__ import annotations

import subprocess
from pathlib import Path


def probe_duration(video_path: str | Path) -> float:
    path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0.0

    if result.returncode != 0:
        return 0.0

    raw = (result.stdout or "").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.0
