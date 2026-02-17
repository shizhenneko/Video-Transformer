from importlib import util
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_gemini_throttle():
    module_path = (
        Path(__file__).resolve().parents[1] / "src" / "utils" / "gemini_throttle.py"
    )
    spec = util.spec_from_file_location("gemini_throttle", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load gemini_throttle module")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gemini_throttle = _load_gemini_throttle()
GeminiThrottle = gemini_throttle.GeminiThrottle


def test_extract_retry_delay_uses_structured_retry_after():
    error = '{"error": {"details": [{"retryDelay": {"seconds": 12, "nanos": 0}}]}}'
    with patch.object(gemini_throttle.random, "random", return_value=0.0):
        delay = GeminiThrottle._extract_retry_delay(error, attempt=1, base_delay=30.0)

    assert delay == pytest.approx(10.8)


def test_call_with_retry_enforces_max_retries():
    throttle = GeminiThrottle(
        min_interval=0.0, files_interval=0.0, max_retries=2, max_total_wait=10.0
    )
    calls = {"count": 0}

    def always_fail() -> None:
        calls["count"] += 1
        raise Exception("429 Retry-After: 1")

    with patch.object(gemini_throttle.time, "sleep") as sleep_mock:
        with patch.object(gemini_throttle.random, "random", return_value=0.0):
            with pytest.raises(RuntimeError):
                throttle.call_with_retry(always_fail)

    assert calls["count"] == 2
    assert sleep_mock.called
