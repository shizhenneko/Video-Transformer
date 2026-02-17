from pathlib import Path
import uuid

from utils.logger import setup_logging


def _make_temp_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    tmp_root = root / "data" / "output" / "logs" / "pytest_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = tmp_root / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def test_logger_writes_file():
    temp_dir = _make_temp_dir()
    logger = setup_logging(temp_dir, log_name="test.log")
    logger.info("logger test message")

    for handler in logger.handlers:
        handler.flush()

    log_file = temp_dir / "test.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "logger test message" in content