import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Union

try:
    from loguru import logger
except ImportError:  # pragma: no cover - environment fallback
    logger = None

# Replace default handler so our Logger instances don't leak to stderr.
# Other loguru users (without logger_id) still get stderr output.
_default_patched = False


def _patch_default_handler() -> None:
    global _default_patched
    if logger is None:
        return
    if not _default_patched:
        logger.remove()
        logger.add(
            sys.stderr,
            filter=lambda r: "logger_id" not in r["extra"],
        )
        _default_patched = True


def _log_level_to_str(level: Union[int, str]) -> str:
    """Convert logging level to loguru-compatible string."""
    if isinstance(level, str):
        return level
    return logging.getLevelName(level) if level else "INFO"


class Logger:
    """Logger with print_log (file only) and print_console (file + console)."""

    def __init__(
        self,
        log_file_path: str,
        log_level: Union[int, str] = logging.INFO,
    ) -> None:
        """
        Initialize the Logger class.

        Args:
            log_file_path: Path to the log file.
            log_level: Logging level, defaults to "INFO".
        """
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file_path = path
        level = _log_level_to_str(log_level)
        if logger is None:
            self._std_logger = logging.getLogger(f"aria.llmtools.{uuid.uuid4()}")
            self._std_logger.setLevel(level)
            self._std_logger.propagate = False
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
            file_handler = logging.FileHandler(path)
            file_handler.setFormatter(formatter)
            self._std_logger.addHandler(file_handler)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self._console_handler = stream_handler
            self._use_loguru = False
            return

        _patch_default_handler()
        self._id = uuid.uuid4()
        self._logger = logger.bind(logger_id=self._id)
        fmt = "{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}"
        logger.add(
            path,
            format=fmt,
            level=level,
            filter=lambda r: r["extra"].get("logger_id") == self._id,
        )
        logger.add(
            sys.stdout,
            format=fmt,
            level=level,
            filter=lambda r: r["extra"].get("logger_id") == self._id
            and r["extra"].get("console", False),
        )
        self._use_loguru = True

    def print_log(self, *args: Any) -> None:
        """Output messages to log file only."""
        message = " ".join(map(str, args))
        if self._use_loguru:
            self._logger.info(message)
        else:
            self._std_logger.info(message)

    def print_console(self, *args: Any) -> None:
        """Output messages to both console and log file."""
        message = " ".join(map(str, args))
        if self._use_loguru:
            self._logger.bind(console=True).info(message)
        else:
            if self._console_handler not in self._std_logger.handlers:
                self._std_logger.addHandler(self._console_handler)
            self._std_logger.info(message)
            self._std_logger.removeHandler(self._console_handler)
