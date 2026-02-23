import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Union

from loguru import logger

# Replace default handler so our Logger instances don't leak to stderr.
# Other loguru users (without logger_id) still get stderr output.
_default_patched = False


def _patch_default_handler() -> None:
    global _default_patched
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
        _patch_default_handler()
        path = Path(log_file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._id = uuid.uuid4()
        self._logger = logger.bind(logger_id=self._id)
        self.log_file_path = path

        level = _log_level_to_str(log_level)
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

    def print_log(self, *args: Any) -> None:
        """Output messages to log file only."""
        self._logger.info(" ".join(map(str, args)))

    def print_console(self, *args: Any) -> None:
        """Output messages to both console and log file."""
        self._logger.bind(console=True).info(" ".join(map(str, args)))
