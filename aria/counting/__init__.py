"""Model counting interfaces and shared result types."""

from aria.counting.api import count, count_from_file, count_result, count_result_from_file
from aria.counting.core import CountResult

__all__ = [
    "CountResult",
    "count",
    "count_from_file",
    "count_result",
    "count_result_from_file",
]
