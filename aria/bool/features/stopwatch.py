"""
Stopwatch class for timing operations.
"""
import time
from typing import Optional


# perhaps the timeit library would be more suitable

class Stopwatch:
    """Stopwatch for measuring elapsed time."""

    def __init__(self) -> None:
        """Initialize the stopwatch."""
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start the stopwatch."""
        self.start_time = time.process_time()

    def lap(self) -> float:
        """
        Get the elapsed time since start without stopping.
        :return: Elapsed time in seconds
        """
        c_time = time.process_time()
        # Hopefully this simple way is equivalent to the CPU time
        # as used in satzilla...
        return c_time - self.start_time  # type: ignore

    def reset(self) -> None:
        """Reset the stopwatch."""
        self.start_time = None

    def stop(self) -> None:
        """Stop the stopwatch."""
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        self.start_time = None

    def get_elapsed_time(self) -> float:
        """
        Get the elapsed time since start.
        :return: Elapsed time in seconds
        """
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        return time.process_time() - self.start_time

    def get_elapsed_time_str(self) -> str:
        """
        Get the elapsed time as a formatted string.
        :return: Formatted elapsed time string
        """
        return f"{self.get_elapsed_time():.2f} seconds"

    def get_elapsed_time_minutes(self) -> float:
        """
        Get the elapsed time in minutes.
        :return: Elapsed time in minutes
        """
        return self.get_elapsed_time() / 60

    def get_elapsed_time_minutes_str(self) -> str:
        """
        Get the elapsed time in minutes as a formatted string.
        :return: Formatted elapsed time string in minutes
        """
        return f"{self.get_elapsed_time_minutes():.2f} minutes"
