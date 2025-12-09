import time
from typing import Optional


# perhaps the timeit library would be more suitable

class Stopwatch:

    def __init__(self) -> None:
        self.start_time: Optional[float] = None

    def start(self) -> None:
        self.start_time = time.process_time()

    def lap(self) -> float:
        c_time = time.process_time()
        # Hopefully this simple way is equivalent to the CPU time as used in satzilla...

        return c_time - self.start_time  # type: ignore

    def reset(self) -> None:
        self.start_time = None

    def stop(self) -> None:
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        self.start_time = None

    def get_elapsed_time(self) -> float:
        if self.start_time is None:
            raise ValueError("Stopwatch not started")
        return time.process_time() - self.start_time

    def get_elapsed_time_str(self) -> str:
        return f"{self.get_elapsed_time():.2f} seconds"

    def get_elapsed_time_minutes(self) -> float:
        return self.get_elapsed_time() / 60

    def get_elapsed_time_minutes_str(self) -> str:
        return f"{self.get_elapsed_time_minutes():.2f} minutes"
