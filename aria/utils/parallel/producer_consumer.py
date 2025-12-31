"""Producer-consumer pattern utilities."""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable, List, TypeVar

from ._shared import END_SENTINEL
from .executor import ParallelExecutor

R = TypeVar("R")


def producer_consumer(
    produce: Callable[[queue.Queue], None],
    consume: Callable[[Any], R],
    *,
    consumer_parallelism: int = 1,
    kind: str = "threads",
    queue_size: int = 256,
) -> List[R]:
    """Run a producer feeding a pool of consumers.

    Returns results in completion order.
    """
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    results: List[R] = []
    results_lock = threading.Lock()

    def consumer_loop() -> None:
        with ParallelExecutor(kind=kind, max_workers=consumer_parallelism) as ex:
            while True:
                item = q.get()
                if item is END_SENTINEL:
                    break
                out = ex.run(consume, [item])[0]
                with results_lock:
                    results.append(out)

    t_prod = threading.Thread(target=produce, args=(q,), daemon=True)
    t_cons = threading.Thread(target=consumer_loop, daemon=True)
    t_prod.start()
    t_cons.start()

    t_prod.join()
    q.put(END_SENTINEL)
    t_cons.join()
    return results


__all__ = ["producer_consumer"]
