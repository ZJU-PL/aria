"""Producer-consumer pattern utilities."""

from __future__ import annotations

from concurrent.futures import as_completed
import queue
import threading
from typing import Any, Callable, List, Optional, TypeVar

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
    produce_error: Optional[BaseException] = None

    def producer_wrapper() -> None:
        nonlocal produce_error
        try:
            produce(q)
        except BaseException as exc:  # pragma: no cover - propagated below
            produce_error = exc
        finally:
            q.put(END_SENTINEL)

    t_prod = threading.Thread(target=producer_wrapper, daemon=True)
    t_prod.start()

    results: List[R] = []
    with ParallelExecutor(kind=kind, max_workers=consumer_parallelism) as ex:
        futures = []
        while True:
            item = q.get()
            if item is END_SENTINEL:
                break
            futures.append(ex.submit(consume, item))

        for fut in as_completed(futures):
            results.append(fut.result())

    t_prod.join()
    if produce_error is not None:
        raise produce_error
    return results


__all__ = ["producer_consumer"]
