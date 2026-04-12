"""Producer-consumer pattern utilities."""

from __future__ import annotations

from concurrent.futures import as_completed
import queue
import threading
import time
from typing import Any, Callable, List, Optional, TypeVar

from ._shared import END_SENTINEL
from .executor import ParallelExecutor

R = TypeVar("R")


class _ClosableQueue(queue.Queue):
    """Queue that can interrupt blocked producers when closed."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._closed = False

    def close(self) -> None:
        with self.not_full:
            self._closed = True
            self.not_full.notify_all()

    def put(
        self,
        item: Any,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        with self.not_full:
            if self._closed:
                raise RuntimeError("producer_consumer queue closed")

            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise queue.Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise RuntimeError("producer_consumer queue closed")
                        self.not_full.wait(timeout=0.05)
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time.monotonic() + timeout
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise RuntimeError("producer_consumer queue closed")
                        remaining = endtime - time.monotonic()
                        if remaining <= 0.0:
                            raise queue.Full
                        self.not_full.wait(timeout=min(remaining, 0.05))

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


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
    q: _ClosableQueue = _ClosableQueue(maxsize=queue_size)
    produce_error: Optional[BaseException] = None

    def producer_wrapper() -> None:
        nonlocal produce_error
        try:
            produce(q)
        except BaseException as exc:  # pragma: no cover - propagated below
            if not (q._closed and isinstance(exc, RuntimeError)):
                produce_error = exc
        finally:
            try:
                q.put(END_SENTINEL)
            except RuntimeError:
                pass

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

        try:
            for fut in as_completed(futures):
                results.append(fut.result())
        except Exception:
            q.close()
            for fut in futures:
                if not fut.done():
                    fut.cancel()
            ex._fast_shutdown = True
            t_prod.join(timeout=0.5)
            raise

    t_prod.join()
    if produce_error is not None:
        raise produce_error
    return results


__all__ = ["producer_consumer"]
