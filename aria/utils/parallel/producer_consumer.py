"""Producer-consumer pattern utilities."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
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
        with self.mutex:
            self._closed = True
            self.not_empty.notify_all()
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
    if consumer_parallelism <= 0:
        raise ValueError("consumer_parallelism must be positive")

    q: _ClosableQueue = _ClosableQueue(maxsize=queue_size)
    produce_error: Optional[BaseException] = None
    producer_failed = threading.Event()

    def producer_wrapper() -> None:
        nonlocal produce_error
        try:
            produce(q)
        except BaseException as exc:  # pragma: no cover - propagated below
            if not (q._closed and isinstance(exc, RuntimeError)):
                produce_error = exc
                producer_failed.set()
                q.close()
        finally:
            if not producer_failed.is_set():
                try:
                    q.put(END_SENTINEL)
                except RuntimeError:
                    pass

    t_prod = threading.Thread(target=producer_wrapper, daemon=True)
    t_prod.start()

    results: List[R] = []
    with ParallelExecutor(kind=kind, max_workers=consumer_parallelism) as ex:
        pending = set()
        producer_done = False

        def raise_if_producer_failed() -> None:
            if produce_error is not None:
                raise produce_error

        def drain_completed(block: bool) -> None:
            if not pending:
                return

            wait_timeout = 0.05 if block else 0
            done, _ = wait(
                    pending, timeout=wait_timeout, return_when=FIRST_COMPLETED
            )
            for fut in done:
                pending.remove(fut)
                results.append(fut.result())

        try:
            while not producer_done or pending:
                if pending and (producer_done or len(pending) >= consumer_parallelism):
                    drain_completed(block=True)
                    continue

                try:
                    item = q.get(timeout=0.05)
                except queue.Empty:
                    raise_if_producer_failed()
                    drain_completed(block=False)
                    continue

                if item is END_SENTINEL:
                    producer_done = True
                else:
                    pending.add(ex.submit(consume, item))

                drain_completed(block=False)
                if q.empty():
                    raise_if_producer_failed()
        except BaseException:
            q.close()
            for fut in pending:
                if not fut.done():
                    fut.cancel()
            ex._fast_shutdown = True
            t_prod.join(timeout=0.5)
            raise

    t_prod.join()
    raise_if_producer_failed()
    return results


__all__ = ["producer_consumer"]
