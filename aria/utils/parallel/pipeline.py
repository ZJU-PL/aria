"""Pipeline pattern built on ParallelExecutor."""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ._shared import END_SENTINEL
from .executor import ParallelExecutor


class _ClosableQueue(queue.Queue):
    """Queue that can interrupt blocked producers during shutdown."""

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self._closed = False

    def close(self) -> None:
        with self.mutex:
            self._closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def put(
        self, item: Any, block: bool = True, timeout: Optional[float] = None
    ) -> None:
        with self.not_full:
            if self._closed:
                raise RuntimeError("pipeline queue closed")

            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise queue.Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise RuntimeError("pipeline queue closed")
                        self.not_full.wait(timeout=0.05)
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time.monotonic() + timeout
                    while self._qsize() >= self.maxsize:
                        if self._closed:
                            raise RuntimeError("pipeline queue closed")
                        remaining = endtime - time.monotonic()
                        if remaining <= 0.0:
                            raise queue.Full
                        self.not_full.wait(timeout=min(remaining, 0.05))

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()


@dataclass
class PipelineStage:
    worker: Callable[[Any], Any]
    parallelism: int = 1


def pipeline(
    stages: Sequence[PipelineStage],
    inputs: Iterable[Any],
    *,
    queue_size: int = 64,
    kind: str = "threads",
) -> List[Any]:
    """Simple M-stage pipeline with bounded queues between stages."""
    logger = logging.getLogger("aria.parallel.pipeline")
    empty_marker = object()
    stage_error: List[BaseException] = []
    error_lock = threading.Lock()
    error_event = threading.Event()

    if not stages:
        return list(inputs)
    for stage in stages:
        if stage.parallelism <= 0:
            raise ValueError("pipeline stage parallelism must be positive")

    queues = [_ClosableQueue(maxsize=queue_size) for _ in range(len(stages) + 1)]

    def stop_pipeline(exc: BaseException) -> None:
        with error_lock:
            if stage_error:
                return
            stage_error.append(exc)
            error_event.set()
        for pipeline_queue in queues:
            pipeline_queue.close()

    # Feed initial input
    def feeder() -> None:
        try:
            for item in inputs:
                if error_event.is_set():
                    return
                queues[0].put(item)
            queues[0].put(END_SENTINEL)
        except BaseException as exc:
            if not (error_event.is_set() and isinstance(exc, RuntimeError)):
                stop_pipeline(exc)

    threads: List[threading.Thread] = []
    t_feed = threading.Thread(target=feeder, daemon=True)
    threads.append(t_feed)
    t_feed.start()

    # Stage workers
    for idx, stage in enumerate(stages):
        out_q = queues[idx + 1]
        in_q = queues[idx]

        def stage_worker(
            n: PipelineStage = stage,
            stage_idx: int = idx,
            in_queue=in_q,
            out_queue=out_q,
        ) -> None:
            try:
                # Bind queues at definition time to avoid late-binding bugs across threads
                with ParallelExecutor(kind=kind, max_workers=n.parallelism) as ex:
                    pending: list = []  # ordered: index 0 is oldest submitted future
                    stop_seen = False

                    def drain_completed(block: bool) -> None:
                        # Drain from the front to preserve input order.
                        while pending:
                            fut = pending[0]
                            if not fut.done():
                                if not block:
                                    break
                                done, _ = wait(
                                    [fut],
                                    timeout=0.05,
                                    return_when=FIRST_COMPLETED,
                                )
                                if not done:
                                    break
                            pending.pop(0)
                            try:
                                out_queue.put(fut.result())
                            except Exception as exc:
                                logger.error(
                                    "pipeline stage failure idx=%s err=%s",
                                    stage_idx,
                                    exc,
                                )
                                ex._fast_shutdown = True
                                stop_pipeline(exc)

                    while (not stop_seen or pending) and not error_event.is_set():
                        if pending and (stop_seen or len(pending) >= n.parallelism):
                            drain_completed(block=True)
                            continue

                        if not stop_seen and not error_event.is_set():
                            try:
                                item = in_queue.get(timeout=0.05)
                            except queue.Empty:
                                item = empty_marker
                            if item is END_SENTINEL:
                                stop_seen = True
                            elif item is not empty_marker:
                                pending.append(ex.submit(n.worker, item))

                        drain_completed(block=False)
                    ex._fast_shutdown = ex._fast_shutdown or error_event.is_set()
            except BaseException as exc:
                stop_pipeline(exc)
            else:
                if not error_event.is_set():
                    try:
                        out_queue.put(END_SENTINEL)
                    except RuntimeError:
                        pass

        t = threading.Thread(target=stage_worker, daemon=True)
        threads.append(t)
        t.start()

    results: List[Any] = []
    while True:
        if error_event.is_set() and queues[-1].empty():
            break
        try:
            item = queues[-1].get(timeout=0.05)
        except queue.Empty:
            if error_event.is_set():
                break
            continue
        if item is END_SENTINEL:
            break
        results.append(item)

    for t in threads:
        t.join()

    if stage_error:
        raise stage_error[0]

    return results


__all__ = ["PipelineStage", "pipeline"]
