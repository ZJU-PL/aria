"""Pipeline pattern built on ParallelExecutor."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

from ._shared import END_SENTINEL
from .executor import ParallelExecutor


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
    logger = logging.getLogger("arlib.parallel.pipeline")

    if not stages:
        return list(inputs)

    queues = [queue.Queue(maxsize=queue_size) for _ in range(len(stages) + 1)]

    # Feed initial input
    def feeder() -> None:
        for item in inputs:
            queues[0].put(item)
        queues[0].put(END_SENTINEL)

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
            # Bind queues at definition time to avoid late-binding bugs across threads
            with ParallelExecutor(kind=kind, max_workers=n.parallelism) as ex:
                pending: set = set()
                stop_seen = False
                while not stop_seen or pending:
                    if not stop_seen:
                        try:
                            item = in_queue.get(timeout=0.05)
                        except queue.Empty:
                            item = None
                        if item is END_SENTINEL:
                            stop_seen = True
                        elif item is not None:
                            pending.add(ex.submit(n.worker, item))
                    done = {f for f in pending if f.done()}
                    for fut in done:
                        try:
                            out_queue.put(fut.result())
                        except Exception as exc:  # pragma: no cover - logged, not raised
                            logger.error("pipeline stage failure idx=%s err=%s", stage_idx, exc)
                        pending.remove(fut)
                out_queue.put(END_SENTINEL)

        t = threading.Thread(target=stage_worker, daemon=True)
        threads.append(t)
        t.start()

    results: List[Any] = []
    while True:
        item = queues[-1].get()
        if item is END_SENTINEL:
            break
        results.append(item)

    for t in threads:
        t.join()

    return results


__all__ = ["PipelineStage", "pipeline"]
