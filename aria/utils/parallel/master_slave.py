"""Master-slave (task fanout) pattern utilities."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar

from .executor import ParallelExecutor

R = TypeVar("R")


def master_slave(
    tasks: Iterable[Any],
    worker: Callable[[Any], R],
    *,
    max_workers: Optional[int] = None,
    kind: str = "threads",
) -> List[R]:
    """Master generates tasks; slaves execute and return results in input order."""
    with ParallelExecutor(kind=kind, max_workers=max_workers) as ex:
        max_in_flight = max_workers or getattr(ex._pool, "_max_workers", 1)
        pending: Dict[Any, int] = {}
        results: Dict[int, R] = {}

        def drain_one_completion() -> None:
            done, _ = wait(pending.keys(), timeout=None, return_when=FIRST_COMPLETED)
            for future in done:
                results[pending.pop(future)] = future.result()

        try:
            for idx, task in enumerate(tasks):
                while len(pending) >= max_in_flight:
                    drain_one_completion()
                pending[ex.submit(worker, task)] = idx

            while pending:
                drain_one_completion()
        except BaseException:
            for future in pending:
                if not future.done():
                    future.cancel()
            ex._fast_shutdown = True
            raise

        return [results[i] for i in range(len(results))]


__all__ = ["master_slave"]
