"""Master-slave (task fanout) pattern utilities."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, TypeVar

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
        return ex.run(worker, list(tasks))


__all__ = ["master_slave"]
