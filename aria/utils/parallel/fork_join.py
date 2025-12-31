"""Fork-join pattern utilities."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, TypeVar

from .executor import ParallelExecutor

R = TypeVar("R")


def fork_join(
    tasks: Sequence[Callable[[], R]],
    *,
    kind: str = "threads",
    max_workers: Optional[int] = None,
) -> List[R]:
    """Run independent callables in parallel and join their results in order."""
    with ParallelExecutor(kind=kind, max_workers=max_workers) as ex:
        return ex.run(lambda fn: fn(), tasks)


__all__ = ["fork_join"]
