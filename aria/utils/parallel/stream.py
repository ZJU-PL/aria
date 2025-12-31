"""Simple streaming primitives for building streaming pipelines.

Provides a minimal Stream abstraction that can map/filter/batch and consume
using thread or process executors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, TypeVar

from .executor import ParallelExecutor

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Stream(Iterable[T]):
    source: Iterable[T]

    def __iter__(self) -> Iterator[T]:
        yield from self.source

    def map(
        self,
        fn: Callable[[T], R],
        *,
        kind: str = "threads",
        max_workers: Optional[int] = None,
    ) -> "Stream[R]":
        def gen() -> Iterator[R]:
            with ParallelExecutor(kind=kind, max_workers=max_workers) as ex:
                yield from ex.run(fn, self.source)
        return Stream(gen())

    def filter(self, pred: Callable[[T], bool]) -> "Stream[T]":
        return Stream(x for x in self.source if pred(x))

    def batch(self, size: int) -> "Stream[Sequence[T]]":
        def gen() -> Iterator[Sequence[T]]:
            buf: List[T] = []
            for x in self.source:
                buf.append(x)
                if len(buf) >= size:
                    yield tuple(buf)
                    buf.clear()
            if buf:
                yield tuple(buf)
        return Stream(gen())

    def collect(self) -> List[T]:
        return list(self.source)
