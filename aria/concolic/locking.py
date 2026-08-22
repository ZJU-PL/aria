"""Synchronization primitives for Z3's process-global Python context."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

# z3py's default global context is not safe for concurrent calls in every
# supported wheel/platform combination. Keep the native backend deterministic
# and memory-safe until it moves each worker to an isolated Z3 Context.
_Z3_GLOBAL_LOCK = RLock()


@contextmanager
def z3_global_lock() -> Iterator[None]:
    with _Z3_GLOBAL_LOCK:
        yield
