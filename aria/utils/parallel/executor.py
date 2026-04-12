"""Lightweight parallel execution helpers.

Features:
- Unified process/thread pool via a single class
- Submit, map, and gather results
- Optional task-level logging (start/end/duration)
- Graceful shutdown and timeouts
"""

from __future__ import annotations

from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


T = TypeVar("T")
R = TypeVar("R")


PoolKind = Union[ThreadPoolExecutor, ProcessPoolExecutor]


@dataclass
class ParallelExecutor:
    """Unified wrapper around thread/process pools.

    kind: "threads" or "processes".
    """

    max_workers: Optional[int] = None
    kind: str = "threads"  # "threads" | "processes"
    initializer: Optional[Callable[..., None]] = None
    initargs: Optional[tuple] = None
    log_events: bool = False
    logger: Optional[logging.Logger] = None
    cancel_on_error: bool = True
    kill_pool_on_timeout: bool = False

    # Internal state — excluded from __init__, __repr__, and equality checks so
    # that dataclasses.replace() produces a clean new executor rather than
    # copying an already-running pool.
    _pool: PoolKind = field(init=False, repr=False, compare=False)
    _shutdown_requested: bool = field(init=False, repr=False, compare=False)
    _fast_shutdown: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._shutdown_requested = False
        self._fast_shutdown = False
        if self.kind not in ("threads", "processes"):
            raise ValueError("kind must be 'threads' or 'processes'")
        if self.logger is None:
            self.logger = logging.getLogger("aria.parallel")
        if self.kind == "threads":
            self._pool: PoolKind = ThreadPoolExecutor(
                max_workers=self.max_workers,
                initializer=self.initializer,
                initargs=self.initargs or (),
            )
        else:
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=self.initializer,
                initargs=self.initargs or (),
            )

    def submit(self, fn: Callable[..., R], *args: Any, **kwargs: Any):
        if self.log_events:
            # Use a top-level wrapper to remain picklable for processes
            logger_name = self.logger.name if self.logger else None
            return self._pool.submit(
                _execute_with_logging, fn, args, kwargs, logger_name
            )
        return self._pool.submit(fn, *args, **kwargs)

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> List[R]:
        return list(self._pool.map(fn, items))

    def iter_map(self, fn: Callable[[T], R], items: Iterable[T]) -> Iterable[R]:
        return self._pool.map(fn, items)

    def shutdown(
        self, wait_for_completion: bool = True, cancel_futures: bool = False
    ) -> None:
        if self._shutdown_requested:
            return
        if self._fast_shutdown:
            wait_for_completion = False
            cancel_futures = True
        self._shutdown_requested = True
        self._pool.shutdown(wait=wait_for_completion, cancel_futures=cancel_futures)

    def run(
        self,
        fn: Callable[[T], R],
        items: Iterable[T],
        timeout: Optional[float] = None,
        *,
        return_exceptions: bool = False,
        preserve_order: bool = True,
    ) -> List[R]:
        """Run fn over items and gather results.

        preserve_order controls whether the returned list matches the submission
        order even when tasks finish out of order.
        """
        try:
            futures = [self.submit(fn, item) for item in items]
            return _gather_results(
                futures,
                timeout=timeout,
                logger=self.logger,
                cancel_on_error=self.cancel_on_error,
                kill_pool_on_timeout=self.kill_pool_on_timeout,
                on_timeout=self._terminate_pool,
                on_timeout_return=self._mark_fast_shutdown,
                return_exceptions=return_exceptions,
                preserve_order=preserve_order,
            )
        except Exception:
            self._fast_shutdown = True
            raise

    def _terminate_pool(self) -> None:
        kill_workers = getattr(self._pool, "kill_workers", None)
        terminate_workers = getattr(self._pool, "terminate_workers", None)

        if callable(kill_workers):
            kill_workers()
            return
        if callable(terminate_workers):
            terminate_workers()
            return

        self.shutdown(wait_for_completion=False, cancel_futures=True)

    def _mark_fast_shutdown(self) -> None:
        self._fast_shutdown = True

    # Optional context manager convenience
    def __enter__(self) -> "ParallelExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    kind: str = "threads",
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Optional[tuple] = None,
) -> List[R]:
    """Convenience parallel map over a sequence.

    Example:
        results = parallel_map(work, data, kind="processes", max_workers=8)
    """
    ex = ParallelExecutor(
        max_workers=max_workers, kind=kind, initializer=initializer, initargs=initargs
    )
    try:
        return ex.run(fn, items, timeout=timeout)
    finally:
        ex.shutdown()


def run_tasks(
    tasks: Sequence[Tuple[Callable[..., R], tuple, dict]],
    *,
    kind: str = "threads",
    max_workers: Optional[int] = None,
    timeout: Optional[float] = None,
    initializer: Optional[Callable[..., None]] = None,
    initargs: Optional[tuple] = None,
) -> List[R]:
    """Run heterogeneous callables with different args/kwargs.

    tasks: list of (callable, args, kwargs)
    """
    ex = ParallelExecutor(
        max_workers=max_workers, kind=kind, initializer=initializer, initargs=initargs
    )
    try:
        futures = [ex.submit(fn, *args, **kwargs) for fn, args, kwargs in tasks]
        try:
            return _gather_results(
                futures,
                timeout=timeout,
                logger=ex.logger,
                cancel_on_error=ex.cancel_on_error,
                kill_pool_on_timeout=ex.kill_pool_on_timeout,
                on_timeout=ex._terminate_pool,
                on_timeout_return=ex._mark_fast_shutdown,
                return_exceptions=False,
                preserve_order=True,
            )
        except Exception:
            ex._fast_shutdown = True
            raise
    finally:
        ex.shutdown()


def _execute_with_logging(
    fn: Callable[..., R], args: tuple, kwargs: dict, logger_name: Optional[str]
) -> R:
    """Execute a callable, logging start/end/duration using a named logger.

    This function is top-level so it's picklable for process pools.
    """
    logger = logging.getLogger(logger_name) if logger_name else None
    name = getattr(fn, "__name__", repr(fn))
    start = time.time()
    if logger:
        logger.debug("task.start name=%s", name)
    try:
        return fn(*args, **kwargs)
    finally:
        if logger:
            logger.debug("task.end name=%s elapsed=%.6fs", name, time.time() - start)


def _gather_results(
    futures: Sequence,
    *,
    timeout: Optional[float],
    logger: Optional[logging.Logger],
    cancel_on_error: bool,
    kill_pool_on_timeout: bool,
    on_timeout: Optional[Callable[[], None]],
    on_timeout_return: Optional[Callable[[], None]],
    return_exceptions: bool,
    preserve_order: bool,
) -> List:
    """Collect results with robust handling for exceptions and timeouts.

    - Preserves submission order
    - On exception: optionally cancels remaining futures and re-raises
      (or returns exceptions)
    - On timeout: cancels remaining futures; optionally signals to kill pool
      by raising TimeoutError
    """
    if timeout is not None and timeout < 0:
        raise ValueError("timeout must be non-negative")

    results: Dict[int, Any] = {}
    completion_order: List[Any] = []
    pending: Dict[Any, int] = {f: idx for idx, f in enumerate(futures)}
    deadline = None if timeout is None else time.monotonic() + timeout

    def _remaining_time() -> Optional[float]:
        if deadline is None:
            return None
        return max(0.0, deadline - time.monotonic())

    def _cancel_all() -> None:
        for rem in pending:
            if not rem.done():
                rem.cancel()

    while pending:
        remaining = _remaining_time()
        done, _ = wait(pending.keys(), timeout=remaining, return_when=FIRST_COMPLETED)
        if not done:
            # timeout expired
            _cancel_all()
            if kill_pool_on_timeout:
                if on_timeout is not None:
                    on_timeout()
                raise TimeoutError("parallel execution timed out")
            if return_exceptions:
                if on_timeout_return is not None:
                    on_timeout_return()
                for idx in pending.values():
                    exc = TimeoutError("parallel execution timed out")
                    results[idx] = exc
                    completion_order.append(exc)
                break
            raise TimeoutError("parallel execution timed out")

        for fut in done:
            idx = pending.pop(fut)
            try:
                result = fut.result(timeout=0)
                results[idx] = result
                completion_order.append(result)
            except Exception as exc:  # includes TimeoutError
                if logger:
                    logger.error("task.error type=%s msg=%s", type(exc).__name__, exc)
                if return_exceptions:
                    results[idx] = exc
                    completion_order.append(exc)
                    continue
                if cancel_on_error:
                    _cancel_all()
                if isinstance(exc, TimeoutError) and kill_pool_on_timeout:
                    _cancel_all()
                    if on_timeout is not None:
                        on_timeout()
                    raise TimeoutError("parallel execution timed out") from exc
                raise

    if preserve_order:
        return [results[i] for i in sorted(results)]
    return completion_order
