"""Async execution utilities."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Dict


def run_async(coro: Any) -> Any:
    """
    Run a coroutine from sync contexts, including running event loops.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine

    Raises:
        Exception: If the coroutine raises an exception
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            box["error"] = exc

    thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = thread.submit(_runner)
    future.result()
    thread.shutdown(wait=True)

    if "error" in box:
        raise box["error"]
    return box.get("value")
