"""Retry logic with exponential backoff."""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Callable

from aria.llmtools.core.base import InferenceResult
from aria.llmtools.logger import Logger
from aria.llmtools.providers.cli.base import LLMResponse


def retry_with_backoff(
    call_func: Callable[[], Any],
    logger: Logger,
    timeout: int = 100,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> InferenceResult:
    """
    Retry API calls with exponential backoff.

    Args:
        call_func: Function to call (should return LLMResponse or str)
        logger: Logger instance
        timeout: Timeout in seconds for each attempt
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff

    Returns:
        InferenceResult with content and error information
    """
    last_error = "No response produced"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for attempt in range(max_retries):
            try:
                future = executor.submit(call_func)
                result = future.result(timeout=timeout)

                if isinstance(result, LLMResponse):
                    if result.finish_reason != "error":
                        return InferenceResult(
                            content=result.content or "",
                            input_tokens=result.usage.get("prompt_tokens", 0),
                            output_tokens=result.usage.get("completion_tokens", 0),
                            finish_reason=result.finish_reason,
                            usage=result.usage,
                        )
                    last_error = result.content or "Unknown provider error"
                elif isinstance(result, InferenceResult):
                    if result.error is None:
                        return result
                    last_error = result.error
                elif isinstance(result, str):
                    if result:
                        return InferenceResult(
                            content=result,
                            input_tokens=0,
                            output_tokens=0,
                            finish_reason="stop",
                        )
                    last_error = "Empty response"
                else:
                    last_error = "Unexpected result type"

            except concurrent.futures.TimeoutError:
                last_error = "Operation timed out"
                logger.print_log(last_error)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = "API error: {0}".format(exc)
                logger.print_log(last_error)

            if attempt < max_retries - 1:
                time.sleep(base_delay * (2**attempt))

    return InferenceResult(
        content="",
        input_tokens=0,
        output_tokens=0,
        finish_reason="error",
        error=last_error,
    )
