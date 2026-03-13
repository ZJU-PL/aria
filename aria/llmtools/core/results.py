"""Helpers for constructing normalized inference results."""

from __future__ import annotations

from typing import Any, Dict, Optional

from aria.llmtools.core.base import InferenceResult
from aria.llmtools.core.responses import LLMResponse


def error_result(error: str) -> InferenceResult:
    """Create a normalized provider error result."""
    return InferenceResult(
        content="",
        input_tokens=0,
        output_tokens=0,
        finish_reason="error",
        error=error,
    )


def result_from_usage(
    content: str,
    finish_reason: str,
    usage: Optional[Dict[str, int]] = None,
) -> InferenceResult:
    """Create a normalized success result."""
    normalized_usage = usage or {}
    return InferenceResult(
        content=content,
        input_tokens=normalized_usage.get("prompt_tokens", 0),
        output_tokens=normalized_usage.get("completion_tokens", 0),
        finish_reason=finish_reason,
        usage=normalized_usage,
    )


def result_from_llm_response(
    response: LLMResponse, default_error: str
) -> InferenceResult:
    """Normalize a CLI-provider response."""
    if response.finish_reason == "error":
        return error_result(response.content or default_error)

    return result_from_usage(
        content=response.content or "",
        finish_reason=response.finish_reason,
        usage=response.usage,
    )


def result_from_openai_response(response: Any) -> InferenceResult:
    """Normalize an OpenAI-compatible response."""
    choice = response.choices[0]
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
            "total_tokens": response.usage.total_tokens or 0,
        }

    return result_from_usage(
        content=choice.message.content or "",
        finish_reason=choice.finish_reason or "stop",
        usage=usage,
    )
