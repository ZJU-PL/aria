"""LLM provider chat protocol and OpenAI-response parsing helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aria.llmtools.core.responses import LLMResponse, ToolCallRequest


class LLMProvider(ABC):
    """Abstract base class for provider chat adapters."""

    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_retry: Optional[Callable[[int, int, float], Awaitable[None]]] = None,
    ) -> LLMResponse:
        """Send a chat completion request."""
        raise NotImplementedError

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        raise NotImplementedError


def error_response(message: str) -> LLMResponse:
    """Create a normalized error response."""
    return LLMResponse(content=message, finish_reason="error")


def parse_openai_chat_response(response: Any) -> LLMResponse:
    """Parse OpenAI-compatible SDK response into the common LLMResponse format."""
    if not getattr(response, "choices", None):
        error_msg = (
            getattr(response, "error", None) or "Unknown error: no choices returned"
        )
        return error_response("API error: {0}".format(error_msg))

    choice = response.choices[0]
    message = choice.message

    tool_calls: List[ToolCallRequest] = []
    raw_tool_calls = getattr(message, "tool_calls", None)
    if raw_tool_calls:
        for tool_call in raw_tool_calls:
            args: Any = tool_call.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            if not isinstance(args, dict):
                args = {"value": args}

            tool_calls.append(
                ToolCallRequest(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=args,
                )
            )

    usage: Dict[str, int] = {}
    response_usage = getattr(response, "usage", None)
    if response_usage:
        prompt_tokens = getattr(response_usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response_usage, "completion_tokens", 0) or 0
        total_tokens = getattr(response_usage, "total_tokens", 0) or 0
        usage = {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(total_tokens),
        }

    thinking_content = None
    reasoning_content = getattr(message, "reasoning_content", None)
    if reasoning_content:
        thinking_content = reasoning_content
    elif getattr(message, "thinking", None):
        thinking_content = message.thinking

    content = getattr(message, "content", None)
    if not content and thinking_content:
        content = thinking_content
        thinking_content = None

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
        usage=usage,
        reasoning_content=reasoning_content,
        thinking=thinking_content,
    )
