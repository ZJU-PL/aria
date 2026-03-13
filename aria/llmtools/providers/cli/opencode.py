"""OpenCode Zen provider support for free LLM access without API keys."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import AsyncOpenAI

from aria.llmtools.providers.cli.base import (
    LLMResponse,
    error_response,
    parse_openai_chat_response,
)
from aria.llmtools.providers.shared import AsyncChatProvider

OPENCODE_ZEN_BASE = "https://opencode.ai/zen/v1"

OPENCODE_FREE_MODELS = {
    "big-pickle": "Big Pickle (GLM-4.6 backend)",
    "glm-5-free": "GLM 5 Free",
    "kimi-k2.5-free": "Kimi K2.5 Free",
}


class OpenCodeProvider(AsyncChatProvider):
    """`BaseProvider` adapter for OpenCode-backed models."""

    default_model = "opencode/big-pickle"
    error_name = "OpenCode"

    async def create_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> LLMResponse:
        """Call OpenCode and normalize into the shared CLI response shape."""
        model = strip_opencode_prefix(model_name or self.default_model)
        return await chat_opencode(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_length,
        )


class OpenCodeAsyncOpenAI(AsyncOpenAI):
    """AsyncOpenAI subclass that accepts empty API keys for OpenCode Zen."""

    def __init__(self, *, api_key: Optional[str] = None, **kwargs: Any) -> None:
        is_opencode = kwargs.get("base_url", "").startswith("https://opencode.ai")

        if is_opencode and api_key == "":
            super().__init__(api_key="opencode-temp-key", **kwargs)
            self.api_key = ""
        else:
            super().__init__(api_key=api_key, **kwargs)


def is_opencode_model(model: str) -> bool:
    """Check if a model is an OpenCode model."""
    return model.startswith("opencode/")


def strip_opencode_prefix(model: str) -> str:
    """Remove opencode/ prefix from model name."""
    return model.replace("opencode/", "")


async def chat_opencode(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    job_id: Optional[str] = None,
    channel: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> LLMResponse:
    """Handle OpenCode Zen chat completion with bounded retries."""
    del job_id, channel, chat_id

    if "kimi-k2.5" in model.lower():
        temperature = 1.0

    client = OpenCodeAsyncOpenAI(
        api_key="",
        base_url=OPENCODE_ZEN_BASE,
    )

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    max_retries = 5
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(**kwargs)
            return parse_openai_chat_response(response)
        except Exception as exc:  # pragma: no cover - network paths
            error_str = str(exc)
            is_500_error = "500" in error_str or "Internal Server Error" in error_str
            if is_500_error and attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "OpenCode 500 error (attempt {0}/{1}), retrying in {2}s: {3}".format(
                        attempt + 1,
                        max_retries,
                        delay,
                        error_str,
                    )
                )
                await asyncio.sleep(delay)
                continue
            return error_response("Error calling OpenCode Zen: {0}".format(error_str))

    return error_response("Error calling OpenCode Zen: Max retries exceeded")
