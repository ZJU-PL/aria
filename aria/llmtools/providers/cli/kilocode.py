"""Kilo Code provider support (OpenAI-compatible with custom headers)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from aria.llmtools.providers.cli.base import (
    LLMResponse,
    error_response,
    parse_openai_chat_response,
)
from aria.llmtools.providers.shared import AsyncChatProvider

KILOCODE_BASE = "https://api.kilo.ai/api/openrouter"


class KiloCodeProvider(AsyncChatProvider):
    """`BaseProvider` adapter for Kilo-backed models."""

    default_model = "kilo/z-ai/glm-4.5-air:free"
    error_name = "Kilo Code"

    async def create_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> LLMResponse:
        """Call Kilo Code and normalize into the shared CLI response shape."""
        return await chat_kilocode(
            model=model_name or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_length,
            api_key=os.environ.get("KILOCODE_API_KEY", ""),
        )


class KiloCodeAsyncOpenAI(AsyncOpenAI):
    """AsyncOpenAI subclass that adds Kilo Code custom headers."""

    def __init__(self, *, api_key: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(api_key=api_key, **kwargs)
        self.default_headers.update(
            {
                "x-api-key": api_key or "",
                "X-KILOCODE-EDITORNAME": "custom",
            }
        )


def is_kilocode_model(model: str) -> bool:
    """Check if a model is a Kilo Code model."""
    return model.startswith("kilo/")


def strip_kilocode_prefix(model: str) -> str:
    """Remove kilo/ prefix from model name."""
    return model.replace("kilo/", "")


async def chat_kilocode(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
) -> LLMResponse:
    """Handle Kilo Code chat completion."""
    try:
        model_name = strip_kilocode_prefix(model)
        client = KiloCodeAsyncOpenAI(
            api_key=api_key or "",
            base_url=KILOCODE_BASE,
        )

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        return parse_openai_chat_response(response)
    except Exception as exc:  # pragma: no cover - network paths
        return error_response("Error calling Kilo Code: {0}".format(exc))
