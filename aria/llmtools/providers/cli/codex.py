"""OpenAI-compatible Codex provider."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from openai import AsyncOpenAI

try:
    from oauth_cli_kit import get_token as get_codex_token  # type: ignore[import-untyped]
except ImportError:
    get_codex_token = None

from aria.llmtools.providers.cli.base import (
    LLMProvider,
    LLMResponse,
    error_response,
    parse_openai_chat_response,
)
from aria.llmtools.providers.shared import AsyncChatProvider

DEFAULT_CODEX_BASE_URL = "http://localhost:12580/tingly/openai"


class OpenAICodexProvider(LLMProvider):
    """Use an OpenAI-compatible Codex endpoint."""

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.1-codex",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        super().__init__(api_key=api_key, api_base=api_base)
        self.default_model = default_model

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        on_retry: Optional[Callable[[int, int, float], Awaitable[None]]] = None,
    ) -> LLMResponse:
        client = AsyncOpenAI(
            api_key=await self._get_api_key(),
            base_url=self._get_base_url(),
        )

        kwargs: Dict[str, Any] = {
            "model": _strip_model_prefix(model or self.default_model),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(**kwargs)
                return parse_openai_chat_response(response)
            except Exception as exc:  # pragma: no cover - network paths
                if attempt < max_retries - 1:
                    delay = float(2**attempt)
                    if on_retry is not None:
                        await on_retry(attempt + 1, max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                return error_response("Error calling Codex: {0}".format(exc))

        return error_response("Error calling Codex")

    def get_default_model(self) -> str:
        return self.default_model

    async def _get_api_key(self) -> str:
        """Resolve the API key for the Codex endpoint."""
        if self.api_key:
            return self.api_key

        env_api_key = os.environ.get("ARIA_CODEX_API_KEY")
        if env_api_key:
            return env_api_key

        if get_codex_token is not None:
            token = await asyncio.to_thread(get_codex_token)
            return str(token.access)

        raise ImportError(
            "Set ARIA_CODEX_API_KEY or install oauth-cli-kit for Codex access."
        )

    def _get_base_url(self) -> str:
        """Resolve the OpenAI-compatible Codex base URL."""
        if self.api_base:
            return self.api_base
        return os.environ.get("ARIA_CODEX_BASE_URL", DEFAULT_CODEX_BASE_URL)


class CodexProvider(AsyncChatProvider):
    """`BaseProvider` adapter for Codex-backed models."""

    default_model = "openai-codex/gpt-5.1-codex"
    error_name = "Codex"

    async def create_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> LLMResponse:
        """Call Codex and normalize into the shared CLI response shape."""
        provider = OpenAICodexProvider(default_model=model_name or self.default_model)
        return await provider.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_length,
        )


def _strip_model_prefix(model: str) -> str:
    """Remove the local Codex routing prefix before calling the backend."""
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model
