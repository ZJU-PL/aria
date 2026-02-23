# pylint: disable=invalid-name
"""OpenAI provider (GPT, o1, o3 models)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.providers.cli.base import LLMResponse

try:
    from openai import OpenAI  # pylint: disable=import-error

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


class OpenAIProvider(BaseProvider):
    """OpenAI provider for GPT, o1, o3 models."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference with OpenAI."""
        if not OPENAI_AVAILABLE:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="OpenAI SDK not installed",
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="OPENAI_API_KEY is not set",
            )

        return self._call_api(message, system_role, temperature, max_output_length, api_key)

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        api_key: str,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Call OpenAI API."""
        assert OpenAI is not None
        client = OpenAI(api_key=api_key)

        kwargs: Dict[str, Any] = {
            "model": model_name or "gpt-4",
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": message},
            ],
        }

        if not (model_name or "").startswith("o"):
            kwargs["temperature"] = temperature

        raw = client.chat.completions.create(**kwargs)
        choice = raw.choices[0]
        usage = {}
        if raw.usage:
            usage = {
                "prompt_tokens": raw.usage.prompt_tokens or 0,
                "completion_tokens": raw.usage.completion_tokens or 0,
                "total_tokens": raw.usage.total_tokens or 0,
            }

        return InferenceResult(
            content=choice.message.content or "",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.finish_reason,
            usage=usage,
        )
