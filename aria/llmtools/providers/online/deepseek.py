# pylint: disable=invalid-name
"""DeepSeek provider."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult

try:
    from openai import OpenAI  # pylint: disable=import-error

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


class DeepSeekProvider(BaseProvider):
    """DeepSeek provider (OpenAI-compatible)."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference with DeepSeek."""
        if not OPENAI_AVAILABLE:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="OpenAI SDK not installed",
            )

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY2"
        )
        if not api_key:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="DEEPSEEK_API_KEY is not set",
            )

        return self._call_api(message, system_role, temperature, api_key)

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        api_key: str,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Call DeepSeek API."""
        assert OpenAI is not None
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        raw = client.chat.completions.create(
            model=model_name or "deepseek-chat",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
        )

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
