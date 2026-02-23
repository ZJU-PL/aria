# pylint: disable=invalid-name
"""vLLM provider (OpenAI-compatible local endpoint)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult

try:
    from openai import OpenAI  # pylint: disable=import-error

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


class VLLMProvider(BaseProvider):
    """vLLM provider (OpenAI-compatible)."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference with vLLM."""
        if not OPENAI_AVAILABLE:
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error="OpenAI SDK not installed",
            )

        return self._call_api(message, system_role, temperature, max_output_length)

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
    ) -> InferenceResult:
        """Call vLLM API."""
        assert OpenAI is not None
        client = OpenAI(api_key="vllm", base_url=base_url)

        raw = client.chat.completions.create(
            model=model_name or "local-model",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": message},
            ],
            temperature=temperature,
            max_tokens=max_output_length,
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
