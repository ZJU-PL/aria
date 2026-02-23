# pylint: disable=invalid-name
"""OpenCode provider wrapper."""

from __future__ import annotations

from typing import Optional

from aria.llmtools.core.async_utils import run_async
from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.providers.cli.opencode import chat_opencode, strip_opencode_prefix


class OpenCodeProvider(BaseProvider):
    """OpenCode Zen provider wrapper."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference with OpenCode."""
        return self._call_api(message, system_role, temperature, max_output_length)

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Call OpenCode API."""
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": message},
        ]

        model = strip_opencode_prefix(model_name or "opencode/big-pickle")

        result = run_async(
            chat_opencode(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_length,
            )
        )

        if result.finish_reason == "error":
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error=result.content or "OpenCode error",
            )

        return InferenceResult(
            content=result.content or "",
            input_tokens=result.usage.get("prompt_tokens", 0),
            output_tokens=result.usage.get("completion_tokens", 0),
            finish_reason=result.finish_reason,
            usage=result.usage,
        )
