# pylint: disable=invalid-name
"""Kilo Code provider wrapper."""

from __future__ import annotations

import os
from typing import Optional

from aria.llmtools.core.async_utils import run_async
from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.providers.cli.kilocode import chat_kilocode


class KiloCodeProvider(BaseProvider):
    """Kilo Code provider wrapper."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference with Kilo Code."""
        return self._call_api(message, system_role, temperature, max_output_length)

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Call Kilo Code API."""
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": message},
        ]

        api_key = os.environ.get("KILOCODE_API_KEY", "")

        result = run_async(
            chat_kilocode(
                model=model_name or "kilo/z-ai/glm-4.5-air:free",
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_length,
                api_key=api_key,
            )
        )

        if result.finish_reason == "error":
            return InferenceResult(
                content="",
                input_tokens=0,
                output_tokens=0,
                finish_reason="error",
                error=result.content or "Kilo Code error",
            )

        return InferenceResult(
            content=result.content or "",
            input_tokens=result.usage.get("prompt_tokens", 0),
            output_tokens=result.usage.get("completion_tokens", 0),
            finish_reason=result.finish_reason,
            usage=result.usage,
        )
