# pylint: disable=invalid-name
"""Anthropic Claude provider."""

from __future__ import annotations

import os
from typing import Dict, Optional

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.core.results import error_result, result_from_usage

try:
    from anthropic import Anthropic  # pylint: disable=import-error
    ANTHROPIC_AVAILABLE = True
except ImportError:
    Anthropic = None  # type: ignore
    ANTHROPIC_AVAILABLE = False


MODEL_MAPPING = {
    "claude": "claude-3-5-sonnet-20241022",
    "claude-3": "claude-3-sonnet-20240229",
    "claude-3.5": "claude-3-5-sonnet-20241022",
    "claude-3-5": "claude-3-5-sonnet-20241022",
}


class ClaudeProvider(BaseProvider):
    """Anthropic Claude provider."""

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference with Claude."""
        if not ANTHROPIC_AVAILABLE:
            return error_result("Anthropic SDK not installed")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return error_result("ANTHROPIC_API_KEY is not set")

        return self._call_api(
            message,
            system_role,
            temperature,
            max_output_length,
            api_key,
            model_name,
        )

    def _call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        api_key: str,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Call Claude API."""
        model_id = model_name or "claude"
        if model_id in MODEL_MAPPING:
            model_id = MODEL_MAPPING[model_id]

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=max_output_length,
            temperature=temperature,
            system=system_role,
            messages=[{"role": "user", "content": message}],
        )

        first = response.content[0] if response.content else None
        text = getattr(first, "text", "") if first is not None else ""

        return result_from_usage(content=text, finish_reason="stop")
