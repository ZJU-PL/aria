# pylint: disable=invalid-name
"""LM Studio provider (OpenAI-compatible local endpoint)."""

from __future__ import annotations

from aria.llmtools.providers.shared import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    """LM Studio provider (OpenAI-compatible)."""

    default_model = "local-model"
    base_url = "http://localhost:1234/v1"
    static_api_key = "lm-studio"
