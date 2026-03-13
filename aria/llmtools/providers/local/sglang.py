# pylint: disable=invalid-name
"""SGLang provider (OpenAI-compatible local endpoint)."""

from __future__ import annotations

from aria.llmtools.providers.shared import OpenAICompatibleProvider


class SGLangProvider(OpenAICompatibleProvider):
    """SGLang provider (OpenAI-compatible)."""

    default_model = "local-model"
    base_url = "http://localhost:30000/v1"
    static_api_key = "sglang"
