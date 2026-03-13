# pylint: disable=invalid-name
"""vLLM provider (OpenAI-compatible local endpoint)."""

from __future__ import annotations

from aria.llmtools.providers.adapters import OpenAICompatibleProvider


class VLLMProvider(OpenAICompatibleProvider):
    """vLLM provider (OpenAI-compatible)."""

    default_model = "local-model"
    base_url = "http://localhost:8000/v1"
    static_api_key = "vllm"
