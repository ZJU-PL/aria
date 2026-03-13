# pylint: disable=invalid-name
"""DeepSeek provider."""

from __future__ import annotations

from aria.llmtools.providers.adapters import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """DeepSeek provider (OpenAI-compatible)."""

    default_model = "deepseek-chat"
    base_url = "https://api.deepseek.com"
    api_key_envs = ("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY2")
    supports_max_tokens = False
