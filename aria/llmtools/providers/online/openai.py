# pylint: disable=invalid-name
"""OpenAI provider (GPT, o1, o3 models)."""

from __future__ import annotations

from aria.llmtools.providers.adapters import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI provider for GPT models."""

    default_model = "gpt-5.4"
    api_key_envs = ("OPENAI_API_KEY",)

    def should_send_temperature(self, model_name: str) -> bool:
        """Reasoning models do not accept the temperature parameter."""
        return not model_name.startswith("o")
