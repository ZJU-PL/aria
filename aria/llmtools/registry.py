# pylint: disable=invalid-name
"""Provider registry and routing logic."""

from __future__ import annotations

from typing import Optional

from aria.llmtools.core.base import BaseProvider
from aria.llmtools.logger import Logger
from aria.llmtools.providers.local.cli import LocalCLIProvider
from aria.llmtools.providers.local.lm_studio import LMStudioProvider
from aria.llmtools.providers.local.sglang import SGLangProvider
from aria.llmtools.providers.local.vllm import VLLMProvider
from aria.llmtools.providers.online.claude import ClaudeProvider
from aria.llmtools.providers.online.codex import OpenAICodexProvider
from aria.llmtools.providers.online.deepseek import DeepSeekProvider
from aria.llmtools.providers.online.gemini import GeminiProvider
from aria.llmtools.providers.online.kilocode import KiloCodeProvider
from aria.llmtools.providers.online.openai import OpenAIProvider
from aria.llmtools.providers.online.opencode import OpenCodeProvider


def get_online_provider(model_name: str) -> Optional[BaseProvider]:
    """
    Get online provider based on model name.

    Args:
        model_name: Model name (e.g., "gpt-4", "claude", "gemini-2.0-flash")

    Returns:
        Provider instance or None if unsupported
    """
    model_lower = model_name.lower()

    if model_name.startswith("opencode/"):
        return OpenCodeProvider()
    if model_name.startswith("kilo/"):
        return KiloCodeProvider()
    if "openai-codex" in model_lower or "openai_codex" in model_lower:
        return OpenAICodexProvider()
    if (
        model_lower.startswith("gpt")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
    ):
        return OpenAIProvider()
    if "gemini" in model_lower:
        return GeminiProvider()
    if "claude" in model_lower:
        return ClaudeProvider()
    if "deepseek" in model_lower:
        return DeepSeekProvider()

    return None


def get_local_provider(
    model_name: str, provider: str, logger: Optional[Logger] = None, temperature: float = 0.0
) -> Optional[BaseProvider]:
    """
    Get local provider based on provider type.

    Args:
        model_name: Model name
        provider: Provider type ("lm-studio", "vllm", "sglang", "kilo-cli", "opencode-cli")
        logger: Logger instance (required for CLI providers)
        temperature: Temperature setting (required for CLI providers)

    Returns:
        Provider instance or None if unsupported
    """
    if provider == "lm-studio":
        return LMStudioProvider()
    if provider == "vllm":
        return VLLMProvider()
    if provider == "sglang":
        return SGLangProvider()
    if provider in ["kilo-cli", "opencode-cli"]:
        if logger is None:
            return None
        return LocalCLIProvider(model_name, logger, temperature)

    return None
