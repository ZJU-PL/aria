# pylint: disable=invalid-name
"""Provider registry and routing logic."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, List, Optional, Tuple

from aria.llmtools.core.base import BaseProvider
from aria.llmtools.logger import Logger

OnlineMatcher = Callable[[str, str], bool]
ProviderPath = Tuple[str, str]


def _has_opencode_prefix(model_name: str, _model_lower: str) -> bool:
    return model_name.startswith("opencode/")


def _has_kilo_prefix(model_name: str, _model_lower: str) -> bool:
    return model_name.startswith("kilo/")


def _is_codex(_model_name: str, model_lower: str) -> bool:
    return "openai-codex" in model_lower or "openai_codex" in model_lower


def _is_openai(_model_name: str, model_lower: str) -> bool:
    return (
        model_lower.startswith("gpt")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
    )


def _contains_gemini(_model_name: str, model_lower: str) -> bool:
    return "gemini" in model_lower


def _contains_claude(_model_name: str, model_lower: str) -> bool:
    return "claude" in model_lower


def _contains_deepseek(_model_name: str, model_lower: str) -> bool:
    return "deepseek" in model_lower


ONLINE_PROVIDER_RULES: List[Tuple[OnlineMatcher, ProviderPath]] = [
    (
        _has_opencode_prefix,
        ("aria.llmtools.providers.cli.opencode", "OpenCodeProvider"),
    ),
    (
        _has_kilo_prefix,
        ("aria.llmtools.providers.cli.kilocode", "KiloCodeProvider"),
    ),
    (_is_codex, ("aria.llmtools.providers.cli.codex", "CodexProvider")),
    (_is_openai, ("aria.llmtools.providers.online.openai", "OpenAIProvider")),
    (
        _contains_gemini,
        ("aria.llmtools.providers.online.gemini", "GeminiProvider"),
    ),
    (
        _contains_claude,
        ("aria.llmtools.providers.online.claude", "ClaudeProvider"),
    ),
    (
        _contains_deepseek,
        ("aria.llmtools.providers.online.deepseek", "DeepSeekProvider"),
    ),
]

LOCAL_PROVIDER_FACTORIES = {
    "lm-studio": ("aria.llmtools.providers.local.lm_studio", "LMStudioProvider"),
    "vllm": ("aria.llmtools.providers.local.vllm", "VLLMProvider"),
    "sglang": ("aria.llmtools.providers.local.sglang", "SGLangProvider"),
}

CLI_LOCAL_PROVIDERS = {"kilo-cli", "opencode-cli"}


def _load_provider(provider_path: ProviderPath, *args: Any) -> BaseProvider:
    """Import and instantiate a provider on demand."""
    module_name, class_name = provider_path
    provider_module = import_module(module_name)
    provider_cls = getattr(provider_module, class_name)
    return provider_cls(*args)


def get_online_provider(model_name: str) -> Optional[BaseProvider]:
    """
    Get online provider based on model name.

    Args:
        model_name: Model name (e.g., "gpt-4", "claude", "gemini-2.0-flash")

    Returns:
        Provider instance or None if unsupported
    """
    model_lower = model_name.lower()

    for matches, provider_path in ONLINE_PROVIDER_RULES:
        if matches(model_name, model_lower):
            return _load_provider(provider_path)

    return None


def get_local_provider(
    model_name: str,
    provider: str,
    logger: Optional[Logger] = None,
    temperature: float = 0.0,
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
    provider_cls = LOCAL_PROVIDER_FACTORIES.get(provider)
    if provider_cls is not None:
        return _load_provider(provider_cls)

    if provider in CLI_LOCAL_PROVIDERS:
        if logger is None:
            return None
        return _load_provider(
            ("aria.llmtools.providers.cli.local_cli", "LocalCLIProvider"),
            model_name,
            logger,
            temperature,
        )

    return None
