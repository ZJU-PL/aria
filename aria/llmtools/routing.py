# pylint: disable=invalid-name
"""Provider routing and lazy loading for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Optional, Sequence, Tuple

from aria.llmtools.core.base import BaseProvider
from aria.llmtools.core.client import ProviderResolution
from aria.llmtools.core.logger import Logger

Matcher = Callable[[str, str], bool]
ProviderPath = Tuple[str, str]


@dataclass(frozen=True)
class ProviderSpec:
    """Lazy import specification for a provider implementation."""

    module_name: str
    class_name: str
    timeout: int

    def build(self) -> BaseProvider:
        """Import and instantiate the provider implementation on demand."""
        provider_module = import_module(self.module_name)
        provider_cls = getattr(provider_module, self.class_name)
        return provider_cls()

    def to_resolution(self) -> ProviderResolution:
        """Convert the spec into an executable provider resolution."""
        return ProviderResolution(provider=self.build(), timeout=self.timeout)


@dataclass(frozen=True)
class Route:
    """Route a model name pattern to a provider spec."""

    matches: Matcher
    provider: ProviderSpec


def _starts_with(prefix: str) -> Matcher:
    return lambda model_name, _model_lower: model_name.startswith(prefix)


def _contains(text: str) -> Matcher:
    return lambda _model_name, model_lower: text in model_lower


def _starts_with_lower(prefix: str) -> Matcher:
    return lambda _model_name, model_lower: model_lower.startswith(prefix)


ROUTED_PROVIDERS: Sequence[Route] = (
    Route(
        matches=_starts_with_lower("gpt"),
        provider=ProviderSpec(
            "aria.llmtools.providers.online.openai",
            "OpenAIProvider",
            100,
        ),
    ),
    Route(
        matches=_starts_with_lower("o1"),
        provider=ProviderSpec(
            "aria.llmtools.providers.online.openai",
            "OpenAIProvider",
            100,
        ),
    ),
    Route(
        matches=_starts_with_lower("o3"),
        provider=ProviderSpec(
            "aria.llmtools.providers.online.openai",
            "OpenAIProvider",
            100,
        ),
    ),
    Route(
        matches=_contains("gemini"),
        provider=ProviderSpec(
            "aria.llmtools.providers.online.gemini",
            "GeminiProvider",
            100,
        ),
    ),
    Route(
        matches=_contains("claude"),
        provider=ProviderSpec(
            "aria.llmtools.providers.online.claude",
            "ClaudeProvider",
            100,
        ),
    ),
    Route(
        matches=_contains("deepseek"),
        provider=ProviderSpec(
            "aria.llmtools.providers.online.deepseek",
            "DeepSeekProvider",
            100,
        ),
    ),
)

PROVIDER_HINTS: Dict[str, ProviderSpec] = {
    "lm-studio": ProviderSpec(
        "aria.llmtools.providers.local.lm_studio",
        "LMStudioProvider",
        300,
    ),
    "vllm": ProviderSpec(
        "aria.llmtools.providers.local.vllm",
        "VLLMProvider",
        300,
    ),
    "sglang": ProviderSpec(
        "aria.llmtools.providers.local.sglang",
        "SGLangProvider",
        300,
    ),
}


def resolve_provider(
    model_name: str,
    provider: Optional[str] = None,
    logger: Optional[Logger] = None,
    temperature: float = 0.0,
) -> Optional[ProviderResolution]:
    """Resolve any supported provider from the model and optional provider hint."""
    del logger, temperature
    if provider is not None:
        provider_spec = PROVIDER_HINTS.get(provider)
        if provider_spec is None:
            return None
        return provider_spec.to_resolution()

    model_lower = model_name.lower()
    for route in ROUTED_PROVIDERS:
        if route.matches(model_name, model_lower):
            return route.provider.to_resolution()

    return None
