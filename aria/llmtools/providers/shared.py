"""Shared helpers for provider implementations."""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from aria.llmtools.core.async_utils import run_async
from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.providers.cli.base import LLMResponse

try:
    from openai import OpenAI  # pylint: disable=import-error

    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False


def build_messages(system_role: str, message: str) -> List[Dict[str, str]]:
    """Build the standard two-message chat payload."""
    return [
        {"role": "system", "content": system_role},
        {"role": "user", "content": message},
    ]


def error_result(error: str) -> InferenceResult:
    """Create a normalized provider error result."""
    return InferenceResult(
        content="",
        input_tokens=0,
        output_tokens=0,
        finish_reason="error",
        error=error,
    )


def result_from_usage(
    content: str,
    finish_reason: str,
    usage: Optional[Dict[str, int]] = None,
) -> InferenceResult:
    """Create a normalized success result."""
    normalized_usage = usage or {}
    return InferenceResult(
        content=content,
        input_tokens=normalized_usage.get("prompt_tokens", 0),
        output_tokens=normalized_usage.get("completion_tokens", 0),
        finish_reason=finish_reason,
        usage=normalized_usage,
    )


def result_from_llm_response(
    response: LLMResponse, default_error: str
) -> InferenceResult:
    """Normalize a CLI-provider response."""
    if response.finish_reason == "error":
        return error_result(response.content or default_error)

    return result_from_usage(
        content=response.content or "",
        finish_reason=response.finish_reason,
        usage=response.usage,
    )


def result_from_openai_response(response: Any) -> InferenceResult:
    """Normalize an OpenAI-compatible response."""
    choice = response.choices[0]
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens or 0,
            "completion_tokens": response.usage.completion_tokens or 0,
            "total_tokens": response.usage.total_tokens or 0,
        }

    return result_from_usage(
        content=choice.message.content or "",
        finish_reason=choice.finish_reason or "stop",
        usage=usage,
    )


class AsyncChatProvider(BaseProvider):
    """Base class for providers backed by async CLI adapters."""

    default_model: str = ""
    error_name: str = "Provider"

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference through the provider chat adapter."""
        response = run_async(
            self.create_response(
                messages=build_messages(system_role, message),
                temperature=temperature,
                max_output_length=max_output_length,
                model_name=None,
            )
        )
        return result_from_llm_response(
            response, default_error="{0} error".format(self.error_name)
        )

    @abstractmethod
    async def create_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_output_length: int,
        model_name: Optional[str] = None,
    ) -> LLMResponse:
        """Create a provider response."""
        raise NotImplementedError


class OpenAICompatibleProvider(BaseProvider):
    """Base class for OpenAI-compatible providers."""

    default_model: str = ""
    base_url: Optional[str] = None
    api_key_envs: Tuple[str, ...] = ()
    static_api_key: Optional[str] = None
    supports_max_tokens: bool = True

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference against an OpenAI-compatible endpoint."""
        if not OPENAI_AVAILABLE:
            return error_result("OpenAI SDK not installed")

        api_key = self.get_api_key()
        if api_key is None:
            return error_result(self.get_missing_api_key_error())

        return self.call_api(
            message=message,
            system_role=system_role,
            temperature=temperature,
            max_output_length=max_output_length,
            api_key=api_key,
        )

    def get_api_key(self) -> Optional[str]:
        """Resolve the API key or fixed token for the provider."""
        if self.static_api_key is not None:
            return self.static_api_key

        for env_name in self.api_key_envs:
            api_key = os.environ.get(env_name)
            if api_key:
                return api_key
        return None

    def get_missing_api_key_error(self) -> str:
        """Return the provider-specific missing-key error."""
        if not self.api_key_envs:
            return "API key is not set"
        return "{0} is not set".format(self.api_key_envs[0])

    def get_model_name(self, model_name: Optional[str] = None) -> str:
        """Resolve the model name to send upstream."""
        return model_name or self.default_model

    def should_send_temperature(self, model_name: str) -> bool:
        """Return whether temperature should be included."""
        return True

    def call_api(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
        api_key: str,
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Execute the OpenAI-compatible request."""
        assert OpenAI is not None
        model = self.get_model_name(model_name)
        client = OpenAI(api_key=api_key, base_url=self.base_url)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": build_messages(system_role, message),
        }
        if self.should_send_temperature(model):
            kwargs["temperature"] = temperature
        if self.supports_max_tokens:
            kwargs["max_tokens"] = max_output_length

        response = client.chat.completions.create(**kwargs)
        return result_from_openai_response(response)
