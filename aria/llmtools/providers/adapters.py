"""Provider adapter base classes for inference-oriented clients."""

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI  # pylint: disable=import-error

from aria.llmtools.core.async_utils import run_async
from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.core.results import (
    error_result,
    result_from_llm_response,
    result_from_openai_response,
)
from aria.llmtools.core.responses import LLMResponse


def build_messages(system_role: str, message: str) -> List[Dict[str, str]]:
    """Build the standard two-message chat payload."""
    return [
        {"role": "system", "content": system_role},
        {"role": "user", "content": message},
    ]


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
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference through the provider chat adapter."""
        response = run_async(
            self.create_response(
                messages=build_messages(system_role, message),
                temperature=temperature,
                max_output_length=max_output_length,
                model_name=model_name,
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
    """Base class for providers backed by the OpenAI chat completions API."""

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
        model_name: Optional[str] = None,
    ) -> InferenceResult:
        """Run inference against an OpenAI-compatible endpoint."""
        api_key = self.get_api_key()
        if api_key is None:
            return error_result(self.get_missing_api_key_error())

        return self.call_api(
            message=message,
            system_role=system_role,
            temperature=temperature,
            max_output_length=max_output_length,
            api_key=api_key,
            model_name=model_name,
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
