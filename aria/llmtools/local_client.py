"""Client for OpenAI-compatible local LLM endpoints."""

from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Optional, Tuple

import tiktoken
from openai import OpenAI

from aria.llmtools.core.logger import Logger


class LLMLocal:
    """Local LLM inference client for OpenAI-compatible endpoints."""

    def __init__(
        self,
        offline_model_name: str,
        logger: Logger,
        temperature: float = 0.0,
        system_role: str = (
            "You are an experienced programmer and good at understanding "
            "programs written in mainstream programming languages."
        ),
        max_output_length: int = 4096,
        measure_cost: bool = False,
        provider: str = "lm-studio",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self.measure_cost = measure_cost
        self.offline_model_name = offline_model_name
        self.encoding = None
        if self.measure_cost:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
        self.temperature = temperature
        self.system_role = system_role
        self.logger = logger
        self.max_output_length = max_output_length
        self.provider = provider
        self.base_url = base_url or self._get_default_base_url(provider)
        self.api_key = api_key or self._get_default_api_key(provider)
        self.max_retries = max_retries

    def _get_default_base_url(self, provider: str) -> str:
        if provider == "tingly":
            return "http://localhost:12580/tingly/openai"
        return "http://localhost:1234/v1"

    def _get_default_api_key(self, provider: str) -> str:
        if provider == "tingly":
            return os.environ.get("TINGLY_API_KEY", "")
        return os.environ.get("LOCAL_OPENAI_API_KEY", "lm-studio")

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        """Backward-compatible tuple response API."""
        self.logger.print_log(self.offline_model_name, "is running")
        output = self.infer_with_openai_compatible_model(message)

        encoding = None
        if is_measure_cost:
            encoding = self.encoding or tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

        input_token_cost = (
            0
            if not is_measure_cost
            else len(encoding.encode(self.system_role)) + len(encoding.encode(message))
        )
        output_token_cost = (
            0 if not is_measure_cost else len(encoding.encode(output))
        )
        return output, input_token_cost, output_token_cost

    def run_with_timeout(self, func, timeout_seconds: int) -> str:
        """Run a function with a timeout in a worker thread."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                self.logger.print_log("Operation timed out")
                return ""
            except (ValueError, RuntimeError, ConnectionError) as exc:
                self.logger.print_log(f"Operation failed: {exc}")
                return ""

    def infer_with_openai_compatible_model(self, message: str) -> str:
        """Infer using an OpenAI-compatible API endpoint."""
        model_input = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": message},
        ]

        def call_api() -> str:
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.offline_model_name,
                messages=model_input,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        try_count = 0
        while try_count < self.max_retries:
            try:
                try_count += 1
                output = self.run_with_timeout(call_api, timeout_seconds=100)
                if output:
                    return output
            except Exception as exc:  # pragma: no cover - network path
                self.logger.print_log(f"API error: {exc}")
            if try_count < self.max_retries:
                time.sleep(2 ** (try_count - 1))

        return ""
