"""
Calling local LLMs
  - vLLM
  - sglang
  - LMStudio
  - Tingly OpenAI proxy (set TINGLY_API_KEY)
"""

import concurrent.futures
import logging
import os
import time
from typing import Any, Tuple

import tiktoken
from openai import OpenAI


class SimpleLogger:
    """Simple logger interface for local LLM."""

    def __init__(self, log_level=logging.INFO):
        """Initialize simple logger."""
        pass

    def print_log(self, *args: Any) -> None:
        """Print log message."""
        pass

    def print_console(self, *args: Any) -> None:
        """Print console message."""
        pass


class LLMLocal:
    """Local LLM inference class for offline models."""

    def __init__(
        self,
        offline_model_name: str,
        logger: "SimpleLogger",
        temperature: float = 0.0,
        system_role: str = (
            "You are an experienced programmer and good at understanding "
            "programs written in mainstream programming languages."
        ),
        max_output_length: int = 4096,
        measure_cost: bool = False,
        provider: str = "lm-studio",  # vllm, sglang, lm-studio, etc.
        base_url: str = None,
        api_key: str = None,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize local LLM.

        Args:
            offline_model_name: Name of the offline model
            logger: Logger instance
            temperature: Temperature for inference
            system_role: System role message
            max_output_length: Maximum output length
            measure_cost: Whether to measure token cost
            provider: Provider name for endpoint defaults
            base_url: OpenAI-compatible base URL
            api_key: API key for the compatible endpoint
            max_retries: Maximum number of request retries
        """
        self.measure_cost = measure_cost
        self.offline_model_name = offline_model_name
        self.encoding = None
        if self.measure_cost:
            self.encoding = tiktoken.encoding_for_model(
                "gpt-3.5-turbo-0125"
            )  # We only use gpt-3.5 to measure token cost
        self.temperature = temperature
        self.system_role = system_role
        self.logger = logger
        self.max_output_length = max_output_length
        self.provider = provider
        self.base_url = base_url or self._get_default_base_url(provider)
        self.api_key = api_key or self._get_default_api_key(provider)
        self.max_retries = max_retries

    def _get_default_base_url(self, provider: str) -> str:
        """Get provider-specific default base URL."""
        if provider == "tingly":
            return "http://localhost:12580/tingly/openai"
        return "http://localhost:1234/v1"

    def _get_default_api_key(self, provider: str) -> str:
        """Get provider-specific API key from environment or fallback."""
        if provider == "tingly":
            return os.environ.get("TINGLY_API_KEY", "")
        return os.environ.get("LOCAL_OPENAI_API_KEY", "lm-studio")

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        """
        Infer using the local model.

        Args:
            message: Input message
            is_measure_cost: Whether to measure token cost

        Returns:
            Tuple of (output, input_token_cost, output_token_cost)
        """
        self.logger.print_log(self.offline_model_name, "is running")
        output = self.infer_with_openai_compatible_model(message)

        encoding = None
        if is_measure_cost:
            encoding = self.encoding or tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

        input_token_cost = (
            0
            if not is_measure_cost
            else len(encoding.encode(self.system_role))
            + len(encoding.encode(message))
        )
        output_token_cost = (
            0 if not is_measure_cost else len(encoding.encode(output))
        )
        return output, input_token_cost, output_token_cost

    def run_with_timeout(self, func, timeout_seconds):
        """
        Run a function with timeout that works in multiple threads.

        Args:
            func: Function to execute
            timeout_seconds: Timeout in seconds

        Returns:
            Function result or empty string on timeout/error
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                self.logger.print_log("Operation timed out")
                return ""
            except (ValueError, RuntimeError, ConnectionError) as e:
                self.logger.print_log(f"Operation failed: {e}")
                return ""

    def infer_with_openai_compatible_model(self, message):
        """
        Infer using an OpenAI-compatible API endpoint.

        Args:
            message: Input message

        Returns:
            Model output or empty string on failure
        """
        model_input = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": message},
        ]

        def call_api():
            client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.offline_model_name,
                messages=model_input,
                temperature=self.temperature,
                # max_tokens=self.max_output_length
            )
            return response.choices[0].message.content

        try_count = 0
        while try_count < self.max_retries:
            try:
                try_count += 1
                output = self.run_with_timeout(call_api, timeout_seconds=100)
                if output:
                    return output
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            if try_count < self.max_retries:
                time.sleep(2 ** (try_count - 1))

        return ""


if __name__ == "__main__":
    logger = SimpleLogger()
    model = LLMLocal("qwen/qwen3-coder-30b", logger, temperature=0)
    res = model.infer("1 + 1 = ?")
    print(res)
