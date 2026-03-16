"""LLM utilities for online inference using various LLM providers."""

import concurrent.futures
import os
import time
from typing import Tuple

import tiktoken
from openai import OpenAI

from aria.efmc.llmtools.logger import Logger


class LLM:
    """
    An online inference model using different LLMs:
    - DeepSeek: V3, R1 (uses OpenAI-compatible API)
    - GLM: Zhipu AI models
    """

    def __init__(
        self,
        online_model_name: str,
        logger: Logger,
        temperature: float = 0.0,
        system_role: str = (
            "You are a experienced programmer and good at understanding "
            "programs written in mainstream programming languages."
        ),
    ) -> None:
        """
        Initialize LLM instance.

        Args:
            online_model_name: Name of the online model
            logger: Logger instance
            temperature: Temperature for inference
            system_role: System role message
        """
        self.online_model_name = online_model_name
        self.encoding = tiktoken.encoding_for_model(
            "gpt-3.5-turbo-0125"
        )  # We only use gpt-3.5 to measure token cost
        self.temperature = temperature
        self.system_role = system_role
        self.logger = logger

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        """
        Infer using the online model.

        Args:
            message: Input message
            is_measure_cost: Whether to measure token cost

        Returns:
            Tuple of (output, input_token_cost, output_token_cost)
        """
        self.logger.print_log(self.online_model_name, "is running")
        output = ""
        if "deepseek" in self.online_model_name:
            output = self.infer_with_deepseek_model(message)
        elif "glm" in self.online_model_name:
            output = self.infer_with_glm_model(message)
        else:
            raise ValueError(
                "Unsupported model name. Only DeepSeek and GLM models are supported."
            )

        input_token_cost = (
            0
            if not is_measure_cost
            else len(self.encoding.encode(self.system_role))
            + len(self.encoding.encode(message))
        )
        output_token_cost = (
            0 if not is_measure_cost else len(self.encoding.encode(output))
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

    def infer_with_deepseek_model(self, message):
        """
        Infer using the DeepSeek model (V3, R1, etc.)
        DeepSeek uses OpenAI-compatible API format

        Args:
            message: Input message

        Returns:
            Model output or empty string on failure
        """
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            self.logger.print_log("DeepSeek API key not found in environment variables")
            return ""

        model_input = [
            {
                "role": "system",
                "content": self.system_role,
            },
            {"role": "user", "content": message},
        ]

        def call_api():
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
            response = client.chat.completions.create(
                model=self.online_model_name,
                messages=model_input,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        try_count = 0
        while try_count < 5:
            try_count += 1
            try:
                output = self.run_with_timeout(call_api, timeout_seconds=300)
                if output:
                    return output
            except (ValueError, RuntimeError, ConnectionError) as e:
                self.logger.print_log(f"DeepSeek API error: {e}")
            time.sleep(2)

        return ""

    def infer_with_glm_model(self, message):
        """
        Infer using the GLM model.

        Args:
            message: Input message

        Returns:
            Model output or empty string on failure
        """
        try:
            from zhipuai import ZhipuAI  # pylint: disable=import-outside-toplevel
        except ImportError:
            self.logger.print_log(
                "zhipuai package not installed. Please install it to use GLM models."
            )
            return ""

        api_key = os.environ.get("GLM_API_KEY")
        if not api_key:
            self.logger.print_log("GLM API key not found in environment variables")
            return ""

        model_input = [
            {"role": "system", "content": self.system_role},
            {"role": "user", "content": message},
        ]

        def call_api():
            client = ZhipuAI(api_key=api_key)
            response = client.chat.completions.create(
                model=self.online_model_name,
                messages=model_input,
                temperature=self.temperature,
            )
            return response.choices[0].message.content

        try_count = 0
        while try_count < 5:
            try_count += 1
            try:
                output = self.run_with_timeout(call_api, timeout_seconds=100)
                if output:
                    return output
            except (ValueError, RuntimeError, ConnectionError) as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)

        return ""
