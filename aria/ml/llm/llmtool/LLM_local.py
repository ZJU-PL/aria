# pylint: disable=invalid-name
"""
Calling local LLMs
  - vLLM
  - sglang
  - LMStudio
"""
import concurrent.futures
import importlib
from typing import Tuple, Optional, Any

try:
    from openai import OpenAI  # pylint: disable=import-error
except ImportError:
    OpenAI = None  # type: ignore
try:
    import tiktoken  # pylint: disable=import-error
except ImportError:
    tiktoken = None  # type: ignore

from aria.ml.llm.llmtool.logger import Logger


class LLMLocal:
    """Local LLM inference: vLLM, sglang, LM Studio"""

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
        provider: str = "lm-studio",  # vllm, sglang, lm-studio, etc.
    ) -> None:
        self.offline_model_name = offline_model_name
        self.temperature = temperature
        self.systemRole = system_role  # pylint: disable=invalid-name
        self.logger = logger
        self.max_output_length = max_output_length
        self.provider = provider
        if tiktoken is None:
            raise ImportError("tiktoken is required for LLMLocal")
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        self.logger.print_log(self.offline_model_name, "is running")
        output = ""

        # Route to appropriate provider based on self.provider
        if self.provider == "lm-studio":
            output = self.infer_with_lm_studio(message)
        elif self.provider == "vllm":
            output = self.infer_with_vllm(message)
        elif self.provider == "sglang":
            output = self.infer_with_sglang(message)
        else:
            # Fallback: try to infer from model name
            if "qwen" in self.offline_model_name.lower():
                output = self.infer_with_lm_studio(message)  # Default to LM Studio
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        input_token_cost = (
            0
            if not is_measure_cost
            else len(self.encoding.encode(self.systemRole))
            + len(self.encoding.encode(message))
        )
        output_token_cost = (
            0 if not is_measure_cost else len(self.encoding.encode(output))
        )
        return output, input_token_cost, output_token_cost

    def _retry_api_call(self, call_func, timeout=100, max_retries=5):
        """Common retry logic for all API calls"""
        for attempt in range(max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(call_func)
                    result = future.result(timeout=timeout)
                    if result:
                        return result
            except concurrent.futures.TimeoutError:
                self.logger.print_log("Operation timed out")
            except Exception as e:
                self.logger.print_log(f"API error: {e}")
            time.sleep(2)
        return ""

    def infer_with_lm_studio(self, message):
        """Infer using LM Studio local server"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
            return (
                client.chat.completions.create(
                    model=self.offline_model_name,
                    messages=[
                        {"role": "system", "content": self.systemRole},
                        {"role": "user", "content": message},
                    ],
                    temperature=self.temperature,
                )
                .choices[0]
                .message.content
            )

        return self._retry_api_call(call_api)

    def infer_with_vllm(self, message):
        """Infer using vLLM server"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            # vLLM typically runs on port 8000 with OpenAI-compatible API
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
            return (
                client.chat.completions.create(
                    model=self.offline_model_name,
                    messages=[
                        {"role": "system", "content": self.systemRole},
                        {"role": "user", "content": message},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_length,
                )
                .choices[0]
                .message.content
            )

        return self._retry_api_call(call_api)

    def infer_with_sglang(self, message):
        """Infer using SGLang server"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            # SGLang typically runs on port 30000 with OpenAI-compatible API
            client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
            return (
                client.chat.completions.create(
                    model=self.offline_model_name,
                    messages=[
                        {"role": "system", "content": self.systemRole},
                        {"role": "user", "content": message},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_length,
                )
                .choices[0]
                .message.content
            )

        return self._retry_api_call(call_api)


if __name__ == "__main__":
    logger = Logger("/tmp/llm_local_test.log")
    model = LLMLocal(
        "qwen/qwen3-coder-30b", logger, temperature=0, provider="lm-studio"
    )
    res = model.infer("tell a story")
    print(res)
