# pylint: disable=invalid-name
import concurrent.futures
import importlib
from typing import Tuple, Optional, Any

try:
    from openai import OpenAI  # pylint: disable=import-error
except ImportError:
    OpenAI = None  # type: ignore
try:
    from zhipuai import ZhipuAI  # pylint: disable=import-error
except ImportError:
    ZhipuAI = None  # type: ignore
try:
    import tiktoken  # pylint: disable=import-error
except ImportError:
    tiktoken = None  # type: ignore

from aria.ml.llm.llmtool.logger import Logger


def _optional_import(module_name: str, from_name: str = None) -> Optional[Any]:
    """Safely import optional dependencies"""
    try:
        if from_name:
            module = importlib.import_module(module_name)
            return getattr(module, from_name)
        return importlib.import_module(module_name)
    except ImportError:
        return None

# Import optional dependencies
# FIXME: should we do this? Just installing the dependencies should be easy...
genai = _optional_import("google.generativeai")
anthropic = _optional_import("anthropic", "Anthropic")


class LLM:
    """Multi-provider LLM inference: Gemini, OpenAI, DeepSeek, Claude, GLM"""

    def __init__(self, online_model_name: str, logger: Logger,
                 temperature: float = 0.0,
                 system_role="You are an experienced programmer.") -> None:
        self.online_model_name = online_model_name
        if tiktoken is None:
            raise ImportError("tiktoken is required for LLM")
        self.encoding = tiktoken.encoding_for_model(
            "gpt-3.5-turbo-0125"
        )  # We only use gpt-3.5 to measure token cost
        self.temperature = temperature
        self.systemRole = system_role
        self.logger = logger


    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        self.logger.print_log(self.online_model_name, "is running")
        output = ""
        if "gemini" in self.online_model_name:
            output = self.infer_with_gemini(message)
        elif "gpt" in self.online_model_name:
            output = self.infer_with_openai_model(message)
        elif "o3-mini" in self.online_model_name:
            output = self.infer_with_o3_mini_model(message)
        elif "glm" in self.online_model_name or "zhipu" in self.online_model_name:
            # Zhipu/GLM models (e.g., glm-4, glm-4-flash)
            output = self.infer_with_glm_model(message)
        elif "claude" in self.online_model_name:
            output = self.infer_with_claude(message)
        elif "deepseek" in self.online_model_name:
            output = self.infer_with_deepseek_model(message)
        else:
            raise ValueError("Unsupported model name")

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

    def infer_with_gemini(self, message: str) -> str:
        """Infer using Gemini model"""
        if not genai:
            self.logger.print_log("Gemini SDK not installed")
            return ""

        model = genai.GenerativeModel("gemini-pro")
        def call_api():
            return model.generate_content(
                f"{self.systemRole}\n{message}",
                safety_settings=[{"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"}],
                generation_config=genai.types.GenerationConfig(temperature=self.temperature)
            ).text

        return self._retry_api_call(call_api, timeout=50)

    def infer_with_openai_model(self, message):
        """Infer using OpenAI model"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").split(":")[0])
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}],
                temperature=self.temperature
            ).choices[0].message.content

        return self._retry_api_call(call_api)

    def infer_with_o3_mini_model(self, message):
        """Infer using o3-mini model"""
        if not OpenAI:
            self.logger.print_log("OpenAI SDK not installed")
            return ""

        def call_api():
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "").split(":")[0])
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}]
            ).choices[0].message.content

        return self._retry_api_call(call_api)

    def infer_with_deepseek_model(self, message):
        """Infer using DeepSeek model"""
        def call_api():
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY2"),
                          base_url="https://api.deepseek.com")
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}],
                temperature=self.temperature
            ).choices[0].message.content

        return self._retry_api_call(call_api, timeout=300)

    def infer_with_claude(self, message):
        """Infer using Claude model via direct Anthropic API"""
        if not anthropic:
            self.logger.print_log("Anthropic SDK not installed")
            return ""

        # Map model names to Anthropic API model identifiers
        model_mapping = {
            "claude-3.5": "claude-3-5-sonnet-20241022",
            "claude-3-5": "claude-3-5-sonnet-20241022",
            "claude-3": "claude-3-sonnet-20240229",
            "claude": "claude-3-5-sonnet-20241022"  # default to latest
        }

        # Find matching model or use default
        model_id = "claude-3-5-sonnet-20241022"  # default
        for key, value in model_mapping.items():
            if key in self.online_model_name.lower():
                model_id = value
                break

        def call_api():
            client = anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model=model_id,
                max_tokens=4000,
                temperature=self.temperature,
                system=self.systemRole,
                messages=[{"role": "user", "content": message}]
            )
            return response.content[0].text

        return self._retry_api_call(call_api)


    def infer_with_glm_model(self, message):
        """Infer using GLM model"""
        if not ZhipuAI:
            self.logger.print_log("ZhipuAI SDK not installed")
            return ""

        def call_api():
            client = ZhipuAI(api_key=os.environ.get("GLM_API_KEY") or os.environ.get("ZHIPU_API_KEY"))
            return client.chat.completions.create(
                model=self.online_model_name,
                messages=[{"role": "system", "content": self.systemRole},
                         {"role": "user", "content": message}],
                temperature=self.temperature
            ).choices[0].message.content

        return self._retry_api_call(call_api)


if __name__ == "__main__":
    llm = LLM(online_model_name="glm-4-plus", logger=Logger("tmp.txt"))
    print(llm.infer("Hello, how are you?"))
