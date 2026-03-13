"""
Using Kilo or OpenCode CLI to call free LLMs as the LLM provider.

Note: this file is different from kilocode.py and opencode.py. Those files do not require installing kilocode or opencode locally.

Requires one of:
  - kilo: npm install -g @kilocode/cli
  - opencode: npm install -g @opencode/cli (or similar)

Free models:
  - kilo: kilo/z-ai/glm-4.5-air:free, kilo/qwen/qwen3-4b:free, etc. (kilo models)
  - opencode: opencode/kimi-k2.5-free, etc.
"""

import concurrent.futures
import subprocess
from typing import Any, Callable, Literal, Optional, Tuple

from aria.llmtools.core.base import BaseProvider, InferenceResult
from aria.llmtools.providers.shared import error_result, result_from_usage

_tiktoken: Any = None
try:
    import tiktoken as _tiktoken
except ImportError:
    pass

tiktoken = _tiktoken

CLIProvider = Literal["kilo", "opencode"]


def _infer_provider_from_model(model: str) -> CLIProvider:
    """Infer CLI provider from model prefix."""
    if model.startswith("opencode/"):
        return "opencode"
    if model.startswith("kilo/"):
        return "kilo"
    return "kilo"  # default


def _strip_cli_header(text: str) -> str:
    """Strip CLI header line (e.g. '> code · model' or '> sisyphus · model') from output."""
    lines = text.strip().split("\n")
    while lines and (lines[0].startswith("> ") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def cli_infer(
    prompt: str,
    model: str = "kilo/z-ai/glm-4.5-air:free",
    provider: Optional[CLIProvider] = None,
    timeout_seconds: int = 120,
    auto_approve: bool = True,
    strip_header: bool = True,
) -> str:
    """
    Call a free LLM via Kilo or OpenCode CLI and return the response.

    Args:
        prompt: User message to send to the model.
        model: Model ID with provider prefix (e.g. kilo/z-ai/glm-4.5-air:free,
               opencode/kimi-k2.5-free).
        provider: "kilo" or "opencode". If None, inferred from model prefix.
        timeout_seconds: Max wait time for the subprocess.
        auto_approve: Use --auto for kilo (opencode has no --auto).
        strip_header: Remove CLI header line from output.

    Returns:
        Model response text, or empty string on failure.
    """
    prov = provider or _infer_provider_from_model(model)
    if prov == "kilo" and not model.startswith("kilo/"):
        model = f"kilo/{model}"
    elif prov == "opencode" and not model.startswith("opencode/"):
        model = f"opencode/{model}"

    if prov == "kilo":
        cmd = ["kilo", "-m", model, "run", prompt]
        if auto_approve:
            cmd.insert(-1, "--auto")
        cli_name = "kilo"
        install_hint = "npm install -g @kilocode/cli"
    else:
        cmd = ["opencode", "-m", model, "run", prompt]
        cli_name = "opencode"
        install_hint = "npm install -g opencode (or @opencode/cli)"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=None,
        )
        out = (result.stdout or "").strip()
        if result.returncode != 0 and not out:
            err = (result.stderr or "").strip()
            return "" if not err else f"[{cli_name} error {result.returncode}]: {err}"
        if strip_header:
            out = _strip_cli_header(out)
        return out
    except subprocess.TimeoutExpired:
        return ""
    except FileNotFoundError:
        return f"[{cli_name} not found: install with '{install_hint}']"
    except Exception as e:
        return f"[{cli_name} error]: {e}"


# Backward compatibility
def kilo_infer(
    prompt: str,
    model: str = "kilo/z-ai/glm-4.5-air:free",
    timeout_seconds: int = 120,
    auto_approve: bool = True,
    strip_header: bool = True,
) -> str:
    """Call a free LLM via Kilo CLI. See cli_infer for full API."""
    return cli_infer(
        prompt,
        model=model,
        provider="kilo",
        timeout_seconds=timeout_seconds,
        auto_approve=auto_approve,
        strip_header=strip_header,
    )


class LLMCli:
    """
    LLM backend that calls free models via Kilo or OpenCode CLI.

    Compatible with the infer(message) -> (output, input_cost, output_cost)
    interface used by LLMLocal and LLM.
    """

    def __init__(
        self,
        model_name: str = "kilo/z-ai/glm-4.5-air:free",
        logger: Optional[object] = None,
        temperature: float = 0.0,
        system_role: str = (
            "You are an experienced programmer and good at understanding "
            "programs written in mainstream programming languages."
        ),
        max_output_length: int = 4096,
        measure_cost: bool = False,
        timeout_seconds: int = 120,
        provider: Optional[CLIProvider] = None,
    ) -> None:
        """
        Initialize CLI-based LLM.

        Args:
            model_name: Model ID with prefix (e.g. kilo/z-ai/glm-4.5-air:free,
                        opencode/kimi-k2.5-free).
            logger: Logger with print_log(*args). Can be None.
            temperature: Ignored; kept for API compatibility.
            system_role: Prefixed to user message; kept for compatibility.
            max_output_length: Unused; kept for compatibility.
            measure_cost: Use tiktoken for token cost estimation.
            timeout_seconds: Subprocess timeout per call.
            provider: "kilo" or "opencode". If None, inferred from model prefix.
        """
        self.model_name = model_name
        self.provider = provider or _infer_provider_from_model(model_name)
        self.logger = logger
        self.system_role = system_role
        self.max_output_length = max_output_length
        self.measure_cost = measure_cost
        self.timeout_seconds = timeout_seconds
        self.encoding = None
        if measure_cost and tiktoken:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")

    def _log(self, *args: Any) -> None:
        if self.logger and hasattr(self.logger, "print_log"):
            self.logger.print_log(*args)

    def infer(
        self, message: str, is_measure_cost: bool = False
    ) -> Tuple[str, int, int]:
        """
        Infer using the CLI free model.

        Args:
            message: Input message (or full prompt).
            is_measure_cost: Whether to estimate token cost.

        Returns:
            Tuple of (output, input_token_cost, output_token_cost).
        """
        self._log(self.model_name, "is running")
        full_prompt = (
            f"{self.system_role}\n\n{message}" if self.system_role else message
        )
        output = cli_infer(
            full_prompt,
            model=self.model_name,
            provider=self.provider,
            timeout_seconds=self.timeout_seconds,
            auto_approve=True,
            strip_header=True,
        )
        input_token_cost = 0
        output_token_cost = 0
        if (is_measure_cost or self.measure_cost) and tiktoken:
            enc = self.encoding or tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
            input_token_cost = len(enc.encode(self.system_role)) + len(
                enc.encode(message)
            )
            output_token_cost = len(enc.encode(output)) if output else 0
        return output, input_token_cost, output_token_cost

    def run_with_timeout(
        self,
        func: Callable[[], str],
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """
        Run a function with timeout (for compatibility with LLMLocal interface).

        Args:
            func: Callable with no args, returns str.
            timeout_seconds: Override instance timeout.

        Returns:
            Function result or empty string on timeout/error.
        """
        timeout = (
            timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)
            try:
                result = future.result(timeout=timeout)
                return str(result)
            except concurrent.futures.TimeoutError:
                self._log("Operation timed out")
                return ""
            except (ValueError, RuntimeError, ConnectionError, OSError) as e:
                self._log("Operation failed:", e)
                return ""


# Backward compatibility
LLMKiloCode = LLMCli


class LocalCLIProvider(BaseProvider):
    """`BaseProvider` adapter for local CLI-backed models."""

    def __init__(self, model_name: str, logger: Any, temperature: float):
        self.cli_provider = LLMCli(
            model_name=model_name,
            logger=logger,
            temperature=temperature,
            system_role="",
            measure_cost=False,
        )

    def infer(
        self,
        message: str,
        system_role: str,
        temperature: float,
        max_output_length: int,
    ) -> InferenceResult:
        """Run inference via the local CLI wrapper."""
        del system_role, temperature, max_output_length
        output, _, _ = self.cli_provider.infer(message)

        if not output:
            return error_result("Empty CLI response")

        return result_from_usage(content=output, finish_reason="stop")


if __name__ == "__main__":
    import sys

    from aria.llmtools.logger import Logger

    logger = Logger("llm_cli.log")
    # Default: kilo. Use opencode/kimi-k2.5-free for opencode.
    model = sys.argv[1] if len(sys.argv) > 1 else "kilo/z-ai/glm-5:free"
    llm = LLMCli(
        model_name=model,
        logger=logger,
        temperature=0,
        measure_cost=True,
    )
    out, inp_cost, out_cost = llm.infer("Tell a story.")
    print("Output:", repr(out))
    print("Input tokens:", inp_cost, "Output tokens:", out_cost)
