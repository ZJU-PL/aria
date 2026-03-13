"""Demo script for calling the local Codex-compatible endpoint."""

import asyncio
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse

from aria.llmtools.providers.base import LLMResponse
from aria.llmtools.providers.local.codex import (
    DEFAULT_CODEX_BASE_URL,
    OpenAICodexProvider,
)


REQUEST_TIMEOUT_SECONDS = 30.0
DEMO_MODEL = "gpt-5.2"
RetryCallback = Callable[[int, int, float], Awaitable[None]]


async def _ensure_demo_ready(provider: OpenAICodexProvider) -> bool:
    """Report missing local prerequisites before issuing requests."""
    if not os.environ.get("CODEX_API_KEY"):
        print("Missing CODEX_API_KEY. Export it before running this demo.")
        return False

    base_url = provider._get_base_url()  # pylint: disable=protected-access
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=2.0,
        )
        writer.close()
        await writer.wait_closed()
        del reader
    except Exception as exc:
        print("Codex endpoint is unreachable at {0}: {1}".format(base_url, exc))
        return False

    return True


def _print_header(title: str, prefix: str = "\n") -> None:
    """Print a section header with consistent formatting."""
    print(prefix + "=" * 60)
    print(title)
    print("=" * 60)


def _print_timeout() -> None:
    """Print the shared timeout error message."""
    print(
        "Timed out after {0:.0f}s waiting for the Codex endpoint.".format(
            REQUEST_TIMEOUT_SECONDS
        )
    )


def _print_usage(response: LLMResponse) -> None:
    """Print token usage when it is present."""
    if not response.usage:
        return

    print(
        "Tokens - Prompt: {0}, Completion: {1}, Total: {2}".format(
            response.usage.get("prompt_tokens", 0),
            response.usage.get("completion_tokens", 0),
            response.usage.get("total_tokens", 0),
        )
    )


def _build_provider() -> OpenAICodexProvider:
    """Create the provider used by all demos."""
    return OpenAICodexProvider(default_model=DEMO_MODEL)


async def _run_chat(
    provider: OpenAICodexProvider,
    messages: List[Dict[str, Any]],
    on_retry: Optional[RetryCallback] = None,
) -> LLMResponse:
    """Run a provider chat request under the demo timeout."""
    return await asyncio.wait_for(
        provider.chat(
            messages=messages,
            model=provider.get_default_model(),
            on_retry=on_retry,
        ),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _print_provider_info(provider: OpenAICodexProvider) -> None:
    """Print model and base URL for the current demo run."""
    print(f"\nUsing model: {provider.get_default_model()}")
    print(f"Base URL: {provider._get_base_url()}")  # pylint: disable=protected-access


async def demo_basic_chat() -> None:
    """Demonstrate a basic Codex chat completion."""
    _print_header("Demo: Basic Chat Completion", prefix="")
    provider = _build_provider()

    _print_provider_info(provider)
    if not await _ensure_demo_ready(provider):
        return

    print("\nSending request...")
    try:
        response = await _run_chat(
            provider=provider,
            messages=[{"role": "user", "content": "How are you"}],
        )
    except asyncio.TimeoutError:
        _print_timeout()
        return

    print("\n" + "-" * 40)
    print("Response:")
    print("-" * 40)
    print(f"Content: {response.content}")
    print(f"Finish Reason: {response.finish_reason}")
    _print_usage(response)


async def demo_multi_turn_conversation() -> None:
    """Demonstrate a multi-turn Codex conversation."""
    _print_header("Demo: Multi-turn Conversation (Interactive)")
    provider = _build_provider()

    _print_provider_info(provider)
    if not await _ensure_demo_ready(provider):
        return

    messages: List[Dict[str, str]] = []
    prompts = [
        ("Turn 1: User asks about capital", "What is the capital of France?"),
        ("Turn 2: User asks follow-up question", "What is its population?"),
    ]

    for title, prompt in prompts:
        print("\n" + "-" * 40)
        print(title)
        print("-" * 40)
        messages.append({"role": "user", "content": prompt})
        print(f"User: {prompt}")

        request_label = "Sending request..."
        if len(messages) > 1:
            request_label = "\nSending request with full conversation history..."
        print(request_label)

        try:
            response = await _run_chat(provider=provider, messages=messages)
        except asyncio.TimeoutError:
            _print_timeout()
            return

        assistant_reply = response.content or ""
        print(f"Assistant: {assistant_reply}")
        messages.append({"role": "assistant", "content": assistant_reply})

    print("\n" + "=" * 40)
    print("Complete Conversation History:")
    print("=" * 40)
    for i, msg in enumerate(messages):
        print(f"{i + 1}. [{msg['role']}]: {msg['content'][:100]}...")
    print("Final finish reason: {0}".format(response.finish_reason))


async def demo_retry_logging() -> None:
    """Demonstrate the provider retry callback used around Codex calls."""
    _print_header("Demo: Retry Logging")

    provider = _build_provider()
    _print_provider_info(provider)
    if not await _ensure_demo_ready(provider):
        return

    async def on_retry(attempt: int, max_retries: int, delay: float) -> None:
        print(
            "Retrying request ({0}/{1}) in {2:.1f}s".format(
                attempt, max_retries, delay
            )
        )

    print("\nSending request...")
    try:
        response = await _run_chat(
            provider=provider,
            messages=[{"role": "user", "content": "Reply in one short sentence."}],
            on_retry=on_retry,
        )
    except asyncio.TimeoutError:
        _print_timeout()
        return

    print("Response: {0}".format(response.content))
    print("Finish Reason: {0}".format(response.finish_reason))
    _print_usage(response)


async def main() -> None:
    """Run all demos against the local Codex-compatible endpoint."""
    print("\n" + "#" * 60)
    print("# Codex LLM Demo")
    print("#" * 60)
    print("Endpoint: {0}".format(DEFAULT_CODEX_BASE_URL))

    await demo_basic_chat()
    await demo_multi_turn_conversation()
    await demo_retry_logging()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
