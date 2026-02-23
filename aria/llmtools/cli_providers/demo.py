"""Demo script for calling LLM via OpenCode Zen provider.

This demo shows how to use the OpenCode Zen free LLM service
to make chat completion requests without requiring an API key.
"""

import asyncio

from aria.llmtools.cli_providers.opencode import (
    chat_opencode,
    OPENCODE_FREE_MODELS,
    is_opencode_model,
    strip_opencode_prefix,
)


async def demo_basic_chat() -> None:
    """Demonstrate basic chat completion with OpenCode Zen."""
    print("=" * 60)
    print("Demo: Basic Chat Completion")
    print("=" * 60)

    # Use one of the free models
    model = "big-pickle"
    print(f"\nUsing model: {model}")
    print(f"Description: {OPENCODE_FREE_MODELS.get(model, 'Unknown')}")

    # Simple message
    messages = [
        {"role": "user", "content": "Hello! Please introduce yourself briefly."}
    ]

    print("\nSending request...")
    response = await chat_opencode(
        model=model,
        messages=messages,
        max_tokens=2560,
        temperature=0.7,
    )

    print("\n" + "-" * 40)
    print("Response:")
    print("-" * 40)
    print(f"Content: {response.content}")
    print(f"Finish Reason: {response.finish_reason}")
    if response.usage:
        print(
            f"Tokens - Prompt: {response.usage.get('prompt_tokens')}, "
            f"Completion: {response.usage.get('completion_tokens')}, "
            f"Total: {response.usage.get('total_tokens')}"
        )


async def demo_multi_turn_conversation() -> None:
    """Demonstrate multi-turn conversation with OpenCode Zen.

    This demo shows a TRUE multi-turn conversation:
    1. Send first message and get actual response
    2. Append response to message history
    3. Send follow-up question based on the response
    """
    print("\n" + "=" * 60)
    print("Demo: Multi-turn Conversation (Interactive)")
    print("=" * 60)

    model = "glm-5-free"
    print(f"\nUsing model: {model}")
    print(f"Description: {OPENCODE_FREE_MODELS.get(model, 'Unknown')}")

    # Initialize message history
    messages = []

    # ===== First turn =====
    print("\n" + "-" * 40)
    print("Turn 1: User asks about capital")
    print("-" * 40)

    messages.append({"role": "user", "content": "What is the capital of France?"})
    print(f"User: What is the capital of France?")

    print("\nSending request...")
    response1 = await chat_opencode(
        model=model,
        messages=messages,
        max_tokens=2560,
        temperature=0.7,
    )

    print(f"Assistant: {response1.content}")

    # Append assistant's response to message history
    messages.append({"role": "assistant", "content": response1.content})

    # ===== Second turn =====
    print("\n" + "-" * 40)
    print("Turn 2: User asks follow-up question")
    print("-" * 40)

    # User asks a follow-up question based on the previous response
    messages.append({"role": "user", "content": "What is its population?"})
    print("User: What is its population?")

    print("\nSending request with full conversation history...")
    response2 = await chat_opencode(
        model=model,
        messages=messages,
        max_tokens=2560,
        temperature=0.7,
    )

    print(f"Assistant: {response2.content}")

    # Show the complete conversation history
    print("\n" + "=" * 40)
    print("Complete Conversation History:")
    print("=" * 40)
    for i, msg in enumerate(messages):
        print(f"{i+1}. [{msg['role']}]: {msg['content'][:100]}...")
    print(f"{len(messages)+1}. [assistant]: {response2.content[:100]}...")


async def main() -> None:
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# OpenCode Zen LLM Demo")
    print("#" * 60)

    # Run demos
    await demo_basic_chat()
    await demo_multi_turn_conversation()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
