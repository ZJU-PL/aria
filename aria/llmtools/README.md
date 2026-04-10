# aria.llmtools

LLM inference with routing across online and local providers.

## Layout

- **`core/`** – Shared types and client flow:
  - `base`: `BaseProvider`, `InferenceResult` (sync infer API).
  - `responses`: `LLMResponse`, `ToolCallRequest` (raw provider response types).
  - `client`: `BaseLLMClient`, `ProviderResolution` (execution and retry).
  - `results`, `retry`, `token_counter`, `async_utils`, `logger`.

- **`providers/`** – Provider implementations and adapter base classes:
  - `base`: `LLMProvider` (async chat API), `error_response`, `parse_openai_chat_response`.
  - `adapters`: `AsyncChatProvider`, `OpenAICompatibleProvider`, `build_messages`.
  - `online/`: OpenAI, Claude, Gemini, DeepSeek.
  - `local/`: LM Studio, vLLM, SGLang.

- **`routing.py`** – Model-name and provider-hint routing; `resolve_provider()`.

- **`client.py`** – Public `LLM` class (wraps `BaseLLMClient` + routing).

## Public API

```python
from aria.llmtools import LLM, Logger, resolve_provider
```
