# Provider Organization
Provider modules are split by deployment target:

- `online/`: hosted providers reached directly through Python SDKs or HTTP APIs, such as OpenAI-compatible endpoints, Gemini, Claude, and DeepSeek.
- `local/`: self-hosted or workstation-local providers, such as LM Studio, vLLM, SGLang, and the local Codex-compatible endpoint.

Cross-cutting provider helpers live alongside those implementations:

- `base.py`: async chat-provider protocol (`LLMProvider`) and OpenAI-response parsing helpers. Raw response types (`LLMResponse`, `ToolCallRequest`) live in `core.responses`.
- `adapters.py`: inference-oriented adapter base classes (`AsyncChatProvider`, `OpenAICompatibleProvider`, `build_messages`) used by the routed `LLM` client.
