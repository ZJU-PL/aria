# Provider Subdirectories

- `online/`: hosted providers reached directly through Python SDKs or HTTP APIs, such as OpenAI-compatible endpoints, Gemini, Claude, and DeepSeek.
- `local/`: self-hosted providers exposed as local OpenAI-compatible HTTP servers, such as LM Studio, vLLM, and SGLang.
- `cli/`: providers that depend on an installed CLI tool or custom transport layer rather than a simple direct SDK wrapper; this includes Kilo, OpenCode, Codex, shared CLI response parsing, and the local CLI adapter.

This split is based on transport style:

- `online` means remote API integration.
- `local` means local server integration.
- `cli` means subprocess, OAuth, or custom async transport integration.
