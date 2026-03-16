# AGENTS.md - LLM Tools

`aria.llmtools` is the provider/routing layer for LLM inference across online
and local backends.

Read `aria/llmtools/README.md` first. For provider-specific work, also inspect
`aria/llmtools/providers/README.md`.

## Architecture

- `client.py`: public entrypoint
- `core/`: shared types, retry logic, token counting, logging, async helpers
- `providers/`: provider implementations and adapters
- `routing.py`: model/provider resolution

## Working Rules

- Preserve the separation between provider-agnostic flow and provider-specific
  adapters.
- Be explicit about sync vs async boundaries.
- Avoid introducing network-dependent behavior into unit tests unless the task
  explicitly requires integration coverage.
- Prefer mocking provider responses over hitting real APIs.
- When changing provider parsing, check both structured response handling and
  error translation paths.

## Testing

Run the smallest relevant test slice. If no direct tests exist, validate imports
and the touched code path locally, and say what remains untested.
