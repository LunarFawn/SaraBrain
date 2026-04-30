# sara_reader

Provider-agnostic SDK for using Sara Brain in production applications. The LLM is the orchestrator — `sara_reader` exposes Sara's retrieval tools to the model via the provider's native function-calling API and runs the tool-call loop until the model produces a final answer.

## Supported providers

- **anthropic** — Claude via the official `anthropic` SDK
- **ollama** — local models via the Ollama HTTP API (Llama 3.2, Mistral, etc.)

## NOT supported: OpenAI

OpenAI is **deliberately excluded** as a provider in this SDK. This is an ethical stance by the project author, not an oversight or a future-work item. The provider loader will raise `ValueError` on any request for `"openai"`.

This is policy. Forks adding OpenAI support would not be supported by this project.

## Installation

This package depends on `sara-brain` being installable in the same environment. From the workspace root:

```bash
pip install -e ../sara_brain
pip install -e .
```

## Usage — Python

```python
from sara_reader import SaraReader

reader = SaraReader(
    brain_path="/path/to/aptamer_full.db",
    provider="anthropic",
    model="claude-haiku-4-5",
)

answer = reader.ask("what is the molecular snare?")
```

The brain path can be a local file, a network-mounted file, a symlink — anything the OS resolves as a filesystem path.

## Usage — CLI

```bash
sara-ask "what is the molecular snare?" \
    --brain /path/to/aptamer_full.db \
    --provider anthropic \
    --model claude-haiku-4-5
```

Add `--trace` to see the full retrieval trace alongside the answer.

## How it works

The LLM is given retrieval tools (`brain_explore`, `brain_why`, `brain_trace`, `brain_recognize`, `brain_did_you_mean`) and a system prompt that directs it to call `brain_explore` for any factual question about the substrate. The reader loop:

1. Sends the user's question to the model with the retrieval tools available
2. If the model requests a tool call, runs it against the loaded brain
3. Feeds the tool result back to the model
4. Loops until the model produces a final text answer

The model retains agency over which tools to call. The system prompt nudges it toward `brain_explore` first, but the model decides. If you need stricter retrieval enforcement (e.g., for measurement protocols), use the deterministic harness in `sara_brain/papers/instrument_validation/` instead.

## Write tools are not exposed

`brain_teach_triple`, `brain_refute`, and `brain_ingest` are intentionally NOT in the tool surface. `sara_reader` is read-only by default. Apps that need to teach should use the lower-level `Brain.teach_triple` API directly with explicit user authorization.

## Provider environment

- `anthropic` — requires `ANTHROPIC_API_KEY` in the environment, or pass `provider_kwargs={"api_key": "..."}` to the `SaraReader` constructor.
- `ollama` — requires Ollama running locally on the default port (`http://localhost:11434`). Pass `provider_kwargs={"base_url": "..."}` to point elsewhere.
