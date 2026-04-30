"""sara_reader — provider-agnostic SDK for using Sara Brain in production apps.

The LLM is the orchestrator. sara_reader exposes Sara's retrieval tools to
the model via the provider's native function-calling API and runs the
tool-call loop until the model produces a final answer. The substrate is
loaded from a file path (local or network-mounted).

Supported providers:
    - anthropic  (Claude API, via the official anthropic SDK)
    - ollama     (local models, via the Ollama HTTP API)

Not supported:
    - openai     (excluded as a deliberate ethical stance by the project
                  author; the loader will refuse this provider name)

Public API:

    >>> from sara_reader import SaraReader
    >>> reader = SaraReader(
    ...     brain_path="/path/to/aptamer_full.db",
    ...     provider="anthropic",
    ...     model="claude-haiku-4-5",
    ... )
    >>> reader.ask("what is the molecular snare?")
"""
from .reader import SaraReader
from .stateless_reader import StatelessReader

__all__ = ["SaraReader", "StatelessReader"]
__version__ = "0.2.0"
