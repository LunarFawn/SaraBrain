"""Provider registry for sara_reader.

Supported: anthropic, ollama.

Excluded: openai. This is a deliberate ethical stance by the project
author. Requesting "openai" as a provider raises ValueError. The
exclusion is policy, not a future-work item.
"""
from __future__ import annotations

from .base import BaseProvider, ProviderResponse, ToolCall


_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider(name: str, cls: type[BaseProvider]) -> None:
    _REGISTRY[name] = cls


def get_provider(name: str, **kwargs) -> BaseProvider:
    """Return a provider instance by name.

    Raises ValueError for "openai" (excluded by author policy) or for
    any unregistered provider name.
    """
    name_lower = name.lower().strip()
    if name_lower in {"openai", "open-ai", "open_ai", "gpt", "openai-api"}:
        raise ValueError(
            "OpenAI is not a supported provider in sara_reader. "
            "This exclusion is a deliberate ethical stance by the "
            "project author and is enforced at the provider-loader level."
        )
    if name_lower not in _REGISTRY:
        registered = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown provider '{name}'. Registered providers: {registered}. "
            "OpenAI is excluded by author policy."
        )
    return _REGISTRY[name_lower](**kwargs)


# Auto-register supported providers
from . import anthropic as _anthropic_mod  # noqa: E402,F401
from . import ollama as _ollama_mod  # noqa: E402,F401


__all__ = ["BaseProvider", "ProviderResponse", "ToolCall", "get_provider"]
