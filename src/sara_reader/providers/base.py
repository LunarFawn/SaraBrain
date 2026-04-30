"""Base provider interface for sara_reader.

Each provider implements ``chat(messages, tools, model) -> ProviderResponse``.
The reader loop sends messages, receives a normalized response, executes
any tool calls, and feeds tool results back as messages until the model
produces a final text response.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A tool invocation request from the model.

    Attributes:
        id: provider-specific identifier the model assigned to this call.
        name: tool name (e.g., "brain_explore").
        arguments: parsed JSON arguments dict.
    """
    id: str
    name: str
    arguments: dict


@dataclass
class ProviderResponse:
    """Normalized response from a provider chat call.

    Either ``text`` is non-empty (final answer) OR ``tool_calls`` is
    non-empty (model wants to invoke tools). They may both be present
    if the model produced both text and tool calls in one turn; the
    reader loop should run the tool calls first and then loop.

    Attributes:
        text: final text content from the model, if any.
        tool_calls: tool invocation requests.
        raw: the provider's raw response object, for debugging.
    """
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class BaseProvider:
    """Provider interface. Subclasses implement chat for a specific API."""

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        system_prompt: str = "",
    ) -> ProviderResponse:
        """Send a chat request with tools available.

        Args:
            messages: prior conversation in OpenAI-style shape:
                [{"role": "user"|"assistant"|"tool", "content": str,
                  optional "tool_calls", optional "tool_call_id"}, ...].
                Provider implementations translate to their native shape.
            tools: tool schemas in the provider-agnostic shape from
                tools.get_tool_schemas(): [{"name", "description",
                "parameters"}, ...].
            model: model identifier specific to the provider.
            system_prompt: optional system instruction prepended.

        Returns:
            ProviderResponse with text and/or tool_calls.
        """
        raise NotImplementedError
