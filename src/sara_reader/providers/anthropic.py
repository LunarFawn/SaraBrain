"""Anthropic provider — Claude API with tool use."""
from __future__ import annotations

import os
from typing import Any

from .base import BaseProvider, ProviderResponse, ToolCall


class AnthropicProvider(BaseProvider):
    """Wrap the official anthropic SDK.

    Requires ``ANTHROPIC_API_KEY`` in the environment, or pass
    ``api_key=...`` to the constructor.
    """

    def __init__(self, api_key: str | None = None) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "anthropic SDK is required for the Anthropic provider. "
                "Install with: pip install anthropic"
            ) from e
        from anthropic import Anthropic
        self._client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        system_prompt: str = "",
    ) -> ProviderResponse:
        # Translate generic messages → Anthropic shape
        anth_messages = self._translate_messages(messages)
        anth_tools = self._translate_tools(tools)

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": anth_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anth_tools:
            kwargs["tools"] = anth_tools

        response = self._client.messages.create(**kwargs)

        # Parse the response: it has a list of content blocks, each
        # either text or tool_use.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(block.input),
                ))
        return ProviderResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=response,
        )

    def _translate_messages(self, messages: list[dict]) -> list[dict]:
        """Convert generic messages to Anthropic format.

        Generic shape (OpenAI-style):
            {"role": "user"|"assistant"|"tool",
             "content": str,
             "tool_calls"?: [...],
             "tool_call_id"?: str}

        Anthropic shape:
            user/assistant: {"role", "content": [blocks]}
            assistant with tool calls: content includes tool_use blocks
            tool results: {"role": "user", "content": [{"type":
                "tool_result", "tool_use_id": ..., "content": ...}]}
        """
        out: list[dict] = []
        for msg in messages:
            role = msg["role"]
            if role in {"user", "assistant"}:
                content_blocks: list[dict] = []
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": msg["content"]})
                if role == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": tc["arguments"],
                        })
                if not content_blocks:
                    content_blocks.append({"type": "text", "text": ""})
                out.append({"role": role, "content": content_blocks})
            elif role == "tool":
                # Tool results land as a user message with a tool_result
                # content block in Anthropic's protocol.
                out.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg["tool_call_id"],
                        "content": msg["content"],
                    }],
                })
            else:
                # Unknown role; skip
                continue
        return out

    def _translate_tools(self, tools: list[dict]) -> list[dict]:
        """Convert generic tool schemas to Anthropic tool format."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]


# Auto-register on import
from . import register_provider  # noqa: E402
register_provider("anthropic", AnthropicProvider)
