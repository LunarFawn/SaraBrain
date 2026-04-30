"""Ollama provider — local models via Ollama's HTTP API.

Ollama's tool-calling support is uneven across models. Llama 3.2 1B/3B
sometimes emit tool calls as JSON text in the message content field
instead of the structured ``tool_calls`` field, and sometimes with
unquoted identifiers. The fallback parser in this module handles those
malformations so the calling code sees a consistent ToolCall shape.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from .base import BaseProvider, ProviderResponse, ToolCall


_DEFAULT_URL = "http://localhost:11434/api/chat"

# Regex for the malformed-but-recoverable pattern observed on llama3.2:
#   {"name": brain_why, "parameters": {"label": "x"}}
# (note the unquoted identifier after "name":)
_NAME_FIELD_RE = re.compile(
    r'"name"\s*:\s*"?(\w+)"?\s*,\s*"(?:parameters|arguments)"\s*:\s*',
    re.DOTALL,
)


class OllamaProvider(BaseProvider):
    """Wrap Ollama's /api/chat endpoint."""

    def __init__(self, base_url: str = _DEFAULT_URL, timeout: int = 180) -> None:
        self._url = base_url
        self._timeout = timeout

    def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: str,
        system_prompt: str = "",
    ) -> ProviderResponse:
        ollama_messages = self._translate_messages(messages, system_prompt)
        ollama_tools = self._translate_tools(tools) if tools else None

        payload: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1500},
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            return ProviderResponse(
                text=f"<<OLLAMA_ERROR: {e}>>",
                raw={"error": str(e)},
            )

        msg = body.get("message", {})
        content = (msg.get("content") or "").strip()
        tool_calls_raw = msg.get("tool_calls") or []

        # Primary path: structured tool_calls
        tool_calls: list[ToolCall] = []
        for i, tc in enumerate(tool_calls_raw):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {}) or {}
            if isinstance(args, str):
                # Some Ollama versions stringify arguments
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", f"call_{i}"),
                name=name,
                arguments=args,
            ))

        # Fallback path: small Ollama models emit tool calls as JSON-ish
        # text in the content field. Detect and parse those.
        if not tool_calls and content:
            tool_names = {t["name"] for t in tools}
            recovered = self._recover_tool_calls_from_text(content, tool_names)
            if recovered:
                tool_calls = recovered
                content = ""  # the "content" was actually a tool call

        return ProviderResponse(
            text=content,
            tool_calls=tool_calls,
            raw=body,
        )

    def _translate_messages(
        self, messages: list[dict], system_prompt: str,
    ) -> list[dict]:
        out: list[dict] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        for msg in messages:
            role = msg["role"]
            if role in {"system", "user", "assistant"}:
                m = {"role": role, "content": msg.get("content", "") or ""}
                if role == "assistant" and msg.get("tool_calls"):
                    m["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in msg["tool_calls"]
                    ]
                out.append(m)
            elif role == "tool":
                out.append({
                    "role": "tool",
                    "content": msg.get("content", ""),
                })
        return out

    def _translate_tools(self, tools: list[dict]) -> list[dict]:
        """Ollama's tool format is OpenAI-compatible function-calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def _recover_tool_calls_from_text(
        self, content: str, tool_names: set[str],
    ) -> list[ToolCall]:
        """Parse JSON-ish tool-call text that small Ollama models emit
        in place of the structured tool_calls field.

        Handles the unquoted-identifier pattern:
            {"name": brain_why, "parameters": {"label": "x"}}
        """
        if "name" not in content:
            return []
        calls: list[ToolCall] = []
        pos = 0
        idx = 0
        while pos < len(content):
            m = _NAME_FIELD_RE.search(content, pos)
            if not m:
                break
            name = m.group(1)
            if name not in tool_names:
                pos = m.end()
                continue
            brace_start = content.find("{", m.end())
            if brace_start == -1:
                break
            depth = 0
            brace_end = None
            for j in range(brace_start, len(content)):
                if content[j] == "{":
                    depth += 1
                elif content[j] == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = j + 1
                        break
            if brace_end is None:
                break
            params_raw = content[brace_start:brace_end]
            try:
                args = json.loads(params_raw)
            except json.JSONDecodeError:
                # Try a narrow repair: quote unquoted identifier values
                try:
                    patched = re.sub(
                        r':\s*([A-Za-z_]\w*)(\s*[,}])',
                        r': "\1"\2',
                        params_raw,
                    )
                    args = json.loads(patched)
                except json.JSONDecodeError:
                    pos = brace_end
                    continue
            if isinstance(args, dict):
                calls.append(ToolCall(
                    id=f"recovered_{idx}",
                    name=name,
                    arguments=args,
                ))
                idx += 1
            pos = brace_end
        return calls


# Auto-register on import
from . import register_provider  # noqa: E402
register_provider("ollama", OllamaProvider)
