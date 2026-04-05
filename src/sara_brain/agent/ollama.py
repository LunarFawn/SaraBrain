"""Ollama HTTP client — health check, model listing, chat with tool calling.

Uses only stdlib (urllib). Separate from nlp/provider.py because this serves
the agent orchestration role (chat + tools), not the sensory perception role.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from ..nlp.provider import DEFAULT_URLS

DEFAULT_BASE_URL = DEFAULT_URLS.get("ollama", "http://localhost:11434")


def check_health(base_url: str = DEFAULT_BASE_URL) -> bool:
    """Check if Ollama is running."""
    try:
        req = urllib.request.Request(base_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def list_models(base_url: str = DEFAULT_BASE_URL) -> list[dict]:
    """List available Ollama models. Returns list of model info dicts."""
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
            return body.get("models", [])
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return []


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """Send a chat completion request with optional tool calling.

    Returns the raw response dict. Raises on HTTP or parse errors.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"

    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())

    return body


def extract_response(body: dict) -> dict:
    """Extract the assistant message from a chat response.

    Returns a dict with keys: 'content' (str|None), 'tool_calls' (list|None).

    Handles two cases:
    1. Model supports structured tool calling → tool_calls field is populated
    2. Model outputs tool calls as text → parse JSON from content as fallback
    """
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content")
    tool_calls = message.get("tool_calls")

    # If structured tool calls exist, use them directly
    if tool_calls:
        return {"content": content, "tool_calls": tool_calls}

    # Fallback: try to parse tool calls from text content
    # Small models often output {"name": "tool", "arguments": {...}} as text
    if content:
        parsed = _try_parse_text_tool_calls(content)
        if parsed:
            return {"content": None, "tool_calls": parsed}

    return {"content": content, "tool_calls": None}


def _try_parse_text_tool_calls(text: str) -> list[dict] | None:
    """Attempt to extract tool calls from text output.

    Small models that don't support structured tool calling will often
    output JSON blocks like: {"name": "read_file", "arguments": {"path": "..."}}
    This function finds and parses those.
    """
    import re

    # Known tool names to look for
    tool_names = {
        "brain_query", "brain_recognize", "brain_context", "brain_summarize",
        "brain_teach", "brain_observe", "brain_validate", "brain_stats",
        "read_file", "write_file", "list_directory", "search_files",
        "search_content", "execute_python", "shell_command",
    }

    # Look for JSON objects containing "name" and "arguments"
    # Match patterns like {"name": "tool_name", "arguments": {...}}
    json_pattern = re.compile(r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}')

    found = []
    for match in json_pattern.finditer(text):
        name = match.group(1)
        if name in tool_names:
            try:
                args = json.loads(match.group(2))
                found.append({
                    "id": f"text_parsed_{len(found)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                })
            except json.JSONDecodeError:
                continue

    # Also try: just a bare tool name and arguments on separate lines
    # e.g., the model says: I'll call read_file with path "/some/path"
    if not found:
        for tool_name in tool_names:
            if tool_name in text:
                # Try to find a JSON block after mentioning the tool
                after_mention = text[text.index(tool_name):]
                brace_match = re.search(r'\{[^{}]+\}', after_mention)
                if brace_match:
                    try:
                        args = json.loads(brace_match.group())
                        found.append({
                            "id": f"text_parsed_{len(found)}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args),
                            },
                        })
                        break  # Take the first match
                    except json.JSONDecodeError:
                        continue

    return found if found else None
