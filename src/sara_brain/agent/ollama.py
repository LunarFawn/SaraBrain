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
    """
    choice = body.get("choices", [{}])[0]
    message = choice.get("message", {})
    return {
        "content": message.get("content"),
        "tool_calls": message.get("tool_calls"),
    }
