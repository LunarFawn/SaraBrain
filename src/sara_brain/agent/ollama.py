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

        # Strip malformed tool call fragments from text
        # e.g., {"name": brain_context} with no quotes/args
        content = _strip_malformed_tool_calls(content)
        if not content.strip():
            content = None

    return {"content": content, "tool_calls": None}


def _strip_malformed_tool_calls(text: str) -> str:
    """Remove broken tool call JSON fragments from text output."""
    import re
    # Match {"name": tool_name} or {"name": "tool_name"} without arguments
    cleaned = re.sub(
        r'\{\s*"name"\s*:\s*"?\w+"?\s*\}',
        "",
        text,
    )
    return cleaned.strip()


def _try_parse_text_tool_calls(text: str) -> list[dict] | None:
    """Attempt to extract tool calls from text output.

    Small models that don't support structured tool calling will often
    output JSON blocks like: {"name": "read_file", "arguments": {"path": "..."}}
    This function finds and parses those, including parameter-less tool
    calls like {"name": "brain_scan_pollution", "arguments": {}}.

    The list of known tool names is loaded dynamically from the tools
    module so that new tools are automatically recognized.
    """
    import re

    # Load known tool names dynamically from the tools registry
    try:
        from .tools import get_tool_definitions
        tool_names = {
            t["function"]["name"] for t in get_tool_definitions()
        }
    except Exception:
        # Fallback list if the import fails
        tool_names = {
            "brain_query", "brain_recognize", "brain_context", "brain_summarize",
            "brain_teach", "brain_refute", "brain_did_you_mean", "brain_ingest",
            "brain_import", "brain_stats",
            "brain_scan_pollution", "brain_cleanup_articles",
            "brain_cleanup_pronouns", "brain_list_suspected_typos",
            "read_file", "write_file", "list_directory", "search_files",
            "search_content", "execute_python", "shell_command",
            "voice_listen", "voice_transcribe",
        }

    found: list[dict] = []

    # Strategy 1: balanced-brace JSON object scan.
    # Find every {...} block in the text (handling nested braces) and try
    # to parse each one. If it has a "name" field matching a known tool,
    # that's a tool call.
    for obj_text in _find_balanced_braces(text):
        try:
            data = json.loads(obj_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get("name")
        if not isinstance(name, str) or name not in tool_names:
            continue
        args = data.get("arguments", {})
        if isinstance(args, str):
            # Sometimes models nest a JSON string here
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        found.append({
            "id": f"text_parsed_{len(found)}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        })

    # Strategy 2: bare tool name mentioned in narrative text
    # e.g., "I'll call read_file with path /tmp/foo"
    if not found:
        for tool_name in tool_names:
            if tool_name in text:
                after_mention = text[text.index(tool_name):]
                brace_match = re.search(r'\{[^{}]*\}', after_mention)
                if brace_match:
                    try:
                        args = json.loads(brace_match.group())
                        if isinstance(args, dict):
                            found.append({
                                "id": f"text_parsed_{len(found)}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(args),
                                },
                            })
                            break
                    except json.JSONDecodeError:
                        continue

    return found if found else None


def _find_balanced_braces(text: str) -> list[str]:
    """Yield substrings of `text` that are balanced JSON objects.

    Walks the text tracking brace depth so we can extract objects that
    contain nested objects (e.g., {"name": "x", "arguments": {"y": 1}}).
    Strings inside the objects are respected so that braces inside
    string literals don't break balance counting.
    """
    results = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        # Found a {. Walk forward, tracking depth and string state.
        start = i
        depth = 0
        in_string = False
        escape = False
        while i < n:
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\" and in_string:
                escape = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        results.append(text[start:i + 1])
                        i += 1
                        break
            i += 1
        else:
            # Reached end without closing — discard partial
            break
    return results
