"""Optional LLM translation layer for natural language queries.

Supports multiple LLM providers via the provider abstraction.
OpenAI endpoints are explicitly blocked when using providers
that require an API key.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

from .provider import LLMProvider, AnthropicProvider

_BLOCKED_DOMAINS = frozenset({
    "api.openai.com",
    "openai.azure.com",
    "api.openai.org",
})

_DEFAULT_API_URL = "https://api.anthropic.com"


def is_blocked_domain(url: str) -> bool:
    """Return True if *url* points to a blocked (OpenAI) domain."""
    try:
        hostname = urlparse(url).hostname or ""
    except Exception:
        return False
    return any(hostname == d or hostname.endswith("." + d) for d in _BLOCKED_DOMAINS)


class LLMTranslator:
    """Translates natural language to structured Sara Brain commands.

    Works with any LLMProvider (defaults to AnthropicProvider for backward compat).
    """

    def __init__(self, api_url: str, api_key: str, model: str,
                 provider: LLMProvider | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider or AnthropicProvider()

    def translate(self, user_input: str, available_commands: list[str]) -> str | None:
        """Send to LLM, get back a structured sara command. Returns None on failure.

        Raises ValueError if the configured URL points to a blocked domain
        (only enforced for providers that require an API key).
        """
        if self.provider.needs_api_key() and is_blocked_domain(self.api_url):
            raise ValueError(
                f"Blocked API domain: {self.api_url}. "
                "Only Anthropic (Claude) endpoints are allowed."
            )

        system_prompt = self.build_system_prompt(available_commands)

        messages = [{"role": "user", "content": user_input}]
        payload = self.provider.build_chat_payload(
            model=self.model,
            system=system_prompt,
            messages=messages,
            temperature=0,
            max_tokens=100,
        )

        url = self.provider.build_endpoint_url(self.api_url)
        headers = self.provider.build_headers(self.api_key)

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = self.provider.parse_text_response(body)
                if content is None:
                    return None
                content = content.strip()
                if content.upper() == "UNKNOWN":
                    return None
                return content
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
            return None

    def build_system_prompt(self, available_commands: list[str]) -> str:
        """Build the system prompt (exposed for testing)."""
        commands_list = "\n".join(f"  - {cmd}" for cmd in available_commands)
        return (
            "You are a translator for Sara Brain, a cognitive simulation.\n"
            "Your job is to convert natural language questions into structured commands.\n"
            "Available commands:\n"
            f"{commands_list}\n\n"
            "Respond with ONLY the structured command, nothing else.\n"
            "If you cannot translate the input, respond with: UNKNOWN"
        )
