"""Optional LLM translation layer for natural language queries."""

from __future__ import annotations

import json
import urllib.request
import urllib.error


class LLMTranslator:
    """Translates natural language to structured Sara Brain commands.

    Uses any OpenAI-compatible chat completions endpoint.
    """

    def __init__(self, api_url: str, api_key: str, model: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def translate(self, user_input: str, available_commands: list[str]) -> str | None:
        """Send to LLM, get back a structured sara command. Returns None on failure."""
        commands_list = "\n".join(f"  - {cmd}" for cmd in available_commands)
        system_prompt = (
            "You are a translator for Sara Brain, a cognitive simulation.\n"
            "Your job is to convert natural language questions into structured commands.\n"
            "Available commands:\n"
            f"{commands_list}\n\n"
            "Respond with ONLY the structured command, nothing else.\n"
            "If you cannot translate the input, respond with: UNKNOWN"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            "temperature": 0,
            "max_tokens": 100,
        }

        url = f"{self.api_url}/chat/completions"
        if "/v1/chat/completions" not in url and "/chat/completions" not in self.api_url:
            url = f"{self.api_url}/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key.lower() != "none":
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
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
