"""LLM provider abstraction layer.

Supports Anthropic (Claude), Ollama (Llama/local LLMs), and Q (Amazon Q IDE agent).
The LLM serves as Sara's sensory interface — not the brain itself.
Zero external dependencies (stdlib only)."""

from __future__ import annotations


class LLMProvider:
    """Base class for LLM API providers."""

    name: str = ""

    def build_endpoint_url(self, base_url: str) -> str:
        raise NotImplementedError

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        raise NotImplementedError

    def build_chat_payload(
        self,
        model: str,
        system: str | None,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        raise NotImplementedError

    def build_image_block(self, b64_data: str, media_type: str) -> dict:
        raise NotImplementedError

    def parse_text_response(self, body: dict) -> str | None:
        raise NotImplementedError

    def needs_api_key(self) -> bool:
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API (Claude)."""

    name = "anthropic"

    def build_endpoint_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/v1/messages"

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key and api_key.lower() != "none":
            headers["x-api-key"] = api_key
        return headers

    def build_chat_payload(
        self,
        model: str,
        system: str | None,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            payload["system"] = system
        return payload

    def build_image_block(self, b64_data: str, media_type: str) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        }

    def parse_text_response(self, body: dict) -> str | None:
        try:
            return body["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return None

    def needs_api_key(self) -> bool:
        return True


class OllamaProvider(LLMProvider):
    """Ollama local LLM (OpenAI-compatible API)."""

    name = "ollama"

    def build_endpoint_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/v1/chat/completions"

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def build_chat_payload(
        self,
        model: str,
        system: str | None,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        return {
            "model": model,
            "messages": all_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def build_image_block(self, b64_data: str, media_type: str) -> dict:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64_data}"},
        }

    def parse_text_response(self, body: dict) -> str | None:
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None

    def needs_api_key(self) -> bool:
        return False


class QProvider(LLMProvider):
    """Amazon Q IDE agent — acts as cortex directly, no HTTP.

    Q is already present in the IDE. It calls Sara's brain directly
    through Python execution rather than HTTP API calls.
    No API key, no URL, no proxy needed.
    """

    name = "q"

    def build_endpoint_url(self, base_url: str) -> str:
        return ""  # Q doesn't use HTTP

    def build_headers(self, api_key: str | None) -> dict[str, str]:
        return {}  # Q doesn't use HTTP

    def build_chat_payload(
        self,
        model: str,
        system: str | None,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        return {"system": system, "messages": messages}

    def build_image_block(self, b64_data: str, media_type: str) -> dict:
        return {}  # Q uses filesystem, not base64

    def parse_text_response(self, body: dict) -> str | None:
        return body.get("text")

    def needs_api_key(self) -> bool:
        return False


DEFAULT_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "ollama": "http://localhost:11434",
    "llama": "http://localhost:11434",
    "q": "",
}

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "llama": OllamaProvider,
    "q": QProvider,
}


def get_provider(name: str) -> LLMProvider:
    """Factory: 'anthropic' | 'ollama' | 'llama' | 'q' -> provider instance."""
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name!r}. Choose from: {', '.join(_PROVIDERS)}")
    return cls()
