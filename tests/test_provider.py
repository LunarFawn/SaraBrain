"""Tests for the LLM provider abstraction layer."""

import pytest

from sara_brain.nlp.provider import (
    AnthropicProvider,
    OllamaProvider,
    get_provider,
    DEFAULT_URLS,
)


class TestAnthropicProvider:
    def setup_method(self):
        self.provider = AnthropicProvider()

    def test_name(self):
        assert self.provider.name == "anthropic"

    def test_needs_api_key(self):
        assert self.provider.needs_api_key() is True

    def test_build_endpoint_url(self):
        url = self.provider.build_endpoint_url("https://api.anthropic.com")
        assert url == "https://api.anthropic.com/v1/messages"

    def test_build_endpoint_url_strips_trailing_slash(self):
        url = self.provider.build_endpoint_url("https://api.anthropic.com/")
        assert url == "https://api.anthropic.com/v1/messages"

    def test_build_headers_with_key(self):
        headers = self.provider.build_headers("sk-ant-test")
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["Content-Type"] == "application/json"

    def test_build_headers_none_key_omitted(self):
        headers = self.provider.build_headers("none")
        assert "x-api-key" not in headers

    def test_build_headers_null_key_omitted(self):
        headers = self.provider.build_headers(None)
        assert "x-api-key" not in headers

    def test_build_chat_payload_with_system(self):
        payload = self.provider.build_chat_payload(
            model="claude-sonnet-4-20250514",
            system="You are helpful.",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=100,
        )
        assert payload["system"] == "You are helpful."
        assert payload["model"] == "claude-sonnet-4-20250514"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 100

    def test_build_chat_payload_no_system(self):
        payload = self.provider.build_chat_payload(
            model="claude-sonnet-4-20250514",
            system=None,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=100,
        )
        assert "system" not in payload

    def test_build_image_block(self):
        block = self.provider.build_image_block("abc123", "image/png")
        assert block == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
        }

    def test_parse_text_response(self):
        body = {"content": [{"type": "text", "text": "hello world"}]}
        assert self.provider.parse_text_response(body) == "hello world"

    def test_parse_text_response_missing(self):
        assert self.provider.parse_text_response({}) is None
        assert self.provider.parse_text_response({"content": []}) is None


class TestOllamaProvider:
    def setup_method(self):
        self.provider = OllamaProvider()

    def test_name(self):
        assert self.provider.name == "ollama"

    def test_needs_api_key(self):
        assert self.provider.needs_api_key() is False

    def test_build_endpoint_url(self):
        url = self.provider.build_endpoint_url("http://localhost:11434")
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_build_endpoint_url_strips_trailing_slash(self):
        url = self.provider.build_endpoint_url("http://localhost:11434/")
        assert url == "http://localhost:11434/v1/chat/completions"

    def test_build_headers_no_auth(self):
        headers = self.provider.build_headers(None)
        assert headers == {"Content-Type": "application/json"}

    def test_build_headers_key_ignored(self):
        headers = self.provider.build_headers("some-key")
        assert "x-api-key" not in headers
        assert "Authorization" not in headers

    def test_build_chat_payload_with_system(self):
        payload = self.provider.build_chat_payload(
            model="llava",
            system="You are helpful.",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=100,
        )
        assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert payload["messages"][1] == {"role": "user", "content": "hello"}
        assert payload["model"] == "llava"
        assert "system" not in payload  # no top-level system key

    def test_build_chat_payload_no_system(self):
        payload = self.provider.build_chat_payload(
            model="llava",
            system=None,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=100,
        )
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    def test_build_image_block(self):
        block = self.provider.build_image_block("abc123", "image/jpeg")
        assert block == {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,abc123"},
        }

    def test_parse_text_response(self):
        body = {"choices": [{"message": {"content": "hello world"}}]}
        assert self.provider.parse_text_response(body) == "hello world"

    def test_parse_text_response_missing(self):
        assert self.provider.parse_text_response({}) is None
        assert self.provider.parse_text_response({"choices": []}) is None


class TestGetProvider:
    def test_get_anthropic(self):
        p = get_provider("anthropic")
        assert isinstance(p, AnthropicProvider)

    def test_get_ollama(self):
        p = get_provider("ollama")
        assert isinstance(p, OllamaProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("openai")


class TestDefaultURLs:
    def test_anthropic_url(self):
        assert DEFAULT_URLS["anthropic"] == "https://api.anthropic.com"

    def test_ollama_url(self):
        assert DEFAULT_URLS["ollama"] == "http://localhost:11434"
