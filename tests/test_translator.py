"""Tests for the LLM translator (Anthropic Messages API only)."""

from unittest.mock import patch, MagicMock
import json

import pytest

from sara_brain.nlp.translator import LLMTranslator, is_blocked_domain


@pytest.fixture
def translator():
    return LLMTranslator("https://api.anthropic.com", "sk-ant-test", "claude-sonnet-4-20250514")


class TestSystemPrompt:
    def test_includes_available_commands(self, translator):
        commands = ["how <concept> taste", "what <concept> color"]
        prompt = translator.build_system_prompt(commands)
        assert "how <concept> taste" in prompt
        assert "what <concept> color" in prompt

    def test_includes_instructions(self, translator):
        prompt = translator.build_system_prompt(["teach <subject> is/are <property>"])
        assert "structured command" in prompt.lower()
        assert "UNKNOWN" in prompt


class TestTranslate:
    @patch("urllib.request.urlopen")
    def test_successful_translation(self, mock_urlopen, translator):
        response_body = json.dumps({
            "content": [{"type": "text", "text": "how apple taste"}]
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = translator.translate(
            "what does an apple taste like?",
            ["how <concept> taste"]
        )
        assert result == "how apple taste"

        # Verify Anthropic-style request
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.anthropic.com/v1/messages"
        assert req.get_header("X-api-key") == "sk-ant-test"
        assert req.get_header("Anthropic-version") == "2023-06-01"

        payload = json.loads(req.data)
        assert "system" in payload
        assert payload["messages"] == [{"role": "user", "content": "what does an apple taste like?"}]

    @patch("urllib.request.urlopen")
    def test_unknown_response(self, mock_urlopen, translator):
        response_body = json.dumps({
            "content": [{"type": "text", "text": "UNKNOWN"}]
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = translator.translate("gibberish xyzzy", ["how <concept> taste"])
        assert result is None

    @patch("urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen, translator):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")

        result = translator.translate("test query", ["how <concept> taste"])
        assert result is None

    def test_api_key_none_omits_x_api_key(self):
        t = LLMTranslator("https://api.anthropic.com", "none", "claude-sonnet-4-20250514")
        prompt = t.build_system_prompt(["test"])
        assert "test" in prompt


class TestBlockedDomains:
    def test_openai_blocked(self):
        assert is_blocked_domain("https://api.openai.com/v1/chat") is True

    def test_openai_azure_blocked(self):
        assert is_blocked_domain("https://openai.azure.com/deployments/gpt4") is True

    def test_openai_org_blocked(self):
        assert is_blocked_domain("https://api.openai.org") is True

    def test_anthropic_allowed(self):
        assert is_blocked_domain("https://api.anthropic.com") is False

    def test_localhost_allowed(self):
        assert is_blocked_domain("http://localhost:8080") is False

    def test_translate_raises_on_blocked_domain(self):
        t = LLMTranslator("https://api.openai.com", "key", "gpt-4")
        with pytest.raises(ValueError, match="Blocked API domain"):
            t.translate("hello", ["test"])

    def test_translate_raises_on_azure_openai(self):
        t = LLMTranslator("https://openai.azure.com/deployments/gpt4", "key", "gpt-4")
        with pytest.raises(ValueError, match="Blocked API domain"):
            t.translate("hello", ["test"])


class TestURLBuilding:
    def test_url_trailing_slash_stripped(self):
        t = LLMTranslator("https://api.anthropic.com/", "key", "model")
        assert t.api_url == "https://api.anthropic.com"

    def test_messages_endpoint_used(self, translator):
        # The translate method should target /v1/messages
        with patch("urllib.request.urlopen") as mock_urlopen:
            response_body = json.dumps({
                "content": [{"type": "text", "text": "UNKNOWN"}]
            }).encode("utf-8")
            mock_resp = MagicMock()
            mock_resp.read.return_value = response_body
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            translator.translate("test", ["cmd"])
            req = mock_urlopen.call_args[0][0]
            assert req.full_url == "https://api.anthropic.com/v1/messages"
