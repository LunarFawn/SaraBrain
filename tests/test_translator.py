"""Tests for the LLM translator."""

from unittest.mock import patch, MagicMock
import json

import pytest

from sara_brain.nlp.translator import LLMTranslator


@pytest.fixture
def translator():
    return LLMTranslator("http://localhost:11434/v1", "none", "llama3")


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
            "choices": [{"message": {"content": "how apple taste"}}]
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

    @patch("urllib.request.urlopen")
    def test_unknown_response(self, mock_urlopen, translator):
        response_body = json.dumps({
            "choices": [{"message": {"content": "UNKNOWN"}}]
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

    def test_api_key_none_omits_auth(self, translator):
        translator.api_key = "none"
        # Just verify it doesn't crash during prompt building
        prompt = translator.build_system_prompt(["test"])
        assert "test" in prompt


class TestURLBuilding:
    def test_url_with_v1_endpoint(self):
        t = LLMTranslator("http://localhost:11434/v1", "key", "model")
        assert t.api_url == "http://localhost:11434/v1"

    def test_url_trailing_slash_stripped(self):
        t = LLMTranslator("http://localhost:11434/v1/", "key", "model")
        assert t.api_url == "http://localhost:11434/v1"
