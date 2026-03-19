"""Tests for the Claude Vision API client."""

from unittest.mock import patch, MagicMock
import json
import tempfile
import os

import pytest

from sara_brain.nlp.vision import VisionObserver


@pytest.fixture
def observer():
    return VisionObserver("https://api.anthropic.com", "sk-ant-test", "claude-sonnet-4-20250514")


@pytest.fixture
def test_image(tmp_path):
    """Create a minimal valid PNG file for testing."""
    # Minimal 1x1 white PNG
    import base64
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mP8/58BAwAI/AL+hc2rNAAAAABJRU5ErkJggg=="
    )
    png_data = base64.b64decode(png_b64)
    img_path = tmp_path / "test.png"
    img_path.write_bytes(png_data)
    return str(img_path)


@pytest.fixture
def test_jpeg(tmp_path):
    """Create a minimal test JPEG file."""
    # Minimal JPEG (not valid image data, but enough for base64 encoding)
    data = b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"\xff\xd9"
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(data)
    return str(img_path)


def _mock_api_response(text):
    """Create a mock urllib response returning the given text."""
    response_body = json.dumps({
        "content": [{"type": "text", "text": text}]
    }).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestLoadImage:
    def test_loads_png(self, observer, test_image):
        b64, media_type = observer._load_image(test_image)
        assert media_type == "image/png"
        assert len(b64) > 0

    def test_loads_jpeg(self, observer, test_jpeg):
        b64, media_type = observer._load_image(test_jpeg)
        assert media_type == "image/jpeg"

    def test_rejects_unsupported_format(self, observer, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("not an image")
        with pytest.raises(ValueError, match="Unsupported image format"):
            observer._load_image(str(txt))


class TestSanitize:
    def test_simple_labels(self):
        raw = "red\nround\nsmooth\nshiny"
        result = VisionObserver._sanitize(raw)
        assert result == ["red", "round", "smooth", "shiny"]

    def test_strips_bullets(self):
        raw = "- red\n- round\n* smooth"
        result = VisionObserver._sanitize(raw)
        assert result == ["red", "round", "smooth"]

    def test_strips_colons(self):
        raw = "color: red\nshape: round"
        result = VisionObserver._sanitize(raw)
        assert result == ["red", "round"]

    def test_handles_commas(self):
        raw = "red, round, smooth"
        result = VisionObserver._sanitize(raw)
        assert result == ["red", "round", "smooth"]

    def test_rejects_urls(self):
        raw = "red\nhttps://evil.com\nround"
        result = VisionObserver._sanitize(raw)
        assert "https" not in str(result)
        assert "red" in result
        assert "round" in result

    def test_rejects_code(self):
        raw = "red\nimport os\ndef hack():\nround"
        result = VisionObserver._sanitize(raw)
        assert all("import" not in r and "def" not in r for r in result)
        assert "red" in result
        assert "round" in result

    def test_rejects_special_characters(self):
        raw = "red\n<script>alert('xss')</script>\nround"
        result = VisionObserver._sanitize(raw)
        assert all("<" not in r and ">" not in r for r in result)

    def test_deduplicates(self):
        raw = "red\nred\nround"
        result = VisionObserver._sanitize(raw)
        assert result == ["red", "round"]

    def test_max_length(self):
        raw = "a" * 50  # Too long
        result = VisionObserver._sanitize(raw)
        assert len(result) == 0

    def test_lowercase_enforcement(self):
        raw = "RED\nRound\nSMOOTH"
        result = VisionObserver._sanitize(raw)
        assert result == ["red", "round", "smooth"]


class TestObserveInitial:
    @patch("urllib.request.urlopen")
    def test_returns_sanitized_labels(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response("red\nround\nsmooth\nshiny\nsmall")
        result = observer.observe_initial(test_image)
        assert result == ["red", "round", "smooth", "shiny", "small"]

    @patch("urllib.request.urlopen")
    def test_network_error_returns_empty(self, mock_urlopen, observer, test_image):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        result = observer.observe_initial(test_image)
        assert result == []

    @patch("urllib.request.urlopen")
    def test_api_payload_structure(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response("red")
        observer.observe_initial(test_image)

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.anthropic.com/v1/messages"
        assert req.get_header("X-api-key") == "sk-ant-test"

        payload = json.loads(req.data)
        msg = payload["messages"][0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][0]["source"]["type"] == "base64"
        assert msg["content"][0]["source"]["media_type"] == "image/png"
        assert msg["content"][1]["type"] == "text"


class TestObserveDirected:
    @patch("urllib.request.urlopen")
    def test_parses_directed_response(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response(
            "taste: sweet\ntexture: smooth\ntemperature: cannot determine"
        )
        questions = {
            "taste": "What does this taste like?",
            "texture": "What texture does this have?",
            "temperature": "What temperature is this?",
        }
        result = observer.observe_directed(test_image, questions)
        assert result["taste"] == "sweet"
        assert result["texture"] == "smooth"
        assert result["temperature"] is None

    @patch("urllib.request.urlopen")
    def test_missing_answers_return_none(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response("taste: sweet")
        questions = {"taste": "taste?", "size": "size?"}
        result = observer.observe_directed(test_image, questions)
        assert result["taste"] == "sweet"
        assert result["size"] is None

    def test_empty_questions(self, observer, test_image):
        result = observer.observe_directed(test_image, {})
        assert result == {}


class TestVerifyProperty:
    @patch("urllib.request.urlopen")
    def test_yes(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response("YES")
        assert observer.verify_property(test_image, "crunchy") is True

    @patch("urllib.request.urlopen")
    def test_no(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response("NO")
        assert observer.verify_property(test_image, "crunchy") is False

    @patch("urllib.request.urlopen")
    def test_indeterminate(self, mock_urlopen, observer, test_image):
        mock_urlopen.return_value = _mock_api_response("CANNOT DETERMINE")
        assert observer.verify_property(test_image, "crunchy") is None

    @patch("urllib.request.urlopen")
    def test_network_error(self, mock_urlopen, observer, test_image):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("fail")
        assert observer.verify_property(test_image, "crunchy") is None


class TestBlockedDomain:
    def test_blocked_domain_raises(self, test_image):
        obs = VisionObserver("https://api.openai.com", "key", "model")
        with pytest.raises(ValueError, match="Blocked API domain"):
            obs.observe_initial(test_image)
