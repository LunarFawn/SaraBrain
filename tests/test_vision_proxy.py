"""Tests for the local CORS proxy server."""

from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

from sara_brain.vision_proxy import CORSProxyHandler


@pytest.fixture()
def proxy_server():
    """Start proxy on an OS-assigned port in a daemon thread, yield URL, shutdown."""
    server = HTTPServer(("127.0.0.1", 0), CORSProxyHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


def _http_request(proxy_url, path, method="GET", body=None, headers=None):
    """Make an HTTP request using http.client (unaffected by urllib mocks)."""
    parsed = urlparse(proxy_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    return resp


class TestHealth:
    def test_returns_ok(self, proxy_server):
        resp = _http_request(proxy_server, "/health")
        body = json.loads(resp.read())
        assert body == {"status": "ok"}
        assert resp.status == 200

    def test_has_cors_headers(self, proxy_server):
        resp = _http_request(proxy_server, "/health")
        resp.read()
        assert resp.getheader("Access-Control-Allow-Origin") == "*"

    def test_unknown_get_returns_404(self, proxy_server):
        resp = _http_request(proxy_server, "/unknown")
        resp.read()
        assert resp.status == 404


class TestCORS:
    def test_options_returns_204(self, proxy_server):
        resp = _http_request(proxy_server, "/v1/messages", method="OPTIONS")
        resp.read()
        assert resp.status == 204

    def test_allows_required_headers(self, proxy_server):
        resp = _http_request(proxy_server, "/v1/messages", method="OPTIONS")
        resp.read()
        allowed = resp.getheader("Access-Control-Allow-Headers")
        for header in ("Content-Type", "x-api-key", "anthropic-version"):
            assert header in allowed


class TestForwarding:
    @patch("sara_brain.vision_proxy.urllib.request.urlopen")
    def test_forwards_to_anthropic(self, mock_urlopen, proxy_server):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"content": [{"text": "hello"}]}'
        mock_resp.getheader.return_value = "application/json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        payload = json.dumps({"model": "test", "messages": []}).encode()
        resp = _http_request(proxy_server, "/v1/messages", method="POST", body=payload, headers={
            "Content-Type": "application/json",
            "x-api-key": "sk-test-123",
            "anthropic-version": "2023-06-01",
        })
        body = json.loads(resp.read())
        assert body == {"content": [{"text": "hello"}]}

        # Verify the outgoing request was to Anthropic
        call_args = mock_urlopen.call_args
        outgoing_req = call_args[0][0]
        assert outgoing_req.full_url == "https://api.anthropic.com/v1/messages"
        assert outgoing_req.get_header("X-api-key") == "sk-test-123"
        assert outgoing_req.get_header("Anthropic-version") == "2023-06-01"

    @patch("sara_brain.vision_proxy.urllib.request.urlopen")
    def test_response_has_cors(self, mock_urlopen, proxy_server):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok": true}'
        mock_resp.getheader.return_value = "application/json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        payload = json.dumps({}).encode()
        resp = _http_request(proxy_server, "/v1/messages", method="POST", body=payload, headers={
            "Content-Type": "application/json",
        })
        resp.read()
        assert resp.getheader("Access-Control-Allow-Origin") == "*"

    @patch("sara_brain.vision_proxy.urllib.request.urlopen")
    def test_forwards_http_errors(self, mock_urlopen, proxy_server):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.anthropic.com/v1/messages",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=MagicMock(read=MagicMock(return_value=b'{"error": "invalid key"}')),
        )

        payload = json.dumps({}).encode()
        resp = _http_request(proxy_server, "/v1/messages", method="POST", body=payload, headers={
            "Content-Type": "application/json",
        })
        resp.read()
        assert resp.status == 401
