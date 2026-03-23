"""Local CORS proxy for forwarding Vision API requests.

Supports Anthropic (Claude) and Ollama (local LLMs).
Runs on localhost. Stdlib only — no dependencies beyond Python 3.9+.

Usage:
    python vision_proxy.py                          # Anthropic (default)
    python vision_proxy.py --provider ollama        # Ollama at localhost:11434
    python vision_proxy.py --provider ollama --ollama-url http://localhost:11434
    python vision_proxy.py --port 8765              # custom port
"""

from __future__ import annotations

import argparse
import http.server
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime

ANTHROPIC_TARGET = "https://api.anthropic.com"
OLLAMA_DEFAULT_URL = "http://localhost:11434"

# Set by main() before server starts
_provider: str = "anthropic"
_ollama_url: str = OLLAMA_DEFAULT_URL


class CORSProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward POST requests to the configured provider with CORS headers."""

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/health":
            body = json.dumps({"status": "ok", "provider": _provider}).encode("utf-8")
            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self._add_cors_headers()
            self.end_headers()

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(204)
        self._add_cors_headers()
        self.end_headers()

    def do_POST(self):
        """Forward request body to the configured provider."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        if _provider == "ollama":
            self._forward_ollama(body)
        else:
            self._forward_anthropic(body)

    def _forward_anthropic(self, body: bytes) -> None:
        """Forward to api.anthropic.com, preserving API key and version headers."""
        url = f"{ANTHROPIC_TARGET}{self.path}"
        headers = {"Content-Type": "application/json"}

        api_key = self.headers.get("x-api-key")
        if api_key:
            headers["x-api-key"] = api_key
        anthropic_version = self.headers.get("anthropic-version")
        if anthropic_version:
            headers["anthropic-version"] = anthropic_version

        self._do_forward(url, headers, body)

    def _forward_ollama(self, body: bytes) -> None:
        """Forward to Ollama (OpenAI-compatible endpoint), no auth headers."""
        url = f"{_ollama_url.rstrip('/')}{self.path}"
        headers = {"Content-Type": "application/json"}
        self._do_forward(url, headers, body)

    def _do_forward(self, url: str, headers: dict, body: bytes) -> None:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self._add_cors_headers()
                self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                self._log_request(resp.status, len(resp_body))
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            self.send_response(e.code)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
            self._log_request(e.code, len(resp_body))
        except urllib.error.URLError as e:
            err = json.dumps({"error": str(e.reason)}).encode("utf-8")
            self.send_response(502)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            self._log_request(502, len(err))

    def _add_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, x-api-key, anthropic-version")

    def _log_request(self, status, byte_count):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {self.command} {self.path} -> {status} ({byte_count:,} bytes)")

    def log_message(self, format, *args):
        """Suppress default request logging — we use _log_request instead."""
        pass


def main():
    global _provider, _ollama_url

    parser = argparse.ArgumentParser(description="Sara Brain Vision Proxy")
    parser.add_argument("--provider", choices=["anthropic", "ollama"], default="anthropic",
                        help="LLM provider (default: anthropic)")
    parser.add_argument("--ollama-url", default=OLLAMA_DEFAULT_URL,
                        help=f"Ollama base URL (default: {OLLAMA_DEFAULT_URL})")
    parser.add_argument("--port", type=int, default=8765,
                        help="Proxy port (default: 8765)")
    args = parser.parse_args()

    _provider = args.provider
    _ollama_url = args.ollama_url.rstrip("/")

    if _provider == "anthropic":
        target_display = ANTHROPIC_TARGET
    else:
        target_display = _ollama_url

    print(f"Sara Brain Vision Proxy -- provider: {_provider}, forwarding to {target_display}")
    print(f"Listening on http://localhost:{args.port}")
    print()

    server = http.server.HTTPServer(("127.0.0.1", args.port), CORSProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
