"""Local CORS proxy for forwarding Claude Vision API requests.

Runs on localhost, forwards only to api.anthropic.com.
Stdlib only — no dependencies beyond Python 3.9+.

Usage:
    python -m sara_brain.vision_proxy          # from package
    python vision_proxy.py                     # standalone download
"""

from __future__ import annotations

import http.server
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime

TARGET = "https://api.anthropic.com"


class CORSProxyHandler(http.server.BaseHTTPRequestHandler):
    """Forward POST requests to api.anthropic.com with CORS headers."""

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode("utf-8")
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
        """Forward request body to api.anthropic.com."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        url = f"{TARGET}{self.path}"
        headers = {"Content-Type": "application/json"}

        # Forward relevant headers
        api_key = self.headers.get("x-api-key")
        if api_key:
            headers["x-api-key"] = api_key
        anthropic_version = self.headers.get("anthropic-version")
        if anthropic_version:
            headers["anthropic-version"] = anthropic_version

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


def main(port: int = 8765):
    """Start the CORS proxy server."""
    print(f"Sara Brain Vision Proxy -- forwarding to {TARGET}")
    print(f"Listening on http://localhost:{port}")
    print()
    server = http.server.HTTPServer(("127.0.0.1", port), CORSProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Usage: python {sys.argv[0]} [port]")
            sys.exit(1)
    main(port)
