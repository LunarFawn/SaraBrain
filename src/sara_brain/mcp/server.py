"""Sara Brain MCP server — JSON-RPC 2.0 over stdio.

Pure stdlib implementation. No external dependencies.
Any MCP client (Claude, Amazon Q, VS Code, etc.) can connect.

Usage:
    sara-mcp                          # default DB at ~/.sara_brain/sara.db
    sara-mcp --db /path/to/sara.db    # custom DB path

Configuration for Claude Code (.claude/settings.json):
    {
        "mcpServers": {
            "sara-brain": {
                "command": "sara-mcp",
                "args": []
            }
        }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from ..config import default_db_path
from ..core.brain import Brain
from .tools import TOOLS, ToolHandler

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "sara-brain"
SERVER_VERSION = "0.1.0"


def _send(msg: dict) -> None:
    """Write a JSON-RPC message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _error_response(req_id: int | str | None, code: int, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _result_response(req_id: int | str | None, result: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


class MCPServer:
    """JSON-RPC 2.0 MCP server over stdio."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.handler = ToolHandler(brain)
        self.initialized = False

    def handle_message(self, msg: dict) -> dict | None:
        """Process a JSON-RPC message. Returns response or None for notifications."""
        method = msg.get("method", "")
        req_id = msg.get("id")
        params = msg.get("params", {})

        # Notifications (no id) — no response needed
        if req_id is None:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        # Requests (have id) — must respond
        if method == "initialize":
            return self._handle_initialize(req_id, params)
        if method == "tools/list":
            return self._handle_tools_list(req_id)
        if method == "tools/call":
            return self._handle_tools_call(req_id, params)
        if method == "ping":
            return _result_response(req_id, {})

        return _error_response(req_id, -32601, f"Method not found: {method}")

    def _handle_initialize(self, req_id: int | str, params: dict) -> dict:
        return _result_response(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        })

    def _handle_tools_list(self, req_id: int | str) -> dict:
        return _result_response(req_id, {"tools": TOOLS})

    def _handle_tools_call(self, req_id: int | str, params: dict) -> dict:
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            result_text = self.handler.handle(name, arguments)
        except Exception as e:
            result_text = f"Error: {e}"

        return _result_response(req_id, {
            "content": [
                {"type": "text", "text": result_text}
            ],
        })

    def run(self) -> None:
        """Main loop: read stdin, dispatch, write stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                _send(_error_response(None, -32700, "Parse error"))
                continue

            try:
                response = self.handle_message(msg)
                if response is not None:
                    _send(response)
            except Exception:
                req_id = msg.get("id")
                _send(_error_response(req_id, -32603, traceback.format_exc()))


def main() -> None:
    """Entry point for sara-mcp command."""
    parser = argparse.ArgumentParser(
        prog="sara-mcp",
        description="Sara Brain MCP server — exposes brain tools for any LLM client",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Brain database path (default: {default_db_path()})",
    )
    args = parser.parse_args()

    db_path = args.db or default_db_path()
    brain = Brain(db_path)

    try:
        server = MCPServer(brain)
        server.run()
    finally:
        brain.close()


if __name__ == "__main__":
    main()
