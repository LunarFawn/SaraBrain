"""Session persistence — conversation history to disk.

Stores messages as JSON files in ~/.sara_brain/sessions/.
Simple append-only format for conversation replay and resume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class SessionStore:
    """Persist and retrieve conversation sessions."""

    def __init__(self, session_dir: str | None = None) -> None:
        if session_dir is None:
            self.session_dir = Path.home() / ".sara_brain" / "sessions"
        else:
            self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, messages: list[dict]) -> None:
        """Write conversation messages to disk."""
        path = self.session_dir / f"{session_id}.json"
        path.write_text(json.dumps(messages, indent=2), encoding="utf-8")

    def load(self, session_id: str) -> list[dict] | None:
        """Load messages from a previous session. Returns None if not found."""
        path = self.session_dir / f"{session_id}.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def list_sessions(self) -> list[str]:
        """List available session IDs, newest first."""
        sessions = []
        for p in self.session_dir.glob("*.json"):
            sessions.append((p.stem, p.stat().st_mtime))
        sessions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in sessions]

    def new_session_id(self) -> str:
        """Generate a new session ID: YYYY-MM-DD_NNN."""
        date_str = time.strftime("%Y-%m-%d")
        existing = [
            s for s in self.list_sessions() if s.startswith(date_str)
        ]
        n = len(existing) + 1
        return f"{date_str}_{n:03d}"
