from __future__ import annotations

import sqlite3


class SettingsRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM settings WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )

    def delete(self, key: str) -> None:
        self.conn.execute("DELETE FROM settings WHERE key = ?", (key,))
