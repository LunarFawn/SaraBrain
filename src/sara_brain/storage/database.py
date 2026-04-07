from __future__ import annotations

import sqlite3
from pathlib import Path


_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """SQLite connection manager with WAL mode and foreign keys."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._apply_schema()

    def _apply_schema(self) -> None:
        schema_sql = _SCHEMA_PATH.read_text()
        self.conn.executescript(schema_sql)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns to existing tables if missing (safe for fresh DBs)."""
        cols = {
            r[1]
            for r in self.conn.execute("PRAGMA table_info(paths)").fetchall()
        }
        if "account_id" not in cols:
            self.conn.execute("ALTER TABLE paths ADD COLUMN account_id INTEGER REFERENCES accounts(id)")
        if "trust_status" not in cols:
            self.conn.execute("ALTER TABLE paths ADD COLUMN trust_status TEXT")
        if "repetition_count" not in cols:
            self.conn.execute("ALTER TABLE paths ADD COLUMN repetition_count INTEGER NOT NULL DEFAULT 1")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
