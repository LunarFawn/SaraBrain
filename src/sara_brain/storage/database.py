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
        # Force checkpoint after every auto-commit so data lands in the
        # main .db file immediately. Prevents data loss if the process
        # crashes before the default 1000-page auto-checkpoint fires.
        # Sara never forgets — the WAL must not be a memory hole.
        self.conn.execute("PRAGMA wal_autocheckpoint=1")
        self._apply_schema()

    def _apply_schema(self) -> None:
        # Migrate existing tables BEFORE running full schema,
        # because schema may include indexes on new columns.
        self._migrate()
        schema_sql = _SCHEMA_PATH.read_text()
        self.conn.executescript(schema_sql)

    def _migrate(self) -> None:
        """Add columns to existing tables if missing (safe for fresh DBs)."""
        # Skip if segments table doesn't exist yet (fresh DB)
        try:
            cols = {
                r[1]
                for r in self.conn.execute("PRAGMA table_info(segments)").fetchall()
            }
        except sqlite3.OperationalError:
            return
        if not cols:
            return
        if "refutations" not in cols:
            self.conn.execute(
                "ALTER TABLE segments ADD COLUMN refutations INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
