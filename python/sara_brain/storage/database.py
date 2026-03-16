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

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
