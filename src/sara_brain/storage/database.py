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

    def create_region(self, name: str, description: str = "") -> None:
        """Create a new brain region — its own set of tables."""
        import time as _time
        prefix = name.lower().replace("-", "_").replace(" ", "_")

        # Create the region's tables (mirrors the core schema)
        self.conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS {prefix}_neurons (
                id          INTEGER PRIMARY KEY,
                label       TEXT NOT NULL UNIQUE,
                neuron_type TEXT NOT NULL,
                created_at  REAL,
                metadata    TEXT
            );
            CREATE TABLE IF NOT EXISTS {prefix}_segments (
                id          INTEGER PRIMARY KEY,
                source_id   INTEGER NOT NULL REFERENCES {prefix}_neurons(id),
                target_id   INTEGER NOT NULL REFERENCES {prefix}_neurons(id),
                relation    TEXT NOT NULL,
                strength    REAL NOT NULL DEFAULT 1.0,
                traversals  INTEGER NOT NULL DEFAULT 0,
                refutations INTEGER NOT NULL DEFAULT 0,
                created_at  REAL,
                last_used   REAL,
                UNIQUE(source_id, target_id, relation)
            );
            CREATE TABLE IF NOT EXISTS {prefix}_paths (
                id          INTEGER PRIMARY KEY,
                origin_id   INTEGER NOT NULL REFERENCES {prefix}_neurons(id),
                terminus_id INTEGER NOT NULL REFERENCES {prefix}_neurons(id),
                source_text TEXT,
                created_at  REAL
            );
            CREATE TABLE IF NOT EXISTS {prefix}_path_steps (
                id          INTEGER PRIMARY KEY,
                path_id     INTEGER NOT NULL REFERENCES {prefix}_paths(id),
                step_order  INTEGER NOT NULL,
                segment_id  INTEGER NOT NULL REFERENCES {prefix}_segments(id),
                UNIQUE(path_id, step_order)
            );
            CREATE INDEX IF NOT EXISTS idx_{prefix}_seg_source
                ON {prefix}_segments(source_id, strength DESC);
            CREATE INDEX IF NOT EXISTS idx_{prefix}_seg_target
                ON {prefix}_segments(target_id);
            CREATE INDEX IF NOT EXISTS idx_{prefix}_neuron_label
                ON {prefix}_neurons(label);
            CREATE INDEX IF NOT EXISTS idx_{prefix}_path_terminus
                ON {prefix}_paths(terminus_id);
        """)

        # Register in the regions table
        self.conn.execute(
            "INSERT OR IGNORE INTO regions (name, description, created_at) "
            "VALUES (?, ?, ?)",
            (name, description, _time.time()),
        )
        self.conn.commit()

    def list_regions(self) -> list[dict]:
        """List all registered brain regions."""
        try:
            rows = self.conn.execute(
                "SELECT name, description, created_at FROM regions ORDER BY name"
            ).fetchall()
            return [{"name": r[0], "description": r[1], "created_at": r[2]}
                    for r in rows]
        except Exception:
            return []

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
