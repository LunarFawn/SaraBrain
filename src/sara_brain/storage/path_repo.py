from __future__ import annotations

import sqlite3

from ..models.path import Path, PathStep


class PathRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, path: Path) -> Path:
        cur = self.conn.execute(
            "INSERT INTO paths (origin_id, terminus_id, source_text, created_at) VALUES (?, ?, ?, ?)",
            (path.origin_id, path.terminus_id, path.source_text, path.created_at),
        )
        path.id = cur.lastrowid
        return path

    def add_step(self, step: PathStep) -> PathStep:
        cur = self.conn.execute(
            "INSERT INTO path_steps (path_id, step_order, segment_id) VALUES (?, ?, ?)",
            (step.path_id, step.step_order, step.segment_id),
        )
        step.id = cur.lastrowid
        return step

    def get_by_id(self, path_id: int) -> Path | None:
        row = self.conn.execute("SELECT * FROM paths WHERE id = ?", (path_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_path(row)

    def get_steps(self, path_id: int) -> list[PathStep]:
        rows = self.conn.execute(
            "SELECT * FROM path_steps WHERE path_id = ? ORDER BY step_order",
            (path_id,),
        ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def get_paths_to(self, terminus_id: int) -> list[Path]:
        rows = self.conn.execute(
            "SELECT * FROM paths WHERE terminus_id = ? ORDER BY id",
            (terminus_id,),
        ).fetchall()
        return [self._row_to_path(r) for r in rows]

    def get_paths_from(self, origin_id: int) -> list[Path]:
        rows = self.conn.execute(
            "SELECT * FROM paths WHERE origin_id = ? ORDER BY id",
            (origin_id,),
        ).fetchall()
        return [self._row_to_path(r) for r in rows]

    def list_all(self) -> list[Path]:
        rows = self.conn.execute("SELECT * FROM paths ORDER BY id").fetchall()
        return [self._row_to_path(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM paths").fetchone()[0]

    @staticmethod
    def _row_to_path(row: tuple) -> Path:
        return Path(
            id=row[0],
            origin_id=row[1],
            terminus_id=row[2],
            source_text=row[3],
            created_at=row[4],
        )

    @staticmethod
    def _row_to_step(row: tuple) -> PathStep:
        return PathStep(
            id=row[0],
            path_id=row[1],
            step_order=row[2],
            segment_id=row[3],
        )
