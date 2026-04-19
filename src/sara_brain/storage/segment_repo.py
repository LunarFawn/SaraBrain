from __future__ import annotations

import sqlite3

from ..models.segment import Segment


class SegmentRepo:
    def __init__(self, conn: sqlite3.Connection, prefix: str = "") -> None:
        self.conn = conn
        self._t = f"{prefix}_segments" if prefix else "segments"

    def create(self, segment: Segment) -> Segment:
        cur = self.conn.execute(
            f"INSERT INTO {self._t} (source_id, target_id, relation, "
            f"strength, traversals, refutations, created_at, last_used, "
            f"operation_tag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (segment.source_id, segment.target_id, segment.relation,
             segment.strength, segment.traversals, segment.refutations,
             segment.created_at, segment.last_used, segment.operation_tag),
        )
        segment.id = cur.lastrowid
        return segment

    def set_operation_tag(self, segment_id: int, tag: str | None) -> None:
        """Attach (or clear) the arithmetic operation tag on a segment."""
        self.conn.execute(
            f"UPDATE {self._t} SET operation_tag = ? WHERE id = ?",
            (tag, segment_id),
        )

    def get_by_id(self, segment_id: int) -> Segment | None:
        row = self.conn.execute(f"SELECT * FROM {self._t} WHERE id = ?", (segment_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_segment(row)

    def find(self, source_id: int, target_id: int, relation: str) -> Segment | None:
        row = self.conn.execute(
            f"SELECT * FROM {self._t} WHERE source_id = ? AND target_id = ? AND relation = ?",
            (source_id, target_id, relation),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_segment(row)

    def get_or_create(self, source_id: int, target_id: int, relation: str) -> tuple[Segment, bool]:
        existing = self.find(source_id, target_id, relation)
        if existing is not None:
            return existing, False
        seg = Segment(id=None, source_id=source_id, target_id=target_id, relation=relation)
        return self.create(seg), True

    def get_outgoing(self, neuron_id: int) -> list[Segment]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t} WHERE source_id = ? ORDER BY strength DESC",
            (neuron_id,),
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_incoming(self, neuron_id: int) -> list[Segment]:
        rows = self.conn.execute(
            f"SELECT * FROM {self._t} WHERE target_id = ? ORDER BY strength DESC",
            (neuron_id,),
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def strengthen(self, segment: Segment) -> None:
        segment.strengthen()
        self.conn.execute(
            f"UPDATE {self._t} SET strength = ?, traversals = ?, last_used = ? WHERE id = ?",
            (segment.strength, segment.traversals, segment.last_used, segment.id),
        )

    def weaken(self, segment: Segment) -> None:
        """Refute a segment — increment refutations, decrease strength.
        Path is never deleted; the refutation becomes part of the knowledge.
        """
        segment.weaken()
        self.conn.execute(
            f"UPDATE {self._t} SET strength = ?, refutations = ?, last_used = ? WHERE id = ?",
            (segment.strength, segment.refutations, segment.last_used, segment.id),
        )

    def list_all(self) -> list[Segment]:
        rows = self.conn.execute(f"SELECT * FROM {self._t} ORDER BY id").fetchall()
        return [self._row_to_segment(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {self._t}").fetchone()[0]

    @staticmethod
    def _row_to_segment(row: tuple) -> Segment:
        # Schema evolution:
        #   8 cols  = oldest (pre-refutations)
        #   9 cols  = +refutations
        #  10 cols  = +operation_tag (current)
        if len(row) == 8:
            return Segment(
                id=row[0], source_id=row[1], target_id=row[2],
                relation=row[3], strength=row[4], traversals=row[5],
                refutations=0, created_at=row[6], last_used=row[7],
            )
        if len(row) == 9:
            return Segment(
                id=row[0], source_id=row[1], target_id=row[2],
                relation=row[3], strength=row[4], traversals=row[5],
                refutations=row[6], created_at=row[7], last_used=row[8],
            )
        return Segment(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            relation=row[3],
            strength=row[4],
            traversals=row[5],
            refutations=row[6],
            created_at=row[7],
            last_used=row[8],
            operation_tag=row[9],
        )
