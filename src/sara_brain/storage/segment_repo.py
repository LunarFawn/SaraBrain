from __future__ import annotations

import sqlite3

from ..models.segment import Segment


class SegmentRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, segment: Segment) -> Segment:
        cur = self.conn.execute(
            "INSERT INTO segments (source_id, target_id, relation, strength, traversals, refutations, created_at, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (segment.source_id, segment.target_id, segment.relation,
             segment.strength, segment.traversals, segment.refutations,
             segment.created_at, segment.last_used),
        )
        segment.id = cur.lastrowid
        return segment

    def get_by_id(self, segment_id: int) -> Segment | None:
        row = self.conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_segment(row)

    def find(self, source_id: int, target_id: int, relation: str) -> Segment | None:
        row = self.conn.execute(
            "SELECT * FROM segments WHERE source_id = ? AND target_id = ? AND relation = ?",
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
            "SELECT * FROM segments WHERE source_id = ? ORDER BY strength DESC",
            (neuron_id,),
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def get_incoming(self, neuron_id: int) -> list[Segment]:
        rows = self.conn.execute(
            "SELECT * FROM segments WHERE target_id = ? ORDER BY strength DESC",
            (neuron_id,),
        ).fetchall()
        return [self._row_to_segment(r) for r in rows]

    def strengthen(self, segment: Segment) -> None:
        segment.strengthen()
        self.conn.execute(
            "UPDATE segments SET strength = ?, traversals = ?, last_used = ? WHERE id = ?",
            (segment.strength, segment.traversals, segment.last_used, segment.id),
        )

    def weaken(self, segment: Segment) -> None:
        """Refute a segment — increment refutations, decrease strength.
        Path is never deleted; the refutation becomes part of the knowledge.
        """
        segment.weaken()
        self.conn.execute(
            "UPDATE segments SET strength = ?, refutations = ?, last_used = ? WHERE id = ?",
            (segment.strength, segment.refutations, segment.last_used, segment.id),
        )

    def list_all(self) -> list[Segment]:
        rows = self.conn.execute("SELECT * FROM segments ORDER BY id").fetchall()
        return [self._row_to_segment(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]

    @staticmethod
    def _row_to_segment(row: tuple) -> Segment:
        # Schema: id, source_id, target_id, relation, strength, traversals,
        #         refutations (new), created_at, last_used
        # Backward compat: if refutations column missing, default to 0
        if len(row) == 8:
            # Old schema without refutations
            return Segment(
                id=row[0], source_id=row[1], target_id=row[2], relation=row[3],
                strength=row[4], traversals=row[5], refutations=0,
                created_at=row[6], last_used=row[7],
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
        )
