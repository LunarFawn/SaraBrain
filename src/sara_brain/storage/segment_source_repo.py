"""Repository for per-segment source provenance.

Tracks which distinct source labels (URLs, filenames, "user") produced
each segment. The UNIQUE(segment_id, source_label) constraint makes
"same source, same fact" a no-op — re-reading the same document does
not inflate the witness count.

Two-witness principle:
    witness_count(segment) >= 2 → confirmed (visible in queries)
    witness_count(segment) == 1 → tentative (below visibility floor)
    witness_count(segment) == 0 → no provenance (legacy data)
"""

from __future__ import annotations

import sqlite3
import time


class SegmentSourceRepo:
    def __init__(self, conn: sqlite3.Connection, prefix: str = "") -> None:
        self.conn = conn
        self._t = (
            f"{prefix}_segment_sources" if prefix else "segment_sources"
        )

    def add(self, segment_id: int, source_label: str) -> bool:
        """Record a source for a segment.

        Returns True if newly inserted (this is a new witness), False if
        the (segment_id, source_label) pair already existed (same source
        re-teaching the same fact — not a new witness).
        """
        label = source_label.strip()
        if not label:
            return False
        try:
            self.conn.execute(
                f"INSERT INTO {self._t} "
                f"(segment_id, source_label, created_at) VALUES (?, ?, ?)",
                (segment_id, label, time.time()),
            )
            return True
        except sqlite3.IntegrityError:
            # UNIQUE constraint — same source already recorded
            return False

    def count_distinct(self, segment_id: int) -> int:
        """Number of distinct source labels supporting this segment."""
        row = self.conn.execute(
            f"SELECT COUNT(DISTINCT source_label) FROM {self._t} "
            f"WHERE segment_id = ?",
            (segment_id,),
        ).fetchone()
        return row[0] if row else 0

    def list_sources(self, segment_id: int) -> list[str]:
        """All distinct source labels for a segment, oldest first."""
        rows = self.conn.execute(
            f"SELECT source_label FROM {self._t} "
            f"WHERE segment_id = ? ORDER BY created_at",
            (segment_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def count_distinct_for_segments(self, segment_ids: list[int]) -> int:
        """Minimum distinct source count across a set of segments.

        For a path (or any multi-segment claim), the weakest link
        determines the witness count. A path is as confirmed as its
        least-confirmed segment.
        """
        if not segment_ids:
            return 0
        counts = []
        for sid in segment_ids:
            counts.append(self.count_distinct(sid))
        return min(counts) if counts else 0
