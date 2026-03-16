from __future__ import annotations

import sqlite3
import time


class AssociationRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, association: str, property_label: str, neuron_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO associations (association, property_label, neuron_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (association, property_label, neuron_id, time.time()),
        )

    def list_all(self) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT association, property_label FROM associations ORDER BY association, property_label"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_properties(self, association: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT property_label FROM associations WHERE association = ? ORDER BY property_label",
            (association,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_associations(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT association FROM associations ORDER BY association"
        ).fetchall()
        return [r[0] for r in rows]

    def set_question_word(self, association: str, question_word: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO question_words (association, question_word) VALUES (?, ?)",
            (association, question_word),
        )

    def get_by_question_word(self, qword: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT association FROM question_words WHERE question_word = ? ORDER BY association",
            (qword,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_question_word(self, association: str) -> str | None:
        row = self.conn.execute(
            "SELECT question_word FROM question_words WHERE association = ?",
            (association,),
        ).fetchone()
        return row[0] if row else None

    def list_question_words(self) -> list[tuple[str, str]]:
        rows = self.conn.execute(
            "SELECT association, question_word FROM question_words ORDER BY question_word, association"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]
