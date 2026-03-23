from __future__ import annotations

import sqlite3


class CategoryRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def set_category(self, label: str, category: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO categories (label, category) VALUES (?, ?)",
            (label, category),
        )

    def get_category(self, label: str) -> str | None:
        row = self.conn.execute(
            "SELECT category FROM categories WHERE label = ?",
            (label,),
        ).fetchone()
        return row[0] if row else None

    def list_by_category(self, category: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT label FROM categories WHERE category = ? ORDER BY label",
            (category,),
        ).fetchall()
        return [r[0] for r in rows]

    def list_categories(self) -> dict[str, list[str]]:
        rows = self.conn.execute(
            "SELECT label, category FROM categories ORDER BY category, label"
        ).fetchall()
        result: dict[str, list[str]] = {}
        for label, category in rows:
            result.setdefault(category, []).append(label)
        return result
