"""Interaction repository — append-only log of all exchanges with Sara."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field


@dataclass
class Interaction:
    id: int | None
    account_id: int
    interaction_type: str  # 'tell', 'ask', 'teach', 'review'
    content: str
    response: str | None = None
    path_ids: list[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class InteractionRepo:
    VALID_TYPES = ("tell", "ask", "teach", "review")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def record(self, interaction: Interaction) -> Interaction:
        path_ids_json = json.dumps(interaction.path_ids) if interaction.path_ids else None
        cur = self.conn.execute(
            "INSERT INTO interactions (account_id, interaction_type, content, response, path_ids, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (interaction.account_id, interaction.interaction_type,
             interaction.content, interaction.response,
             path_ids_json, interaction.created_at),
        )
        interaction.id = cur.lastrowid
        return interaction

    def get_by_id(self, interaction_id: int) -> Interaction | None:
        row = self.conn.execute(
            "SELECT * FROM interactions WHERE id = ?", (interaction_id,)
        ).fetchone()
        return self._row_to_interaction(row) if row else None

    def get_by_account(self, account_id: int, since: float | None = None,
                       until: float | None = None,
                       type_filter: str | None = None) -> list[Interaction]:
        query = "SELECT * FROM interactions WHERE account_id = ?"
        params: list = [account_id]
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since)
        if until is not None:
            query += " AND created_at <= ?"
            params.append(until)
        if type_filter is not None:
            query += " AND interaction_type = ?"
            params.append(type_filter)
        query += " ORDER BY created_at"
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_interaction(r) for r in rows]

    def get_by_time_range(self, since: float, until: float,
                          account_id: int | None = None) -> list[Interaction]:
        if account_id is not None:
            return self.get_by_account(account_id, since=since, until=until)
        rows = self.conn.execute(
            "SELECT * FROM interactions WHERE created_at >= ? AND created_at <= ? ORDER BY created_at",
            (since, until),
        ).fetchall()
        return [self._row_to_interaction(r) for r in rows]

    def count_by_type(self, account_id: int, interaction_type: str,
                      since: float | None = None) -> int:
        query = "SELECT COUNT(*) FROM interactions WHERE account_id = ? AND interaction_type = ?"
        params: list = [account_id, interaction_type]
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since)
        return self.conn.execute(query, params).fetchone()[0]

    @staticmethod
    def _row_to_interaction(row: tuple) -> Interaction:
        path_ids = json.loads(row[5]) if row[5] else []
        return Interaction(
            id=row[0],
            account_id=row[1],
            interaction_type=row[2],
            content=row[3],
            response=row[4],
            path_ids=path_ids,
            created_at=row[6],
        )
