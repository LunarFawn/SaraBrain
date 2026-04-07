"""Account repository — CRUD for user accounts."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass


@dataclass
class Account:
    id: int | None
    name: str
    role: str  # 'reader', 'teacher', 'doctor'
    pin_hash: str | None = None
    neuron_id: int | None = None
    created_at: float | None = None
    is_active: bool = True


class AccountRepo:
    VALID_ROLES = ("reader", "teacher", "doctor")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, name: str, role: str, pin: str | None = None,
               neuron_id: int | None = None) -> Account:
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {self.VALID_ROLES}")
        pin_hash = self._hash_pin(pin) if pin else None
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO accounts (name, role, pin_hash, neuron_id, created_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, 1)",
            (name, role, pin_hash, neuron_id, now),
        )
        return Account(
            id=cur.lastrowid, name=name, role=role,
            pin_hash=pin_hash, neuron_id=neuron_id,
            created_at=now, is_active=True,
        )

    def get_by_id(self, account_id: int) -> Account | None:
        row = self.conn.execute(
            "SELECT * FROM accounts WHERE id = ?", (account_id,)
        ).fetchone()
        return self._row_to_account(row) if row else None

    def get_by_name(self, name: str) -> Account | None:
        row = self.conn.execute(
            "SELECT * FROM accounts WHERE name = ? AND is_active = 1",
            (name,),
        ).fetchone()
        return self._row_to_account(row) if row else None

    def list_all(self, active_only: bool = True) -> list[Account]:
        if active_only:
            rows = self.conn.execute(
                "SELECT * FROM accounts WHERE is_active = 1 ORDER BY id"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM accounts ORDER BY id"
            ).fetchall()
        return [self._row_to_account(r) for r in rows]

    def list_by_role(self, role: str) -> list[Account]:
        rows = self.conn.execute(
            "SELECT * FROM accounts WHERE role = ? AND is_active = 1 ORDER BY id",
            (role,),
        ).fetchall()
        return [self._row_to_account(r) for r in rows]

    def get_reader(self) -> Account | None:
        """Get the patient (reader) account. Assumes one reader per instance."""
        readers = self.list_by_role("reader")
        return readers[0] if readers else None

    def verify_pin(self, account_id: int, pin: str) -> bool:
        account = self.get_by_id(account_id)
        if account is None:
            return False
        if account.pin_hash is None:
            return True  # no PIN set means open access
        return account.pin_hash == self._hash_pin(pin)

    def deactivate(self, account_id: int) -> None:
        self.conn.execute(
            "UPDATE accounts SET is_active = 0 WHERE id = ?", (account_id,)
        )

    @staticmethod
    def _hash_pin(pin: str) -> str:
        return hashlib.sha256(pin.encode()).hexdigest()

    @staticmethod
    def _row_to_account(row: tuple) -> Account:
        return Account(
            id=row[0],
            name=row[1],
            role=row[2],
            pin_hash=row[3],
            neuron_id=row[4],
            created_at=row[5],
            is_active=bool(row[6]),
        )
