from __future__ import annotations

import sqlite3

from ..models.neuron import Neuron, NeuronType


class NeuronRepo:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, neuron: Neuron) -> Neuron:
        cur = self.conn.execute(
            "INSERT INTO neurons (label, neuron_type, created_at, metadata) VALUES (?, ?, ?, ?)",
            (neuron.label, neuron.neuron_type.value, neuron.created_at, neuron.metadata_json()),
        )
        neuron.id = cur.lastrowid
        return neuron

    def get_by_id(self, neuron_id: int) -> Neuron | None:
        row = self.conn.execute("SELECT * FROM neurons WHERE id = ?", (neuron_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_neuron(row)

    def get_by_label(self, label: str) -> Neuron | None:
        row = self.conn.execute("SELECT * FROM neurons WHERE label = ?", (label,)).fetchone()
        if row is None:
            return None
        return self._row_to_neuron(row)

    def get_or_create(self, label: str, neuron_type: NeuronType) -> tuple[Neuron, bool]:
        """Return (neuron, created). If exists, returns it; otherwise creates."""
        existing = self.get_by_label(label)
        if existing is not None:
            return existing, False
        neuron = Neuron(id=None, label=label, neuron_type=neuron_type)
        return self.create(neuron), True

    def list_all(self) -> list[Neuron]:
        rows = self.conn.execute("SELECT * FROM neurons ORDER BY id").fetchall()
        return [self._row_to_neuron(r) for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]

    @staticmethod
    def _row_to_neuron(row: tuple) -> Neuron:
        return Neuron(
            id=row[0],
            label=row[1],
            neuron_type=NeuronType(row[2]),
            created_at=row[3],
            metadata=Neuron.metadata_from_json(row[4]),
        )
