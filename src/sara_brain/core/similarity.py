"""Path similarity detection: observe commonality across parallel paths."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..models.neuron import NeuronType
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.queries import traverse_from


@dataclass
class SimilarityLink:
    neuron_a_label: str
    neuron_b_label: str
    shared_paths: int
    overlap_ratio: float


class SimilarityAnalyzer:
    def __init__(self, neuron_repo: NeuronRepo, segment_repo: SegmentRepo, conn) -> None:
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.conn = conn

    def analyze(self) -> list[SimilarityLink]:
        """Scan all property neurons, find pairs that share downstream paths."""
        properties = [
            n for n in self.neuron_repo.list_all()
            if n.neuron_type == NeuronType.PROPERTY
        ]

        # Build downstream reachability for each property
        downstream: dict[int, set[int]] = {}
        for prop in properties:
            rows = traverse_from(self.conn, prop.id)
            downstream[prop.id] = {row[0] for row in rows}

        # Compare all pairs
        links: list[SimilarityLink] = []
        prop_list = list(properties)
        for i in range(len(prop_list)):
            for j in range(i + 1, len(prop_list)):
                a = prop_list[i]
                b = prop_list[j]
                set_a = downstream.get(a.id, set())
                set_b = downstream.get(b.id, set())
                if not set_a or not set_b:
                    continue

                shared = set_a & set_b
                if not shared:
                    continue

                union = set_a | set_b
                overlap = len(shared) / len(union) if union else 0.0

                # Persist
                self._upsert_similarity(a.id, b.id, len(shared), overlap)

                links.append(SimilarityLink(
                    neuron_a_label=a.label,
                    neuron_b_label=b.label,
                    shared_paths=len(shared),
                    overlap_ratio=overlap,
                ))

        return links

    def get_similar(self, label: str) -> list[SimilarityLink]:
        """Get all similarity links for a given neuron."""
        neuron = self.neuron_repo.get_by_label(label.strip().lower())
        if neuron is None:
            return []

        rows = self.conn.execute(
            "SELECT neuron_a_id, neuron_b_id, shared_paths, overlap_ratio "
            "FROM similarities WHERE neuron_a_id = ? OR neuron_b_id = ?",
            (neuron.id, neuron.id),
        ).fetchall()

        links: list[SimilarityLink] = []
        for row in rows:
            other_id = row[1] if row[0] == neuron.id else row[0]
            other = self.neuron_repo.get_by_id(other_id)
            if other is None:
                continue
            links.append(SimilarityLink(
                neuron_a_label=neuron.label,
                neuron_b_label=other.label,
                shared_paths=row[2],
                overlap_ratio=row[3],
            ))

        return links

    def _upsert_similarity(self, a_id: int, b_id: int, shared: int, ratio: float) -> None:
        # Ensure consistent ordering
        if a_id > b_id:
            a_id, b_id = b_id, a_id
        self.conn.execute(
            "INSERT INTO similarities (neuron_a_id, neuron_b_id, shared_paths, overlap_ratio, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(neuron_a_id, neuron_b_id) DO UPDATE SET shared_paths=?, overlap_ratio=?, created_at=?",
            (a_id, b_id, shared, ratio, time.time(), shared, ratio, time.time()),
        )
        self.conn.commit()
