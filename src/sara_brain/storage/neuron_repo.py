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

    def resolve(self, label: str) -> Neuron | None:
        """Fuzzy neuron lookup — returns best match or None.

        For queries where only one result is needed. If the match required
        edit distance (misspelling), use resolve_candidates() instead to
        let the user confirm.
        """
        candidates = self.resolve_candidates(label)
        if not candidates:
            return None
        # If exact/inflect/prefix match, return directly
        # If edit distance match, still return best — caller can use
        # resolve_candidates() for disambiguation
        return candidates[0][0]

    def resolve_candidates(self, label: str, max_results: int = 5) -> list[tuple[Neuron, int, str]]:
        """Fuzzy neuron lookup returning ranked candidates.

        Returns list of (neuron, edit_distance, match_method) sorted by
        best match first. edit_distance=0 means exact/inflect/prefix match.

        Match methods: 'exact', 'inflect', 'prefix', 'contains', 'fuzzy'
        """
        label = label.strip().lower()
        results: list[tuple[Neuron, int, str]] = []
        seen_ids: set[int] = set()

        def _add(neuron: Neuron, dist: int, method: str) -> None:
            if neuron.id not in seen_ids:
                seen_ids.add(neuron.id)
                results.append((neuron, dist, method))

        # 1. Exact match
        n = self.get_by_label(label)
        if n is not None:
            _add(n, 0, "exact")
            return results  # exact match, no need to keep looking

        # 2. Plural/singular variants
        for v in self._inflect(label):
            n = self.get_by_label(v)
            if n is not None:
                _add(n, 0, "inflect")
        if results:
            return results  # inflection match is confident

        # 3. Prefix match — 'sumer' matches 'sumerian'
        rows = self.conn.execute(
            "SELECT * FROM neurons WHERE label LIKE ? ORDER BY length(label) LIMIT ?",
            (label + "%", max_results),
        ).fetchall()
        for row in rows:
            _add(self._row_to_neuron(row), 0, "prefix")
        if results:
            return results  # prefix match is confident

        # 4. Contains — 'sumer' found inside 'ancient sumerian'
        rows = self.conn.execute(
            "SELECT * FROM neurons WHERE label LIKE ? ORDER BY length(label) LIMIT ?",
            ("%" + label + "%", max_results),
        ).fetchall()
        for row in rows:
            _add(self._row_to_neuron(row), 0, "contains")
        if results:
            return results

        # 5. Edit distance — handles misspellings
        max_dist = max(2, len(label) // 3)
        candidates: list[tuple[tuple, int]] = []
        all_neurons = self.conn.execute("SELECT * FROM neurons").fetchall()
        for row in all_neurons:
            stored = row[1]
            # Check direct edit distance
            d = self._edit_distance(label, stored, max_dist)
            if d <= max_dist:
                candidates.append((row, d))
                continue
            # Check inflected variants of stored label
            for v in self._inflect(stored):
                d2 = self._edit_distance(label, v, max_dist)
                if d2 <= max_dist:
                    candidates.append((row, d2))
                    break

        # Sort by edit distance, then by length similarity
        candidates.sort(key=lambda x: (x[1], abs(len(label) - len(x[0][1]))))

        for row, dist in candidates[:max_results]:
            _add(self._row_to_neuron(row), dist, "fuzzy")

        return results

    @staticmethod
    def _edit_distance(a: str, b: str, max_dist: int) -> int:
        """Levenshtein distance with early exit. Pure Python, no deps."""
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        if a == b:
            return 0
        la, lb = len(a), len(b)
        # Use single-row DP for memory efficiency
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            curr = [i] + [0] * lb
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            # Early exit if minimum in row exceeds max_dist
            if min(curr) > max_dist:
                return max_dist + 1
            prev = curr
        return prev[lb]

    @staticmethod
    def _inflect(word: str) -> list[str]:
        """Generate singular/plural variants of a word."""
        variants = []
        # Try adding common suffixes
        variants.append(word + "s")
        variants.append(word + "es")
        variants.append(word + "ian")
        variants.append(word + "ians")
        variants.append(word + "an")
        # Try removing common suffixes
        if word.endswith("ians"):
            variants.append(word[:-4])
            variants.append(word[:-1])  # -ian
        elif word.endswith("ian"):
            variants.append(word[:-3])
            variants.append(word + "s")  # -ians
        elif word.endswith("ans"):
            variants.append(word[:-3])
            variants.append(word[:-1])  # -an
        elif word.endswith("an"):
            variants.append(word[:-2])
        if word.endswith("ies"):
            variants.append(word[:-3] + "y")
        elif word.endswith("es"):
            variants.append(word[:-2])
        elif word.endswith("s") and not word.endswith("ss"):
            variants.append(word[:-1])
        return variants

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
