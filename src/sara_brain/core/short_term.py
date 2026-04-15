"""Short-term (working) memory region for Sara Brain.

A session-scoped scratchpad that holds the state of a single event — a
query, an ingest, a benchmark question, a conversation turn. It is NOT
written to the long-term path graph. It is discarded when the event ends
unless explicitly consolidated (consolidation is a future feature).

Why this exists: the long-term graph should not be mutated by the act of
looking at it. A real brain has a hippocampus (working memory) where
current-event context is held, and a cortex (long-term storage) where
consolidated knowledge lives. Sleep decides what transfers. Sara needed
the hippocampus it was missing — this is it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShortTerm:
    """Session-scoped working memory. Not persisted to the graph.

    Holds wavefront convergence maps, tentative observations, and
    significance markers for the current event. Intended to be created
    via ``brain.short_term()`` as a context manager, used during an
    event, and discarded when the context exits. Consolidation
    (write-back to long-term) is a future feature.
    """

    event_id: str  # timestamp or session id
    event_type: str  # "query", "ingest", "benchmark", etc.

    # convergence_map: {neuron_id: accumulated_weight}
    # Weights from every wavefront that reached this neuron sum here.
    convergence_map: dict[int, float] = field(default_factory=dict)

    # convergence_sources: {neuron_id: {source_seed_id, ...}}
    # Tracks which starting seeds each converging neuron was reached
    # from. Count >= 2 means multiple independent wavefronts converged
    # on that neuron — a real intersection, not a single-path neighbor.
    convergence_sources: dict[int, set[int]] = field(default_factory=dict)

    # Tentative observations — facts noted but not committed to long-term
    observations: list[str] = field(default_factory=list)

    # Significance markers — reasons something might warrant consolidation.
    # Each entry is a dict like {"type": "correction", "content": "..."}.
    significance: list[dict] = field(default_factory=list)

    def add_convergence(self, neuron_id: int, weight: float,
                        source_id: int) -> None:
        """Record that a wavefront from source_id reached neuron_id with weight."""
        self.convergence_map[neuron_id] = (
            self.convergence_map.get(neuron_id, 0.0) + weight
        )
        self.convergence_sources.setdefault(neuron_id, set()).add(source_id)

    def intersections(self, min_sources: int = 2
                      ) -> list[tuple[int, float, int]]:
        """Return neurons reached by >= min_sources distinct wavefronts.

        Returns list of (neuron_id, total_weight, source_count) sorted
        by total weight descending. These are the true intersections —
        points where multiple independent lines of reasoning converge.
        """
        result = []
        for nid, sources in self.convergence_sources.items():
            if len(sources) >= min_sources:
                result.append((nid, self.convergence_map[nid], len(sources)))
        result.sort(key=lambda t: t[1], reverse=True)
        return result

    def align_score(self, candidate_neuron_ids: list[int]
                    ) -> tuple[float, int]:
        """Measure how well a candidate's concepts align with the convergence.

        For multiple-choice reasoning: the question builds a convergence
        map, then each choice's concept ids are checked against it.
        Returns (total_weight_of_hits, count_of_hits).
        """
        weight = 0.0
        hits = 0
        for nid in candidate_neuron_ids:
            if nid in self.convergence_map:
                weight += self.convergence_map[nid]
                hits += 1
        return weight, hits

    def mark_significant(self, marker_type: str, content: str) -> None:
        """Attach a significance marker — justification for consolidation.

        Consolidation (future feature) will inspect these markers to
        decide whether any of this short-term state should transfer to
        long-term. Markers like 'correction', 'novelty', 'user_confirmed',
        'contradiction_resolved' are the signals that something matters.
        """
        self.significance.append({"type": marker_type, "content": content})
