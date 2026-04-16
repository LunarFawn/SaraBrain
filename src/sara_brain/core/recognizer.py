"""Recognition: parallel wavefront propagation and intersection detection."""

from __future__ import annotations

from ..models.neuron import Neuron
from ..models.result import PathTrace, RecognitionResult
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo


class Recognizer:
    def __init__(
        self,
        neuron_repo: NeuronRepo,
        segment_repo: SegmentRepo,
        max_depth: int = 3,
        min_strength: float = 0.5,
    ) -> None:
        """Propagation engine for path-of-thought recognition.

        max_depth: BFS hop limit. Shallow is the right default — biological
            signals converge in a few hops. Large max_depth floods a dense
            graph and makes every concept appear to intersect with every
            other one (the transformer attention problem).
        min_strength: Segments below this are pruned from traversal. Weak
            associations (0.1) and refuted segments (<0) get filtered out
            by default. Pass 0.0 to include everything (useful for
            debugging or querying with full context).
        """
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.max_depth = max_depth
        self.min_strength = min_strength

    def recognize(self, input_labels: list[str],
                  min_strength: float | None = None) -> list[RecognitionResult]:
        """Launch parallel wavefronts from all input neurons, find intersections."""
        # Resolve input labels to neurons
        start_neurons: list[Neuron] = []
        for label in input_labels:
            n = self.neuron_repo.resolve(label.strip().lower())
            if n is not None:
                start_neurons.append(n)

        if not start_neurons:
            return []

        # Effective min_strength: caller override, else instance default
        effective_min = self.min_strength if min_strength is None else min_strength

        # Launch parallel wavefronts (each independently explores the graph)
        wavefront_results: dict[int, dict[int, list[list[Neuron]]]] = {}
        for neuron in start_neurons:
            wavefront_results[neuron.id] = self._propagate(
                neuron, min_strength=effective_min
            )

        # Find intersections: neurons reached by 2+ wavefronts
        all_reached: dict[int, dict[int, list[list[Neuron]]]] = {}
        for source_id, reached in wavefront_results.items():
            for target_id, paths in reached.items():
                all_reached.setdefault(target_id, {})[source_id] = paths

        results: list[RecognitionResult] = []
        for target_id, sources in all_reached.items():
            target_neuron = self.neuron_repo.get_by_id(target_id)
            if target_neuron is None:
                continue

            traces: list[PathTrace] = []
            for _source_id, path_lists in sources.items():
                for path_neurons in path_lists:
                    weight = self._path_weight(path_neurons)
                    traces.append(PathTrace(neurons=path_neurons, weight=weight))

            results.append(RecognitionResult(neuron=target_neuron, converging_paths=traces))

        # Strengthen traversed segments (only for non-refuted recognitions)
        self._strengthen_traversed(wavefront_results)

        # Sort by signed confidence (most strongly recognized first;
        # refuted concepts sink to the bottom)
        results.sort(key=lambda r: r.signed_confidence, reverse=True)

        return results

    def _path_weight(self, path_neurons: list[Neuron]) -> float:
        """Compute the signed weight of a path as the average of segment strengths.

        A path of strong (positive) segments has positive weight.
        A path of refuted (negative) segments has negative weight.
        Mixed paths cancel proportionally.
        """
        if len(path_neurons) < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(len(path_neurons) - 1):
            src_id = path_neurons[i].id
            tgt_id = path_neurons[i + 1].id
            segments = self.segment_repo.get_outgoing(src_id)
            for seg in segments:
                if seg.target_id == tgt_id:
                    total += seg.strength
                    count += 1
                    break
        return total / count if count > 0 else 0.0

    def _propagate(self, start: Neuron,
                   min_strength: float | None = None,
                   bidirectional: bool = False) -> dict[int, list[list[Neuron]]]:
        """BFS wavefront from a single neuron. Returns {reached_id: [[path_neurons]]}.

        Segments with strength < min_strength are pruned from traversal.

        When bidirectional=True, each step follows BOTH outgoing AND
        incoming edges. This lets wavefronts that start from concept
        neurons (which are path termini with no outgoing edges) discover
        their connected property neurons by walking backward. Thoughts
        ping in both directions.
        """
        if min_strength is None:
            min_strength = self.min_strength

        reached: dict[int, list[list[Neuron]]] = {}
        queue: list[tuple[Neuron, list[Neuron]]] = [(start, [start])]
        visited: set[int] = {start.id}

        depth = 0
        while queue and depth < self.max_depth:
            next_queue: list[tuple[Neuron, list[Neuron]]] = []
            for current, path in queue:
                # Outgoing edges: property → relation → concept
                for seg in self.segment_repo.get_outgoing(current.id):
                    if seg.strength < min_strength:
                        continue
                    if seg.target_id in visited:
                        continue
                    target = self.neuron_repo.get_by_id(seg.target_id)
                    if target is None:
                        continue
                    visited.add(target.id)
                    new_path = path + [target]
                    reached.setdefault(target.id, []).append(new_path)
                    next_queue.append((target, new_path))
                # Incoming edges: concept ← relation ← property
                if bidirectional:
                    for seg in self.segment_repo.get_incoming(current.id):
                        if seg.strength < min_strength:
                            continue
                        if seg.source_id in visited:
                            continue
                        source = self.neuron_repo.get_by_id(seg.source_id)
                        if source is None:
                            continue
                        visited.add(source.id)
                        new_path = path + [source]
                        reached.setdefault(source.id, []).append(new_path)
                        next_queue.append((source, new_path))
            queue = next_queue
            depth += 1

        return reached

    def _strengthen_traversed(
        self, wavefront_results: dict[int, dict[int, list[list[Neuron]]]]
    ) -> None:
        """Strengthen all segments that were traversed during recognition."""
        strengthened: set[tuple[int, int]] = set()
        for _source_id, reached in wavefront_results.items():
            for _target_id, path_lists in reached.items():
                for path_neurons in path_lists:
                    for i in range(len(path_neurons) - 1):
                        pair = (path_neurons[i].id, path_neurons[i + 1].id)
                        if pair in strengthened:
                            continue
                        strengthened.add(pair)
                        segments = self.segment_repo.get_outgoing(pair[0])
                        for seg in segments:
                            if seg.target_id == pair[1]:
                                self.segment_repo.strengthen(seg)

    def trace(self, label: str,
              min_strength: float | None = None) -> list[PathTrace]:
        """Trace all outgoing paths from a neuron."""
        neuron = self.neuron_repo.resolve(label.strip().lower())
        if neuron is None:
            return []

        reached = self._propagate(neuron, min_strength=min_strength)
        traces: list[PathTrace] = []
        for _target_id, path_lists in reached.items():
            for path_neurons in path_lists:
                weight = self._path_weight(path_neurons)
                traces.append(PathTrace(neurons=path_neurons, weight=weight))

        return traces

    def propagate_into(self, seed_labels: list[str], short_term,
                       min_strength: float | None = None,
                       exact_only: bool = True) -> None:
        """Launch wavefronts from each seed, accumulate convergence into short_term.

        READ-ONLY: does not call _strengthen_traversed. Segments are not
        mutated. This is the query path that respects the principle
        "just looking at a path should not strengthen it; only being
        told something is right should."

        Args:
            seed_labels: concept labels to launch wavefronts from
            short_term: ShortTerm instance to accumulate convergence into
            min_strength: override for pruning weak edges
            exact_only: when True (default for queries), seed labels must
                match exactly. Fuzzy/prefix/contains matching at query
                time creates false signal — "anther" silently becoming
                "another" poisons the convergence. Fuzzy belongs in
                ingest and disambiguation, not in quiet query paths.

        The short_term argument is a ShortTerm instance — each
        (target, seed) reachability is recorded via short_term.add_convergence.
        Neurons reached from multiple distinct seeds show up as real
        intersections in short_term.intersections().
        """
        # Resolve seeds to neurons (exact match by default at query time)
        seeds = []
        for label in seed_labels:
            n = self.neuron_repo.resolve(
                label.strip().lower(), exact_only=exact_only
            )
            if n is not None:
                seeds.append(n)
        if not seeds:
            return

        effective_min = (
            self.min_strength if min_strength is None else min_strength
        )

        # Each seed independently propagates; the short_term accumulates
        # the union. No strengthening, no mutation.
        for seed in seeds:
            reached = self._propagate(seed, min_strength=effective_min)
            for target_id, path_lists in reached.items():
                if target_id == seed.id:
                    continue
                # Use the best path weight from this seed to this target.
                # Multiple paths may exist; keep the strongest.
                best_weight = max(
                    self._path_weight(p) for p in path_lists
                )
                short_term.add_convergence(target_id, best_weight, seed.id)

    def propagate_echo(self, seed_labels: list[str], short_term,
                       max_rounds: int = 3,
                       min_strength: float | None = None,
                       exact_only: bool = True) -> None:
        """Spreading activation — thought pinging around the graph.

        Iterative bidirectional propagation that echoes back and forth
        until it settles. Each round takes neurons discovered in the
        PREVIOUS round (that haven't been used as seeds yet) and
        propagates them bidirectionally. Everything accumulates in the
        same ShortTerm scratchpad.

        This models how thoughts ping around a brain: "baseballs → I
        like balls → balls are round → I want an orange." Each
        convergence triggers a new wave. The echo settles when no new
        neurons are discovered or max_rounds is reached.

        READ-ONLY: no segment strengthening.

        Args:
            seed_labels: initial concepts to start the echo from
                (question words + choice words typically)
            short_term: ShortTerm scratchpad to accumulate into
            max_rounds: how many echo bounces before stopping
            min_strength: edge threshold for this echo
            exact_only: exact label matching (default for queries)
        """
        effective_min = (
            self.min_strength if min_strength is None else min_strength
        )

        # Resolve initial seeds
        used_ids: set[int] = set()
        current_seeds: list[Neuron] = []
        for label in seed_labels:
            n = self.neuron_repo.resolve(
                label.strip().lower(), exact_only=exact_only
            )
            if n is not None and n.id not in used_ids:
                current_seeds.append(n)
                used_ids.add(n.id)

        if not current_seeds:
            return

        for _round in range(max_rounds):
            new_neurons_this_round: list[Neuron] = []

            for seed in current_seeds:
                reached = self._propagate(
                    seed,
                    min_strength=effective_min,
                    bidirectional=True,
                )
                for target_id, path_lists in reached.items():
                    if target_id == seed.id:
                        continue
                    best_weight = max(
                        self._path_weight(p) for p in path_lists
                    )
                    short_term.add_convergence(
                        target_id, best_weight, seed.id
                    )
                    # If this neuron is new, queue it as a seed for the
                    # next round — the thought pings forward
                    if target_id not in used_ids:
                        used_ids.add(target_id)
                        target_n = self.neuron_repo.get_by_id(target_id)
                        if target_n is not None:
                            new_neurons_this_round.append(target_n)

            # If nothing new was discovered, the echo has settled
            if not new_neurons_this_round:
                break

            # Next round's seeds are the newly discovered neurons
            current_seeds = new_neurons_this_round
