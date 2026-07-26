"""Recognition: parallel wavefront propagation and intersection detection."""

from __future__ import annotations

from ..models.neuron import Neuron
from ..models.result import PathTrace, RecognitionResult
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo


# IS-A segments carry inheritance semantics, not propagation semantics.
# A wavefront arriving at `nerve_cell` may INHERIT definitional paths
# attached to `cell` (via upward IS-A walk at scoring time), but it
# must NOT leave `nerve_cell` via the IS-A edge to continue outward
# through `cell`'s other children (`daughter_cell`, `bacterial_cell`,
# …). That would cross-contaminate unrelated specializations sharing
# a head noun — the transformer polysemy problem this graph is meant
# to solve.
_NON_PROPAGATING_RELATIONS = frozenset({"is_a"})


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
        best_weights: dict[int, float] = {start.id: 0.0}
        # Queue stores: (current_neuron, path, total_strength, path_length)
        queue: list[tuple[Neuron, list[Neuron], float, int]] = [(start, [start], 0.0, 0)]

        depth = 0
        while queue and depth < self.max_depth:
            next_queue: list[tuple[Neuron, list[Neuron], float, int]] = []
            for current, path, total_str, path_len in queue:
                # Outgoing edges: property → relation → concept
                for seg in self.segment_repo.get_outgoing(current.id):
                    if seg.strength < min_strength:
                        continue
                    if seg.relation in _NON_PROPAGATING_RELATIONS:
                        continue
                    
                    new_total = total_str + seg.strength
                    new_len = path_len + 1
                    avg = new_total / new_len
                    
                    if seg.target_id not in best_weights or avg > best_weights[seg.target_id]:
                        target = self.neuron_repo.get_by_id(seg.target_id)
                        if target is None:
                            continue
                        best_weights[target.id] = avg
                        new_path = path + [target]
                        reached.setdefault(target.id, []).append(new_path)
                        next_queue.append((target, new_path, new_total, new_len))
                        
                # Incoming edges: concept ← relation ← property
                if bidirectional:
                    for seg in self.segment_repo.get_incoming(current.id):
                        if seg.strength < min_strength:
                            continue
                        if seg.relation in _NON_PROPAGATING_RELATIONS:
                            continue
                            
                        new_total = total_str + seg.strength
                        new_len = path_len + 1
                        avg = new_total / new_len
                        
                        if seg.source_id not in best_weights or avg > best_weights[seg.source_id]:
                            source = self.neuron_repo.get_by_id(seg.source_id)
                            if source is None:
                                continue
                            best_weights[source.id] = avg
                            new_path = path + [source]
                            reached.setdefault(source.id, []).append(new_path)
                            next_queue.append((source, new_path, new_total, new_len))
            queue = next_queue
            depth += 1

        return reached

    def _propagate_backwave(self, start: Neuron,
                            min_strength: float | None = None) -> dict[int, list[list[Neuron]]]:
        """BFS wavefront that implements true 'Backwave Propagation'.
        
        Starts moving FORWARD (following outgoing edges).
        If it hits a node with zero outgoing edges (a terminal Concept node),
        it reverses polarity to BACKWARD and only follows incoming edges
        from then on.
        """
        if min_strength is None:
            min_strength = self.min_strength

        reached: dict[int, list[list[Neuron]]] = {}
        best_weights: dict[int, float] = {start.id: 0.0}
        
        # Queue stores: (current_neuron, path, direction, total_strength, path_length)
        # direction is "F" (forward) or "B" (backward)
        queue: list[tuple[Neuron, list[Neuron], str, float, int]] = [(start, [start], "F", 0.0, 0)]

        depth = 0
        while queue and depth < self.max_depth:
            next_queue: list[tuple[Neuron, list[Neuron], str, float, int]] = []
            for current, path, direction, total_str, path_len in queue:
                
                # Try forward if direction is F
                if direction == "F":
                    outgoing = [seg for seg in self.segment_repo.get_outgoing(current.id) 
                                if seg.strength >= min_strength and seg.relation not in _NON_PROPAGATING_RELATIONS]
                    
                    if not outgoing:
                        # Hit a terminal concept node! Bounce backward!
                        direction = "B"
                    else:
                        for seg in outgoing:
                            new_total = total_str + seg.strength
                            new_len = path_len + 1
                            avg = new_total / new_len
                            
                            if seg.target_id not in best_weights or avg > best_weights[seg.target_id]:
                                target = self.neuron_repo.get_by_id(seg.target_id)
                                if target is None:
                                    continue
                                best_weights[target.id] = avg
                                new_path = path + [target]
                                reached.setdefault(target.id, []).append(new_path)
                                next_queue.append((target, new_path, "F", new_total, new_len))
                
                # If direction is B (or just flipped to B)
                if direction == "B":
                    incoming = [seg for seg in self.segment_repo.get_incoming(current.id)
                                if seg.strength >= min_strength and seg.relation not in _NON_PROPAGATING_RELATIONS]
                    for seg in incoming:
                        new_total = total_str + seg.strength
                        new_len = path_len + 1
                        avg = new_total / new_len
                        
                        if seg.source_id not in best_weights or avg > best_weights[seg.source_id]:
                            source = self.neuron_repo.get_by_id(seg.source_id)
                            if source is None:
                                continue
                            best_weights[source.id] = avg
                            new_path = path + [source]
                            reached.setdefault(source.id, []).append(new_path)
                            next_queue.append((source, new_path, "B", new_total, new_len))

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

    def inherit_definitions(self, node_id: int,
                            max_hops: int = 4) -> list[int]:
        """Walk IS-A edges upward from node_id, returning ancestor node ids.

        IS-A is traversed ONLY here, not during wavefront propagation.
        This lets the scorer pull in definitional paths attached to a
        compound's head (`nerve_cell`'s ancestors include `cell` and
        `container`) without letting wavefronts leak sideways across
        IS-A siblings.

        Returns ancestors in order of distance (nearest first).
        """
        ancestors: list[int] = []
        seen = {node_id}
        frontier = [node_id]
        for _ in range(max_hops):
            next_frontier: list[int] = []
            for nid in frontier:
                for seg in self.segment_repo.get_outgoing(nid):
                    if seg.relation != "is_a":
                        continue
                    if seg.target_id in seen:
                        continue
                    seen.add(seg.target_id)
                    ancestors.append(seg.target_id)
                    next_frontier.append(seg.target_id)
            if not next_frontier:
                break
            frontier = next_frontier
        return ancestors

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

    def propagate_backwave(self, seed_labels: list[str], short_term,
                           min_strength: float | None = None,
                           exact_only: bool = True) -> None:
        """True backwave propagation: Forward until dead-end, then Backward.
        
        This replaces propagate_echo by enforcing structural traversal
        (hit a concept end node -> bounce backward to find connections)
        instead of flooding the graph in all directions simultaneously.
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

        for seed in seeds:
            reached = self._propagate_backwave(seed, min_strength=effective_min)
            for target_id, path_lists in reached.items():
                if target_id == seed.id:
                    continue
                # Use the best path weight from this seed to this target.
                best_weight = max(
                    self._path_weight(p) for p in path_lists
                )
                short_term.add_convergence(target_id, best_weight, seed.id)

    def propagate_echo(self, seed_labels: list[str], short_term,
                       max_rounds: int = 2,
                       min_strength: float | None = None,
                       exact_only: bool = True,
                       top_k: int = 10) -> None:
        """Focused spreading activation — fast thought pinging.

        Iterative bidirectional propagation that only follows the TOP K
        most relevant neurons from each round. This prevents graph
        flooding while still finding deep connections.

        Args:
            seed_labels: initial concepts to start the echo from
            short_term: ShortTerm scratchpad to accumulate into
            max_rounds: how many echo bounces (keep low for speed)
            min_strength: edge threshold
            exact_only: exact label matching
            top_k: only propagate from the top N most convergent neurons
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

            # FIND TOP K NEURONS FOR NEXT ROUND
            # Only neurons discovered THIS round that haven't been seeds yet.
            # We apply Hub Inhibition here: highly connected nodes are less
            # likely to become the next "thought" seed.
            intersections = short_term.intersections(min_sources=1)
            next_candidates = []
            import math
            for nid, weight, _count in intersections:
                if nid not in used_ids:
                    n = self.neuron_repo.get_by_id(nid)
                    if n:
                        # Connectivity penalty: weight / (conn + 1)
                        out_count = len(self.segment_repo.get_outgoing(nid))
                        in_count = len(self.segment_repo.get_incoming(nid))
                        connectivity = out_count + in_count
                        
                        # Inhibit hubs from becoming seeds
                        inhibited_weight = weight / (connectivity + 1)
                        next_candidates.append((n, inhibited_weight))
            
            if not next_candidates:
                break
                
            # Sort by inhibited weight and pick top K
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            current_seeds = [c[0] for c in next_candidates[:top_k]]
            for n in current_seeds:
                used_ids.add(n.id)
