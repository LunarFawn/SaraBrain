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
        max_depth: int = 10,
    ) -> None:
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.max_depth = max_depth

    def recognize(self, input_labels: list[str]) -> list[RecognitionResult]:
        """Launch parallel wavefronts from all input neurons, find intersections."""
        # Resolve input labels to neurons
        start_neurons: list[Neuron] = []
        for label in input_labels:
            n = self.neuron_repo.get_by_label(label.strip().lower())
            if n is not None:
                start_neurons.append(n)

        if not start_neurons:
            return []

        # Launch parallel wavefronts (each independently explores the graph)
        wavefront_results: dict[int, dict[int, list[list[Neuron]]]] = {}
        for neuron in start_neurons:
            wavefront_results[neuron.id] = self._propagate(neuron)

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
                    traces.append(PathTrace(neurons=path_neurons))

            results.append(RecognitionResult(neuron=target_neuron, converging_paths=traces))

        # Strengthen traversed segments
        self._strengthen_traversed(wavefront_results)

        # Sort by confidence (most converging paths first)
        results.sort(key=lambda r: r.confidence, reverse=True)

        return results

    def _propagate(self, start: Neuron) -> dict[int, list[list[Neuron]]]:
        """BFS wavefront from a single neuron. Returns {reached_id: [[path_neurons]]}."""
        reached: dict[int, list[list[Neuron]]] = {}
        # Queue: (current_neuron, path_so_far)
        queue: list[tuple[Neuron, list[Neuron]]] = [(start, [start])]
        visited: set[int] = {start.id}

        depth = 0
        while queue and depth < self.max_depth:
            next_queue: list[tuple[Neuron, list[Neuron]]] = []
            for current, path in queue:
                segments = self.segment_repo.get_outgoing(current.id)
                for seg in segments:
                    if seg.target_id in visited:
                        continue
                    target = self.neuron_repo.get_by_id(seg.target_id)
                    if target is None:
                        continue
                    visited.add(target.id)
                    new_path = path + [target]
                    reached.setdefault(target.id, []).append(new_path)
                    next_queue.append((target, new_path))
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

    def trace(self, label: str) -> list[PathTrace]:
        """Trace all outgoing paths from a neuron."""
        neuron = self.neuron_repo.get_by_label(label.strip().lower())
        if neuron is None:
            return []

        reached = self._propagate(neuron)
        traces: list[PathTrace] = []
        for _target_id, path_lists in reached.items():
            for path_neurons in path_lists:
                traces.append(PathTrace(neurons=path_neurons))

        return traces
