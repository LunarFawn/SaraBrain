"""Brain: main orchestrator and public API."""

from __future__ import annotations

from ..models.result import RecognitionResult, PathTrace
from ..parsing.statement_parser import StatementParser
from ..parsing.taxonomy import Taxonomy
from ..storage.database import Database
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo
from .learner import Learner, LearnResult
from .recognizer import Recognizer
from .similarity import SimilarityAnalyzer, SimilarityLink


class Brain:
    """The main entry point for Sara Brain.

    Every mutation writes to SQLite immediately. On restart, full state is recovered.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db = Database(db_path)
        self.conn = self.db.conn

        # Repos
        self.neuron_repo = NeuronRepo(self.conn)
        self.segment_repo = SegmentRepo(self.conn)
        self.path_repo = PathRepo(self.conn)

        # Taxonomy & parser
        self.taxonomy = Taxonomy()
        self.parser = StatementParser(self.taxonomy)

        # Core algorithms
        self.learner = Learner(self.parser, self.neuron_repo, self.segment_repo, self.path_repo)
        self.recognizer = Recognizer(self.neuron_repo, self.segment_repo)
        self.similarity = SimilarityAnalyzer(self.neuron_repo, self.segment_repo, self.conn)

    def teach(self, statement: str) -> LearnResult | None:
        """Teach a fact. Returns None if unparseable."""
        result = self.learner.learn(statement)
        if result is not None:
            self.conn.commit()
        return result

    def recognize(self, inputs: str) -> list[RecognitionResult]:
        """Recognize from comma-separated input labels."""
        labels = [l.strip() for l in inputs.split(",") if l.strip()]
        results = self.recognizer.recognize(labels)
        self.conn.commit()
        return results

    def trace(self, label: str) -> list[PathTrace]:
        """Trace all outgoing paths from a neuron."""
        return self.recognizer.trace(label)

    def why(self, label: str) -> list[PathTrace]:
        """Show all paths that lead TO a neuron (reverse lookup)."""
        neuron = self.neuron_repo.get_by_label(label.strip().lower())
        if neuron is None:
            return []

        paths = self.path_repo.get_paths_to(neuron.id)
        traces: list[PathTrace] = []
        for p in paths:
            steps = self.path_repo.get_steps(p.id)
            neurons = []
            # Walk the segments to reconstruct the neuron chain
            for step in steps:
                seg = self.segment_repo.get_by_id(step.segment_id)
                if seg is None:
                    continue
                if not neurons:
                    source = self.neuron_repo.get_by_id(seg.source_id)
                    if source:
                        neurons.append(source)
                target = self.neuron_repo.get_by_id(seg.target_id)
                if target:
                    neurons.append(target)
            traces.append(PathTrace(neurons=neurons, source_text=p.source_text))

        return traces

    def analyze_similarity(self) -> list[SimilarityLink]:
        """Scan for path similarities across all property neurons."""
        return self.similarity.analyze()

    def get_similar(self, label: str) -> list[SimilarityLink]:
        """Get neurons that share downstream paths with the given neuron."""
        return self.similarity.get_similar(label)

    def stats(self) -> dict:
        """Return brain statistics."""
        neurons = self.neuron_repo.count()
        segments = self.segment_repo.count()
        paths = self.path_repo.count()

        strongest = None
        all_segs = self.segment_repo.list_all()
        if all_segs:
            s = max(all_segs, key=lambda s: s.strength)
            src = self.neuron_repo.get_by_id(s.source_id)
            tgt = self.neuron_repo.get_by_id(s.target_id)
            if src and tgt:
                strongest = f"{src.label} → {tgt.label} (strength: {s.strength:.2f})"

        return {
            "neurons": neurons,
            "segments": segments,
            "paths": paths,
            "strongest_segment": strongest,
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Brain:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
