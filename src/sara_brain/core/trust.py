"""Trust manager: handles knowledge trust levels, conflict detection, and doctor annotations.

Nothing is ever erased. Doctors annotate paths — they cannot delete them.
Annotations are themselves paths in the knowledge graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..models.neuron import NeuronType
from ..models.path import Path, PathStep
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo
from ..storage.account_repo import AccountRepo


VALID_TRUST_STATUSES = ("observed", "taught", "verified", "contested")
VALID_ANNOTATIONS = ("misunderstood", "confabulation", "accurate", "disputed")


@dataclass
class Annotation:
    """A doctor's assessment of a path. Stored as a path in the graph."""
    path_id: int        # the path being annotated
    label: str          # misunderstood, confabulation, accurate, disputed
    doctor_account_id: int
    annotation_path_id: int  # the path that records this annotation
    created_at: float


@dataclass
class ContestedPair:
    """Two paths that conflict with each other."""
    path_a: Path
    path_b: Path
    shared_concept: str


class TrustManager:
    """Manages trust status, repetition, conflict detection, and annotations."""

    def __init__(self, neuron_repo: NeuronRepo, segment_repo: SegmentRepo,
                 path_repo: PathRepo, account_repo: AccountRepo) -> None:
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.path_repo = path_repo
        self.account_repo = account_repo

    def annotate(self, path_id: int, doctor_account_id: int,
                 label: str) -> Annotation:
        """Doctor annotates a path. Creates an annotation path in the graph.

        NEVER deletes the original path. The annotation is a new path:
          source_text -> doctor_assessed -> <label>

        Valid labels: misunderstood, confabulation, accurate, disputed
        """
        if label not in VALID_ANNOTATIONS:
            raise ValueError(
                f"Invalid annotation: {label}. Must be one of {VALID_ANNOTATIONS}"
            )

        # Verify doctor role
        doctor = self.account_repo.get_by_id(doctor_account_id)
        if doctor is None or doctor.role != "doctor":
            raise PermissionError("Only doctors can annotate paths")

        # Get the path being annotated
        original_path = self.path_repo.get_by_id(path_id)
        if original_path is None:
            raise ValueError(f"Path {path_id} not found")

        # Create annotation neurons
        # The annotation label neuron (e.g., "misunderstood")
        label_neuron, _ = self.neuron_repo.get_or_create(
            label, NeuronType.PROPERTY
        )

        # The doctor_assessed relation neuron
        assessed_neuron, _ = self.neuron_repo.get_or_create(
            "doctor_assessed", NeuronType.RELATION
        )

        # Get the origin neuron of the original path (what was observed)
        origin_neuron = self.neuron_repo.get_by_id(original_path.origin_id)
        if origin_neuron is None:
            raise ValueError("Original path origin neuron not found")

        # Create segments: origin -> doctor_assessed -> label
        seg1, _ = self.segment_repo.get_or_create(
            origin_neuron.id, assessed_neuron.id, "doctor_assessed"
        )
        seg2, _ = self.segment_repo.get_or_create(
            assessed_neuron.id, label_neuron.id, "assessment_is"
        )

        # Create the annotation path
        now = time.time()
        annotation_source = (
            f"[annotation] path:{path_id} assessed as '{label}' "
            f"by account:{doctor_account_id}"
        )
        annotation_path = Path(
            id=None,
            origin_id=origin_neuron.id,
            terminus_id=label_neuron.id,
            source_text=annotation_source,
            account_id=doctor_account_id,
            trust_status="verified",
        )
        annotation_path = self.path_repo.create(annotation_path)
        self.path_repo.add_step(
            PathStep(id=None, path_id=annotation_path.id, step_order=0,
                     segment_id=seg1.id)
        )
        self.path_repo.add_step(
            PathStep(id=None, path_id=annotation_path.id, step_order=1,
                     segment_id=seg2.id)
        )

        # If annotated as 'accurate', promote the original path
        if label == "accurate":
            self.path_repo.update_trust_status(path_id, "verified")

        return Annotation(
            path_id=path_id,
            label=label,
            doctor_account_id=doctor_account_id,
            annotation_path_id=annotation_path.id,
            created_at=now,
        )

    def promote(self, path_id: int, doctor_account_id: int) -> Annotation:
        """Shortcut: annotate as 'accurate' and set trust_status to 'verified'."""
        return self.annotate(path_id, doctor_account_id, "accurate")

    def get_contested(self) -> list[Path]:
        """Get all paths with trust_status='contested'."""
        return self.path_repo.get_by_trust_status("contested")

    def get_observed(self) -> list[Path]:
        """Get all patient observations awaiting review."""
        return self.path_repo.get_by_trust_status("observed")

    def get_repeated_observations(self, min_count: int = 2) -> list[Path]:
        """Get patient observations that have been repeated multiple times.

        High repetition = patient believes this strongly. Needs investigation.
        """
        observed = self.path_repo.get_by_trust_status("observed")
        return [p for p in observed if p.repetition_count >= min_count]

    def get_annotations(self, path_id: int) -> list[Path]:
        """Get all annotation paths for a given original path.

        Annotations are paths whose source_text starts with '[annotation] path:<id>'.
        """
        all_paths = self.path_repo.list_all()
        prefix = f"[annotation] path:{path_id}"
        return [p for p in all_paths
                if p.source_text and p.source_text.startswith(prefix)]

    def check_conflict(self, new_path: Path, brain) -> bool:
        """Check if a new path conflicts with existing paths about the same concept.

        If conflict detected, mark both as 'contested'.
        Returns True if a conflict was found.
        """
        if new_path.terminus_id is None:
            return False

        # Get all existing paths to the same concept
        existing = self.path_repo.get_paths_to(new_path.terminus_id)

        for existing_path in existing:
            # Skip self, skip annotation paths, skip paths from same account
            if existing_path.id == new_path.id:
                continue
            if existing_path.source_text and existing_path.source_text.startswith("["):
                continue
            if (existing_path.account_id is not None and
                    existing_path.account_id == new_path.account_id):
                continue

            # Check if the paths carry contradictory information
            # (different origins for the same relation to the same concept)
            if (existing_path.origin_id != new_path.origin_id and
                    existing_path.account_id != new_path.account_id):
                # Different people said different things about the same concept
                # via same relation — potential conflict
                new_origin = self.neuron_repo.get_by_id(new_path.origin_id)
                existing_origin = self.neuron_repo.get_by_id(existing_path.origin_id)

                if new_origin and existing_origin:
                    # Check if one is a negation or contradiction of the other
                    # Simple heuristic: if source texts suggest contradiction
                    if self._texts_contradict(new_path.source_text,
                                              existing_path.source_text):
                        self.path_repo.update_trust_status(new_path.id, "contested")
                        self.path_repo.update_trust_status(existing_path.id, "contested")
                        return True

        return False

    def check_repetition(self, new_path: Path) -> int | None:
        """Check if this path duplicates an existing observed path.

        If a patient says the same thing again, increment the repetition count
        on the existing path instead of creating a duplicate.

        Returns the new repetition count, or None if no duplicate found.
        """
        if new_path.account_id is None:
            return None

        existing = self.path_repo.get_by_account(new_path.account_id)
        for ep in existing:
            if ep.id == new_path.id:
                continue
            # Same origin and terminus = same fact
            if (ep.origin_id == new_path.origin_id and
                    ep.terminus_id == new_path.terminus_id):
                return self.path_repo.increment_repetition(ep.id)
        return None

    @staticmethod
    def _texts_contradict(text_a: str | None, text_b: str | None) -> bool:
        """Simple heuristic to detect contradictions between source texts.

        Looks for negation patterns. This is intentionally simple for PoC —
        the LLM sensory layer can provide more sophisticated analysis.
        """
        if not text_a or not text_b:
            return False

        a = text_a.lower()
        b = text_b.lower()

        negations = ("not ", "no ", "never ", "isn't ", "aren't ",
                     "don't ", "doesn't ", "wasn't ", "weren't ")

        # If one text negates a claim in the other
        for neg in negations:
            if neg in a and neg not in b:
                # Check if the core words overlap
                a_words = set(a.replace(neg, "").split())
                b_words = set(b.split())
                if len(a_words & b_words) >= 2:
                    return True
            if neg in b and neg not in a:
                a_words = set(a.split())
                b_words = set(b.replace(neg, "").split())
                if len(a_words & b_words) >= 2:
                    return True

        return False
