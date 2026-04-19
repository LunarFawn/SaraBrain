"""Learning: parse teachings, build path chains, persist to SQLite."""

from __future__ import annotations

from dataclasses import dataclass

from ..models.neuron import NeuronType
from ..models.path import Path, PathStep
from ..models.segment import Segment
from ..parsing.statement_parser import ParsedStatement, StatementParser
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo


@dataclass
class LearnResult:
    path_label: str  # e.g., "red → fruit_color → apple"
    segments_created: int
    neurons_created: int
    path_id: int


class Learner:
    def __init__(
        self,
        parser: StatementParser,
        neuron_repo: NeuronRepo,
        segment_repo: SegmentRepo,
        path_repo: PathRepo,
        segment_source_repo=None,
    ) -> None:
        self.parser = parser
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.path_repo = path_repo
        # Optional — present when multi-source provenance is enabled
        self.segment_source_repo = segment_source_repo

    def learn(self, text: str, initial_strength: float | None = None,
              source_label: str | None = None,
              apply_filter: bool = True) -> LearnResult | None:
        """Parse a statement and build a 3-neuron path chain.

        Chain: property → relation → concept
        E.g.: red → fruit_color → apple

        Args:
            initial_strength: when set, newly-created segments have
                their strength overridden to this value. Used by
                tentative teaching (0.4 is below the query visibility
                floor of 0.5). None means use the default (1.0).
            source_label: provenance tag (URL, filename, "user"). When
                provided and SegmentSourceRepo is available, segments
                are linked to this source. Same-source re-teach becomes
                a no-op for segment strengthening — two DIFFERENT
                sources are required for the two-witness upgrade.
            apply_filter: run the pollution filter before parsing.
                Default True. Set False only for trusted teachers
                (the user via CLI) where filtering is unwanted.
        """
        if apply_filter:
            from .filters import is_polluting_statement
            rejected, _reason = is_polluting_statement(text)
            if rejected:
                return None

        parsed = self.parser.parse(text)
        if parsed is None:
            return None
        # First-class negation: when the parser detects a negated
        # statement (e.g., "X is not Y", "X does not Y"), flow that
        # through as a refutation of the positive relation — same
        # storage path as unlearn()/brain.refute(). Sara now
        # distinguishes "X is Y" from "X is not Y" at the graph level.
        return self._build_chain(
            parsed,
            initial_strength=initial_strength,
            source_label=source_label,
            refute=parsed.negated,
        )

    def unlearn(self, text: str) -> LearnResult | None:
        """Refute a statement. Mirrors learn() but weakens segments
        instead of strengthening them.

        Sara never deletes — she marks the claim as known-to-be-false
        by incrementing refutations on the segments. Strength can go
        negative when refutations exceed validations. The path itself
        is preserved so Sara remembers what was once claimed.
        """
        parsed = self.parser.parse(text)
        if parsed is None:
            return None
        return self._build_chain(parsed, refute=True)

    def _build_chain(
        self,
        parsed: ParsedStatement,
        refute: bool = False,
        initial_strength: float | None = None,
        source_label: str | None = None,
    ) -> LearnResult:
        neurons_created = 0
        segments_created = 0

        # 1. Get or create the property neuron (e.g., "red")
        prop_neuron, created = self.neuron_repo.get_or_create(
            parsed.obj, NeuronType.PROPERTY
        )
        if created:
            neurons_created += 1

        # 2. Get or create the concept neuron (e.g., "apple")
        concept_neuron, created = self.neuron_repo.get_or_create(
            parsed.subject, NeuronType.CONCEPT
        )
        if created:
            neurons_created += 1

        # 3. Get or create the intermediate relation neuron
        relation_label = self.parser.taxonomy.relation_label(
            parsed.subject, parsed.obj
        )
        relation_neuron, created = self.neuron_repo.get_or_create(
            relation_label, NeuronType.RELATION
        )
        if created:
            neurons_created += 1

        # 4. Build segments: property → relation → concept
        # If refute=True, weaken existing or create-then-weaken so the
        # segments record the refutation in their counters.
        #
        # Source-gated strengthening: when source_label is provided, only
        # strengthen an existing segment if the source is a NEW witness.
        # Same source re-teaching the same fact is a no-op. Two distinct
        # sources is the confirmation signal.
        seg1, created = self.segment_repo.get_or_create(
            prop_neuron.id, relation_neuron.id, parsed.relation
        )
        if created:
            segments_created += 1
            if initial_strength is not None:
                self._set_initial_strength(seg1, initial_strength)
        is_new_witness_1 = self._record_source(seg1, source_label)
        if refute:
            self.segment_repo.weaken(seg1)
        elif not created:
            # Strengthen only if this is a new witness (or no source
            # tracking is in use — legacy/user teaching path).
            if source_label is None or is_new_witness_1:
                self.segment_repo.strengthen(seg1)

        seg2, created = self.segment_repo.get_or_create(
            relation_neuron.id, concept_neuron.id, "describes"
        )
        if created:
            segments_created += 1
            if initial_strength is not None:
                self._set_initial_strength(seg2, initial_strength)
        is_new_witness_2 = self._record_source(seg2, source_label)
        if refute:
            self.segment_repo.weaken(seg2)
        elif not created:
            if source_label is None or is_new_witness_2:
                self.segment_repo.strengthen(seg2)

        # 4b. Attach arithmetic operation if the parser detected one.
        # Operation lives on the relation segment (property → relation)
        # because the operation describes HOW the property transforms
        # the concept. Lazy import to avoid circular dependency.
        if getattr(parsed, "operation", None) is not None:
            try:
                from .math import MathLinker
                MathLinker(self.segment_repo).link(seg1.id, parsed.operation)
            except Exception:
                # Never let math linking break a teaching — if the link
                # fails, the fact is still stored in the graph without
                # the operation tag.
                pass

        # 5. Record the path. Source_text is NEVER prefixed or mutated.
        # Refutation state is tracked in the graph via CLEANUP primitives,
        # not via string hacks on source_text.
        path = Path(
            id=None,
            origin_id=prop_neuron.id,
            terminus_id=concept_neuron.id,
            source_text=parsed.original,
        )
        path = self.path_repo.create(path)

        # 6. Record path steps
        self.path_repo.add_step(PathStep(id=None, path_id=path.id, step_order=0, segment_id=seg1.id))
        self.path_repo.add_step(PathStep(id=None, path_id=path.id, step_order=1, segment_id=seg2.id))

        # 7. If refuting, create a concept-specific refutation path
        # grounding in the 'refuted' CLEANUP primitive:
        #   concept → {concept}_refuted → refuted
        # Same pattern as cleanup paths and fact paths. No shared hub.
        if refute:
            refuted_primitive, _ = self.neuron_repo.get_or_create(
                "refuted", NeuronType.PROPERTY
            )
            refute_relation_label = f"{concept_neuron.label}_refuted"
            refute_relation, _ = self.neuron_repo.get_or_create(
                refute_relation_label, NeuronType.RELATION
            )
            seg_r1, _ = self.segment_repo.get_or_create(
                concept_neuron.id, refute_relation.id, "refutation_of"
            )
            seg_r2, _ = self.segment_repo.get_or_create(
                refute_relation.id, refuted_primitive.id, "refutation_status"
            )
            refute_path = Path(
                id=None,
                origin_id=concept_neuron.id,
                terminus_id=refuted_primitive.id,
                source_text=parsed.original,
            )
            refute_path = self.path_repo.create(refute_path)
            self.path_repo.add_step(
                PathStep(id=None, path_id=refute_path.id, step_order=0, segment_id=seg_r1.id)
            )
            self.path_repo.add_step(
                PathStep(id=None, path_id=refute_path.id, step_order=1, segment_id=seg_r2.id)
            )

        # 7. Sub-concept linking: decompose multi-word labels into
        #    individual word neurons with part_of segments so wavefronts
        #    from any single word can reach the compound concept.
        for neuron in (concept_neuron, prop_neuron):
            nc, sc = self._link_sub_concepts(neuron)
            neurons_created += nc
            segments_created += sc

        path_label = f"{prop_neuron.label} → {relation_neuron.label} → {concept_neuron.label}"

        return LearnResult(
            path_label=path_label,
            segments_created=segments_created,
            neurons_created=neurons_created,
            path_id=path.id,
        )

    def _set_initial_strength(self, segment, strength: float) -> None:
        """Override a newly-created segment's strength.

        Used by tentative teaching to write segments below the query
        visibility floor. Reuses the same UPDATE pattern as
        brain.describe_association.
        """
        self.segment_repo.conn.execute(
            f"UPDATE {self.segment_repo._t} SET strength = ? WHERE id = ?",
            (strength, segment.id),
        )
        segment.strength = strength

    def _record_source(self, segment, source_label: str | None) -> bool:
        """Record a source for a segment. Returns True if it's a new witness.

        No-op when source tracking isn't configured or source_label is
        None. When enabled, UNIQUE(segment_id, source_label) in the
        segment_sources table makes same-source re-teach a no-op —
        returns False, which the caller uses to skip strengthen().
        """
        if self.segment_source_repo is None or not source_label:
            return False
        return self.segment_source_repo.add(segment.id, source_label)

    def _link_sub_concepts(self, neuron) -> tuple[int, int]:
        """If neuron label is multi-word, create word → compound segments.

        Returns (neurons_created, segments_created).
        """
        words = neuron.label.split()
        if len(words) < 2:
            return 0, 0

        neurons_created = 0
        segments_created = 0
        for word in words:
            word = word.strip()
            if not word:
                continue
            word_neuron, created = self.neuron_repo.get_or_create(
                word, NeuronType.PROPERTY
            )
            if created:
                neurons_created += 1
            seg, created = self.segment_repo.get_or_create(
                word_neuron.id, neuron.id, "part_of"
            )
            if created:
                segments_created += 1
            else:
                self.segment_repo.strengthen(seg)
        return neurons_created, segments_created
