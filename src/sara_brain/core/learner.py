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
    ) -> None:
        self.parser = parser
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.path_repo = path_repo

    def learn(self, text: str) -> LearnResult | None:
        """Parse a statement and build a 3-neuron path chain.

        Chain: property → relation → concept
        E.g.: red → fruit_color → apple
        """
        parsed = self.parser.parse(text)
        if parsed is None:
            return None
        return self._build_chain(parsed)

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
        seg1, created = self.segment_repo.get_or_create(
            prop_neuron.id, relation_neuron.id, parsed.relation
        )
        if created:
            segments_created += 1
        if refute:
            self.segment_repo.weaken(seg1)
        elif not created:
            self.segment_repo.strengthen(seg1)

        seg2, created = self.segment_repo.get_or_create(
            relation_neuron.id, concept_neuron.id, "describes"
        )
        if created:
            segments_created += 1
        if refute:
            self.segment_repo.weaken(seg2)
        elif not created:
            self.segment_repo.strengthen(seg2)

        # 5. Record the path. Refutations get a [refuted] prefix on
        # source_text so provenance is preserved.
        source_text = (
            f"[refuted] {parsed.original}" if refute else parsed.original
        )
        path = Path(
            id=None,
            origin_id=prop_neuron.id,
            terminus_id=concept_neuron.id,
            source_text=source_text,
        )
        path = self.path_repo.create(path)

        # 6. Record path steps
        self.path_repo.add_step(PathStep(id=None, path_id=path.id, step_order=0, segment_id=seg1.id))
        self.path_repo.add_step(PathStep(id=None, path_id=path.id, step_order=1, segment_id=seg2.id))

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
