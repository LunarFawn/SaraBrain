"""Reverse-direction learner — writes facts to a parallel DB with chain
direction flipped from property→relation→concept to subject→relation→object.

Kept deliberately as a separate DB from the primary brain so existing
consumers of the primary direction (path_repo.get_paths_to, query_association,
recognize, why, refutation walks) keep working unchanged. The reverse DB
is for subject-seeded wavefront propagation — queries that start from
the subject and want to reach the facts taught about it.
"""
from __future__ import annotations

from ..models.neuron import NeuronType
from ..models.path import Path, PathStep
from ..parsing.statement_parser import ParsedStatement, StatementParser
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo


class ReverseLearner:
    """Parallel learner that mirrors `Learner` but stores facts with
    reversed chain direction: subject → relation → object.

    Takes its own repos bound to a separate sqlite connection so writes
    go to a distinct DB (e.g. brain.db.reverse).
    """

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

    def learn(self, text: str) -> bool:
        """Parse the statement and write it in reverse direction.
        Returns True if a path was written."""
        parsed = self.parser.parse(text)
        if parsed is None or parsed.verb_unknown:
            return False
        self._build_reverse_chain(parsed)
        return True

    def _build_reverse_chain(self, parsed: ParsedStatement) -> None:
        # Same three neurons as the primary learner, just different chain order.
        prop_neuron, _ = self.neuron_repo.get_or_create(
            parsed.obj, NeuronType.PROPERTY,
        )
        concept_neuron, _ = self.neuron_repo.get_or_create(
            parsed.subject, NeuronType.CONCEPT,
        )
        relation_label = self.parser.taxonomy.relation_label(
            parsed.subject, parsed.obj,
        )
        relation_neuron, _ = self.neuron_repo.get_or_create(
            relation_label, NeuronType.RELATION,
        )

        # Reversed chain: subject → relation → object
        seg1, _ = self.segment_repo.get_or_create(
            concept_neuron.id, relation_neuron.id, parsed.relation,
        )
        seg2, _ = self.segment_repo.get_or_create(
            relation_neuron.id, prop_neuron.id, "describes",
        )

        path = self.path_repo.create(Path(
            id=None,
            origin_id=concept_neuron.id,
            terminus_id=prop_neuron.id,
            source_text=parsed.original,
        ))
        self.path_repo.add_step(PathStep(
            id=None, path_id=path.id, step_order=0, segment_id=seg1.id,
        ))
        self.path_repo.add_step(PathStep(
            id=None, path_id=path.id, step_order=1, segment_id=seg2.id,
        ))
