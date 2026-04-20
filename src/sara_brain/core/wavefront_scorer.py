"""Wavefront-confluence scoring for multiple-choice questions.

For a question and N choices:
  1. Resolve question → seed list (compound-aware; each seed has power).
  2. Resolve each choice → seed list.
  3. Launch wavefronts from question seeds; record reached nodes and
     how many distinct question seeds reached each (question_power).
  4. For each choice, launch wavefronts; record choice_power per node.
  5. Score the choice by summing (question_power + choice_power) over
     nodes where both sides converged. Nodes that only one side reached
     do not contribute. This is witness-counting: the choice that
     shares the most evidence with the question wins.
  6. Include the seed-resolution bonus: if a question compound seed
     AND a choice compound seed both resolved to the SAME compound
     neuron, that neuron contributes its joint power directly — the
     "collapse at the compound" case Jennifer articulated.

Scoring is path-intersection (witness-counting), never a sum of raw
segment weights. Consistent with the score_by_path_not_sum rule.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .query_resolver import resolve_query, ResolvedSeed
from .recognizer import Recognizer
from ..storage.neuron_repo import NeuronRepo


@dataclass
class ChoiceScore:
    index: int
    text: str
    score: float
    convergence_count: int
    compound_hits: int
    seeds: list[ResolvedSeed]


def _reached_with_power(recognizer: Recognizer,
                        seeds: list[ResolvedSeed]
                        ) -> dict[int, float]:
    """Return {neuron_id: accumulated_power} for nodes reached by
    any seed's wavefront, plus the seeds themselves.

    Power at a node = sum of `seed.power` for each seed whose wavefront
    reached it. Each seed's own resolved neuron starts at `seed.power`.
    """
    power: dict[int, float] = defaultdict(float)
    for seed in seeds:
        n = recognizer.neuron_repo.resolve(seed.label, exact_only=True)
        if n is None:
            continue
        # Seed node itself carries the seed's power
        power[n.id] += seed.power
        # Propagation reach — bidirectional so subject-seeded wavefronts
        # can reach the properties Sara was taught about them (facts are
        # stored with a direction; a forward-only walk starves subjects).
        reached = recognizer._propagate(n, bidirectional=True)
        for target_id in reached:
            if target_id == n.id:
                continue
            power[target_id] += seed.power
    return dict(power)


def score_choices(question: str,
                  choices: list[str],
                  nlp,
                  recognizer: Recognizer,
                  neuron_repo: NeuronRepo,
                  ) -> list[ChoiceScore]:
    """Rank `choices` against `question` by wavefront confluence."""
    q_seeds = resolve_query(question, nlp, neuron_repo)
    q_power = _reached_with_power(recognizer, q_seeds)

    results: list[ChoiceScore] = []
    for i, choice in enumerate(choices):
        c_seeds = resolve_query(choice, nlp, neuron_repo)
        c_power = _reached_with_power(recognizer, c_seeds)

        # Confluence: nodes reached by BOTH sides.
        shared = set(q_power) & set(c_power)
        score = 0.0
        compound_hits = 0
        for nid in shared:
            score += q_power[nid] + c_power[nid]
        # Bonus: if question and choice share a compound-seed neuron
        # directly, count it explicitly. Already captured in `shared`
        # above, but tracked for reporting.
        q_compound_ids = {
            neuron_repo.resolve(s.label, exact_only=True).id
            for s in q_seeds if s.is_compound
            and neuron_repo.resolve(s.label, exact_only=True) is not None
        }
        c_compound_ids = {
            neuron_repo.resolve(s.label, exact_only=True).id
            for s in c_seeds if s.is_compound
            and neuron_repo.resolve(s.label, exact_only=True) is not None
        }
        compound_hits = len(q_compound_ids & c_compound_ids)

        results.append(ChoiceScore(
            index=i,
            text=choice,
            score=score,
            convergence_count=len(shared),
            compound_hits=compound_hits,
            seeds=c_seeds,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
