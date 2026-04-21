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

    Each seed has a fixed evidence mass (`seed.power`). That mass is
    distributed across every node the wavefront reaches: a seed whose
    wavefront reaches R nodes contributes `seed.power / (R + 1)` to
    each reached node (and the seed itself counts as the +1).
    """
    power: dict[int, float] = defaultdict(float)
    for seed in seeds:
        n = recognizer.neuron_repo.resolve(seed.label, exact_only=True)
        if n is None:
            continue
        reached = recognizer._propagate(n, bidirectional=True)
        targets = [tid for tid in reached if tid != n.id]
        total_witnesses = 1 + len(targets)
        per_witness = seed.power / total_witnesses
        power[n.id] += per_witness
        for target_id in targets:
            power[target_id] += per_witness
    return dict(power)


_NEGATION_CUES = ("NOT", "EXCEPT", "LEAST", " FALSE", "UNLESS")


def _is_negation_question(question: str) -> bool:
    """Detect whether the question is asking for the OUTLIER choice —
    the one that DOES NOT match the category the other choices match.
    Common cues: 'NOT', 'EXCEPT', 'LEAST', 'FALSE', 'UNLESS'.

    Case-sensitive on uppercase forms to avoid false positives from
    words like 'note' or 'except' in normal prose. Textbook MC questions
    use the uppercase convention for emphasis (e.g., 'EXCEPT:').
    """
    return any(cue in question for cue in _NEGATION_CUES)


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
        for nid in shared:
            score += q_power[nid] + c_power[nid]

        # Compound-match bonus. When a question compound seed AND a
        # choice compound seed resolve to the SAME compound neuron,
        # both sides independently identified the same specific concept.
        # That's stronger path evidence than hub-atom convergence and
        # should weight more than a diluted atom match.
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
        compound_matches = q_compound_ids & c_compound_ids
        compound_hits = len(compound_matches)
        # Each compound match adds the compound's power product, not a
        # flat bonus — keeps scoring grounded in the graph's own edge
        # mass rather than an invented constant.
        for nid in compound_matches:
            score += q_power.get(nid, 0.0) + c_power.get(nid, 0.0)

        results.append(ChoiceScore(
            index=i,
            text=choice,
            score=score,
            convergence_count=len(shared),
            compound_hits=compound_hits,
            seeds=c_seeds,
        ))

    # Negation-aware ranking: if the question asks for the choice that
    # does NOT belong (NOT, EXCEPT, LEAST, FALSE, UNLESS), the correct
    # answer is the OUTLIER with the LEAST convergence to the question's
    # category. Path evidence logic unchanged — just read inverted.
    negated = _is_negation_question(question)
    results.sort(key=lambda r: r.score, reverse=not negated)
    return results


def pick_choice(ranked: list[ChoiceScore], question: str
                ) -> tuple[int | None, str]:
    """Given a ranked list from `score_choices`, return (pick_idx, reason).

    - For positive questions: pick ranked[0] if its score > 0; else abstain.
    - For negation questions: pick ranked[0] (the outlier-low) if the
      choice set has any variance. Abstain when all choices score
      identically.
    - Tie at the winning position → tie (no pick).
    """
    if not ranked:
        return None, "no_scores"
    negated = _is_negation_question(question)
    top = ranked[0]

    if negated:
        scores = [r.score for r in ranked]
        if max(scores) == min(scores):
            return None, "abstain_all_equal"
        low_count = sum(1 for s in scores if s == top.score)
        if low_count > 1:
            return None, "tie"
        return top.index, "negation_outlier"

    # Positive question
    if top.score <= 0:
        return None, "abstain_zero"
    tied = [r for r in ranked if r.score == top.score]
    if len(tied) > 1:
        return None, "tie"
    return top.index, "top_score"
