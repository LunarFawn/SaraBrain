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
  7. Math boost: if the question contains extractable numbers AND
     operation_tag segments are reachable from seeds, compute the
     result and boost the matching choice.

Scoring is path-intersection (witness-counting), never a sum of raw
segment weights. Consistent with the score_by_path_not_sum rule.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .math import MathCompute, NumberExtractor, tag_to_operation
from .query_resolver import resolve_query, resolve_query_nospacy, ResolvedSeed
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
                        seeds: list[ResolvedSeed],
                        echo: bool = False
                        ) -> dict[int, float]:
    """Return {neuron_id: accumulated_power} for nodes reached by
    any seed's wavefront, plus the seeds themselves.

    Each seed has a fixed evidence mass (`seed.power`). That mass is
    distributed across every node the wavefront reaches: a seed whose
    wavefront reaches R nodes contributes `seed.power / (R + 1)` to
    each reached node (and the seed itself counts as the +1).

    Hub Discrimination:
    To prevent generic hub nodes (e.g., "cell" with 100+ connections)
    from dominating the score, the power contributed to each node is
    scaled inversely by that node's connectivity. A specific node
    shared by both question and choice is a stronger witness than
    a shared generic hub.

    Echo Mode:
    When echo=True, uses iterative spreading activation (propagate_echo)
    to find deep connections.
    """
    if echo:
        from .short_term import ShortTerm
        import time
        st = ShortTerm(event_id=f"score-{time.time()}", event_type="score")
        # Using the new true Backwave propagation instead of the noisy echo
        if hasattr(recognizer, "propagate_backwave"):
            recognizer.propagate_backwave([s.label for s in seeds], st, exact_only=True)
        else:
            recognizer.propagate_echo([s.label for s in seeds], st, exact_only=True)
        
        # We need to convert the ShortTerm weights into the power dict format.
        # ShortTerm already does linear accumulation. We apply hub penalty here.
        power: dict[int, float] = defaultdict(float)
        for nid, weight in st.convergence_map.items():
            out_count = len(recognizer.segment_repo.get_outgoing(nid))
            in_count = len(recognizer.segment_repo.get_incoming(nid))
            connectivity = out_count + in_count
            
            # TODO: Empirical tests show Hub Penalty actively hurts accuracy on Pure Wavefront 
            # (drops from 28% to 24%). Consider removing this in the future.
            # Linear hub penalty
            h_weight = 1.0 / (connectivity + 1)
            power[nid] = weight * h_weight
        return dict(power)

    power: dict[int, float] = defaultdict(float)
    for seed in seeds:
        n = recognizer.neuron_repo.resolve(seed.label, exact_only=True)
        if n is None:
            continue
        reached = recognizer._propagate(n, bidirectional=True)
        targets = [tid for tid in reached if tid != n.id]
        
        # Calculate connectivity-weighted power for each node.
        nodes_to_power = [n.id] + targets
        total_witnesses = len(nodes_to_power)
        base_power_per_witness = seed.power / total_witnesses
        
        for nid in nodes_to_power:
            # Connectivity = total segments (incoming + outgoing)
            out_count = len(recognizer.segment_repo.get_outgoing(nid))
            in_count = len(recognizer.segment_repo.get_incoming(nid))
            connectivity = out_count + in_count
            
            # TODO: Empirical tests show Hub Penalty actively hurts accuracy on Pure Wavefront 
            # (drops from 28% to 24%). Consider removing this in the future.
            # linear scaling: Weight = 1.0 / (connectivity + 1)
            weight = 1.0 / (connectivity + 1)
            power[nid] += base_power_per_witness * weight
            
    return dict(power)


_NEGATION_CUES = ("NOT", "EXCEPT", "LEAST", " FALSE", "UNLESS")


def _compute_math_answers(question: str, recognizer: Recognizer,
                          q_seeds: list[ResolvedSeed]) -> list[float]:
    """If the question has numbers and the brain has operation_tags on
    reachable segments, compute possible numeric answers.

    Returns a list of computed values (may be empty).
    """
    extractor = NumberExtractor()
    numbers = extractor.extract(question)
    if not numbers:
        return []

    # Find operation_tags on segments reachable from question seeds.
    compute = MathCompute()
    results: list[float] = []
    seg_repo = recognizer.segment_repo

    for seed in q_seeds:
        n = recognizer.neuron_repo.resolve(seed.label, exact_only=True)
        if n is None:
            continue
        # Check segments attached to this neuron for operation_tags.
        for seg in seg_repo.get_outgoing(n.id) + seg_repo.get_incoming(n.id):
            tag = getattr(seg, "operation_tag", None)
            if not tag:
                continue
            op = tag_to_operation(tag)
            if op is None:
                continue
            # Apply the operation to each extracted number.
            for _label, value in numbers.items():
                try:
                    result = compute.apply(op, value)
                    results.append(result)
                except (ValueError, ZeroDivisionError):
                    pass
    return results


def _math_boost(choices: list[str], computed: list[float]) -> dict[int, float]:
    """Return {choice_index: boost} for choices whose text matches a
    computed numeric answer."""
    if not computed:
        return {}
    boosts: dict[int, float] = {}
    for i, choice in enumerate(choices):
        # Extract numbers from the choice text.
        choice_nums = re.findall(r"-?\d+(?:\.\d+)?", choice)
        for cn in choice_nums:
            try:
                cv = float(cn)
            except ValueError:
                continue
            for result in computed:
                # Tolerance for float comparison.
                if abs(cv - result) < 0.01:
                    # Large boost — math is definitive when it matches.
                    boosts[i] = boosts.get(i, 0.0) + 100.0
    return boosts


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
                  echo: bool = False,
                  dampened: bool = False,
                  ) -> list[ChoiceScore]:
    """Rank `choices` against `question` by wavefront confluence."""
    if nlp is not None:
        q_seeds = resolve_query(question, nlp, neuron_repo)
    else:
        q_seeds = resolve_query_nospacy(question, neuron_repo)

    q_power = _reached_with_power(recognizer, q_seeds, echo=echo)

    # Math boost: compute numeric answers if operation_tags exist.
    computed = _compute_math_answers(question, recognizer, q_seeds)
    math_boosts = _math_boost(choices, computed)

    results: list[ChoiceScore] = []
    import math
    for i, choice in enumerate(choices):
        if nlp is not None:
            c_seeds = resolve_query(choice, nlp, neuron_repo)
        else:
            c_seeds = resolve_query_nospacy(choice, neuron_repo)
        c_power = _reached_with_power(recognizer, c_seeds, echo=echo)

        # Confluence: nodes reached by BOTH sides.
        shared = set(q_power) & set(c_power)
        score = 0.0

        for nid in shared:
            if dampened:
                # Non-linear log1p dampening (experimental for high-volume echo)
                score += math.log1p(q_power[nid]) + math.log1p(c_power[nid])
            else:
                # Linear baseline (best performing for MC precision)
                score += q_power[nid] + c_power[nid]

        # Compound-match bonus.
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
        
        for nid in compound_matches:
            if dampened:
                score += math.log1p(q_power.get(nid, 0.0)) + math.log1p(c_power.get(nid, 0.0))
            else:
                score += q_power.get(nid, 0.0) + c_power.get(nid, 0.0)

        results.append(ChoiceScore(
            index=i,
            text=choice,
            score=score + math_boosts.get(i, 0.0),
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
