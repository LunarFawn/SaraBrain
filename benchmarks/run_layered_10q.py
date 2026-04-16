#!/usr/bin/env python3
"""Quick layered-brain test on 10 MMLU biology questions.

Three layers (separate DBs), queried in sequence:
  1. Vocabulary — resolve question words to definitions
  2. Science — map definitions to intermediate concepts
  3. Biology — map concepts to domain-specific knowledge

For each choice, score by how many keywords from ALL three layers'
convergence appear in the choice text.
"""

from __future__ import annotations

import json
import re

from sara_brain.core.brain import Brain


def _stem(word: str) -> list[str]:
    """Produce root forms from a word. Returns the word + any stems."""
    forms = [word]
    # Comparative: faster → fast, taller → tall
    if word.endswith("er") and len(word) > 4:
        forms.append(word[:-2])
        # doubled consonant: bigger → big
        if len(word) > 5 and word[-3] == word[-4]:
            forms.append(word[:-3])
    # Superlative: tallest → tall, fastest → fast
    if word.endswith("est") and len(word) > 5:
        forms.append(word[:-3])
    # -ly adverb: rapidly → rapid
    if word.endswith("ly") and len(word) > 4:
        forms.append(word[:-2])
    # -ing: producing → produce
    if word.endswith("ing") and len(word) > 5:
        forms.append(word[:-3])
        forms.append(word[:-3] + "e")
    # Plurals: cells → cell, structures → structure
    if word.endswith("es") and len(word) > 4:
        forms.append(word[:-2])
        forms.append(word[:-1])
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        forms.append(word[:-1])
    return forms


def extract_words(text: str) -> list[str]:
    """Extract content words >= 4 chars with stemmed variants."""
    stops = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "them", "than", "then", "more", "most", "also", "only",
        "each", "both", "some", "many", "such", "very", "just", "into",
        "your", "will", "would", "could", "should", "which", "what",
        "when", "where", "about", "above", "below", "these", "those",
        "able", "result", "following", "example", "best", "likely",
        "occurs", "along", "pass",
    }
    raw = re.findall(r"[a-z][a-z'-]+", text.lower())
    words = []
    seen = set()
    for w in raw:
        if len(w) < 4 or w in stops:
            continue
        for form in _stem(w):
            if len(form) >= 3 and form not in seen and form not in stops:
                seen.add(form)
                words.append(form)
    return words


def echo_keywords_from_layer(brain, seed_labels: list[str],
                             min_strength: float = 0.5,
                             max_rounds: int = 2) -> dict[str, float]:
    """Echo-propagate seeds into a ShortTerm, return reached labels as keywords."""
    keywords: dict[str, float] = {}
    with brain.short_term("echo_query") as st:
        brain.propagate_echo(
            seed_labels, st,
            max_rounds=max_rounds,
            min_strength=min_strength,
            exact_only=True,
        )
        for nid, weight in st.convergence_map.items():
            n = brain.neuron_repo.get_by_id(nid)
            if n and len(n.label) >= 3:
                keywords[n.label.lower()] = max(
                    keywords.get(n.label.lower(), 0), weight
                )
    return keywords


def multi_threshold_echo(brain, seed_labels: list[str],
                         max_rounds: int = 2) -> dict[str, float]:
    """Run echo at three inhibition levels, blend results.

    Focused (0.5): weight 3x — "I know this"
    Relaxed (0.2): weight 2x — "I think this"
    Open   (0.0): weight 1x — "this is possible"

    All thought is a path. Speculation differs from confidence in
    degree, not in kind. An expert's speculation beats a non-expert's
    confidence.
    """
    thresholds = [(0.5, 3.0), (0.3, 2.0), (0.1, 1.0)]
    blended: dict[str, float] = {}

    for min_str, multiplier in thresholds:
        kw = echo_keywords_from_layer(
            brain, seed_labels, min_strength=min_str, max_rounds=max_rounds
        )
        for label, weight in kw.items():
            blended[label] = blended.get(label, 0) + weight * multiplier

    return blended


def score_choice(choice_text: str, all_keywords: dict[str, float]) -> tuple[float, list[str]]:
    """Score a choice by keyword overlap with the combined signal."""
    choice_lower = choice_text.lower()
    matched = []
    total = 0.0
    for kw, w in all_keywords.items():
        if re.search(rf"\b{re.escape(kw)}\b", choice_lower):
            matched.append(kw)
            total += w
    return total, matched


def main():
    with open("benchmarks/bio_10q_questions.json") as f:
        questions = json.load(f)

    vocab_brain = Brain("layer_vocab.db")
    science_brain = Brain("layer_science.db")
    biology_brain = Brain("layer_biology.db")

    # Dictionary region — synonym network within vocab brain
    # Uses prefixed repos so the echo traverses dictionary tables
    from sara_brain.storage.neuron_repo import NeuronRepo
    from sara_brain.storage.segment_repo import SegmentRepo
    from sara_brain.core.recognizer import Recognizer

    dict_neuron_repo = NeuronRepo(vocab_brain.conn, prefix="dictionary")
    dict_segment_repo = SegmentRepo(vocab_brain.conn, prefix="dictionary")
    dict_recognizer = Recognizer(dict_neuron_repo, dict_segment_repo)

    # Wrap in a lightweight object that has short_term + propagate_echo
    class DictLayer:
        def __init__(self, conn, recognizer):
            self.conn = conn
            self.recognizer = recognizer
            self.neuron_repo = recognizer.neuron_repo

        def short_term(self, event_type="query"):
            from sara_brain.core.short_term import ShortTerm
            import time as _time
            from contextlib import contextmanager

            @contextmanager
            def _ctx():
                st = ShortTerm(
                    event_id=f"{event_type}-{_time.time():.3f}",
                    event_type=event_type,
                )
                try:
                    yield st
                finally:
                    pass
            return _ctx()

        def propagate_echo(self, seeds, st, **kwargs):
            self.recognizer.propagate_echo(seeds, st, **kwargs)

    dict_brain = DictLayer(vocab_brain.conn, dict_recognizer)

    correct = 0
    answered = 0
    abstained = 0

    for q in questions:
        qid = q["id"]
        q_words = extract_words(q["question"])

        # For each choice: seed echo with question words + THIS choice's
        # words. Each choice gets its own echo through all three layers.
        # The choice whose echo pings the most through Sara's graph wins.
        choice_scores = []
        for choice in q["choices"]:
            c_words = extract_words(choice)
            seeds = q_words + c_words

            # Dictionary: expand ONLY words that don't exist in any domain
            # layer. Like a student who looks up words they don't know —
            # don't look up words you already understand.
            expanded_seeds = list(seeds)
            for seed in seeds:
                # Skip if any domain layer already knows this word
                known = (
                    vocab_brain.neuron_repo.resolve(seed, exact_only=True)
                    or science_brain.neuron_repo.resolve(seed, exact_only=True)
                    or biology_brain.neuron_repo.resolve(seed, exact_only=True)
                )
                if known:
                    continue
                # Unknown word — look up synonyms
                n = dict_neuron_repo.resolve(seed, exact_only=True)
                if n is None:
                    continue
                for seg in dict_segment_repo.get_outgoing(n.id):
                    if seg.relation == "synonym_of":
                        syn = dict_neuron_repo.get_by_id(seg.target_id)
                        if syn and syn.label not in expanded_seeds:
                            expanded_seeds.append(syn.label)
            seeds = expanded_seeds
            # Other layers: full multi-threshold echo
            vocab_kw = multi_threshold_echo(vocab_brain, seeds)
            science_kw = multi_threshold_echo(science_brain, seeds)
            biology_kw = multi_threshold_echo(biology_brain, seeds)

            # Combine layers (biology weighted highest)
            all_kw: dict[str, float] = {}
            for kw_set, layer_mult in [
                (vocab_kw, 0.1), (science_kw, 1.0), (biology_kw, 10.0),
            ]:
                for kw, w in kw_set.items():
                    all_kw[kw] = all_kw.get(kw, 0) + w * layer_mult

            # Score: how many of the echo keywords appear in the choice text
            weight, matched = score_choice(choice, all_kw)

            # Also score against question text — keywords that appear in
            # BOTH the choice echo AND the question text are the bridges
            q_weight, q_matched = score_choice(q["question"], all_kw)

            combined = weight + q_weight
            all_matched = list(set(matched + q_matched))

            choice_scores.append({
                "combined": combined,
                "choice_w": weight,
                "question_w": q_weight,
                "matched": all_matched,
                "echo_size": len(all_kw),
            })

        # Pick the choice with highest combined score
        best_combined = max(s["combined"] for s in choice_scores)
        if best_combined == 0:
            abstained += 1
            correct_letter = chr(65 + q["answer_idx"])
            print(f"  Q{qid:2d}: ABSTAIN (correct={correct_letter}) "
                  f"echo sizes={[s['echo_size'] for s in choice_scores]}")
            continue

        best_idx = max(
            range(4),
            key=lambda i: (
                choice_scores[i]["combined"],
                len(choice_scores[i]["matched"]),
            ),
        )
        letter = chr(65 + best_idx)
        correct_letter = chr(65 + q["answer_idx"])
        is_correct = best_idx == q["answer_idx"]

        answered += 1
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "WRONG"
        best = choice_scores[best_idx]
        kw_preview = ",".join(best["matched"][:4])
        if len(best["matched"]) > 4:
            kw_preview += f"+{len(best['matched'])-4}"
        print(f"  Q{qid:2d}: {status:7s} chose={letter} correct={correct_letter} "
              f"w={best['combined']:.1f} echo={best['echo_size']} "
              f"kw=[{kw_preview}]")

    print()
    print(f"  Answered: {answered}/10  Abstained: {abstained}/10")
    if answered:
        print(f"  Scored accuracy: {correct}/{answered} = {correct/answered*100:.0f}%")
    print(f"  Overall: {correct}/10 = {correct*10}%")

    vocab_brain.close()
    science_brain.close()
    biology_brain.close()


if __name__ == "__main__":
    main()
