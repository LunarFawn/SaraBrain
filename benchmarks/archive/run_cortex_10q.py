#!/usr/bin/env python3
"""Brain + Cortex benchmark — the architecture as designed.

Sara's brain produces noisy activation (echo propagation through
layered regions). The cortex (3B LLM) reads the activation and
picks the answer. Neither works well alone. Together they should
beat both.

The brain does activation. The cortex does interpretation.
"""

from __future__ import annotations

import json
import re
import urllib.request

from sara_brain.core.brain import Brain
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo
from sara_brain.core.recognizer import Recognizer
from sara_brain.core.short_term import ShortTerm


def extract_words(text: str) -> list[str]:
    stops = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "them", "than", "then", "more", "most", "also", "only",
        "each", "both", "some", "many", "such", "very", "just", "into",
        "your", "will", "would", "could", "should", "which", "what",
        "when", "where", "about", "above", "below", "these", "those",
        "able", "result", "following", "example", "best", "likely",
        "occurs", "along", "pass", "and", "for", "the", "are", "not",
    }
    raw = re.findall(r"[a-z][a-z'-]+", text.lower())
    words = []
    seen = set()
    for w in raw:
        if len(w) < 4 or w in stops:
            continue
        forms = [w]
        if w.endswith("er") and len(w) > 4:
            forms.append(w[:-2])
        if w.endswith("est") and len(w) > 5:
            forms.append(w[:-3])
        if w.endswith("ly") and len(w) > 4:
            forms.append(w[:-2])
        if w.endswith("ing") and len(w) > 5:
            forms.append(w[:-3])
            forms.append(w[:-3] + "e")
        if w.endswith("s") and not w.endswith("ss") and len(w) > 4:
            forms.append(w[:-1])
        for form in forms:
            if len(form) >= 3 and form not in seen and form not in stops:
                seen.add(form)
                words.append(form)
    return words


def call_ollama(prompt: str, system: str, model: str,
                base_url: str = "http://localhost:11434") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


def get_activation(brains: list, dict_nr, dict_sr, seeds: list[str],
                   known_brains: list) -> dict[str, float]:
    """Run echo through all layers, return activation pattern."""
    # Dictionary 2-hop expansion for unknown words
    expanded = set(seeds)
    for seed in seeds:
        known = any(
            b.neuron_repo.resolve(seed, exact_only=True)
            for b in known_brains
        )
        if known:
            continue
        n = dict_nr.resolve(seed, exact_only=True)
        if n is None:
            continue
        hop1_ids = []
        for seg in dict_sr.get_outgoing(n.id):
            if seg.relation == "synonym_of":
                hop1_ids.append(seg.target_id)
                syn = dict_nr.get_by_id(seg.target_id)
                if syn:
                    expanded.add(syn.label)
        for tid in hop1_ids:
            for seg2 in dict_sr.get_outgoing(tid):
                if seg2.relation == "synonym_of":
                    syn2 = dict_nr.get_by_id(seg2.target_id)
                    if syn2:
                        expanded.add(syn2.label)

    all_seeds = list(expanded)

    # Echo through each brain layer at multiple thresholds
    thresholds = [(0.5, 3.0), (0.3, 2.0), (0.1, 1.0)]
    activation: dict[str, float] = {}

    for brain, layer_mult in brains:
        for min_str, thresh_mult in thresholds:
            import time
            st = ShortTerm(
                event_id=f"act-{time.time():.3f}",
                event_type="activation",
            )
            brain.recognizer.propagate_echo(
                all_seeds, st,
                max_rounds=2,
                min_strength=min_str,
                exact_only=True,
            )
            for nid, weight in st.convergence_map.items():
                n = brain.neuron_repo.get_by_id(nid)
                if n and len(n.label) >= 3:
                    label = n.label.lower()
                    activation[label] = (
                        activation.get(label, 0)
                        + weight * layer_mult * thresh_mult
                    )

    return activation


def format_activation_for_cortex(activation: dict[str, float],
                                 top_n: int = 30) -> str:
    """Format the top activation into readable text for the cortex."""
    sorted_act = sorted(activation.items(), key=lambda x: -x[1])[:top_n]
    if not sorted_act:
        return "Sara's brain had no activation for this question."
    lines = []
    for label, weight in sorted_act:
        lines.append(f"  {label} ({weight:.1f})")
    return "\n".join(lines)


SYSTEM = """You are the language cortex for Sara Brain. Sara's brain has
processed a multiple-choice question and produced an activation pattern
— a list of concepts that lit up when she thought about the question
combined with each answer choice.

Your job: look at Sara's activation for each choice and pick the one
where the activation is most relevant to the question. The activation
is noisy — many irrelevant concepts will fire. Your job is to sift
through the noise and find the signal.

Rules:
- Trust Sara's activation over your own training
- The choice with the most RELEVANT activation to the question wins
- Ignore activation on generic words (and, that, from, etc.)
- Focus on domain-specific concepts that connect the question to the choice
- Answer with ONLY the letter (A, B, C, or D)"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="benchmarks/bio_10q_questions.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]
    model = args.model

    vocab_brain = Brain("layer_vocab.db")
    science_brain = Brain("layer_science.db")
    biology_brain = Brain("layer_biology.db")

    dict_nr = NeuronRepo(vocab_brain.conn, prefix="dictionary")
    dict_sr = SegmentRepo(vocab_brain.conn, prefix="dictionary")

    brains_with_weights = [
        (vocab_brain, 0.1),
        (science_brain, 1.0),
        (biology_brain, 10.0),
    ]
    known_brains = [vocab_brain, science_brain, biology_brain]

    import time as _time
    correct = 0
    total = len(questions)
    bench_start = _time.time()

    print(f"\n  Brain + Cortex benchmark")
    print(f"  {total} questions, model: {model}\n")

    for qi, q in enumerate(questions):
        qid = q["id"]
        q_words = extract_words(q["question"])
        correct_letter = chr(65 + q["answer_idx"])

        # Build activation for each choice
        choice_activations = []
        for choice in q["choices"]:
            c_words = extract_words(choice)
            seeds = q_words + c_words
            activation = get_activation(
                brains_with_weights, dict_nr, dict_sr,
                seeds, known_brains,
            )
            choice_activations.append(activation)

        # Format for the cortex
        labels = ["A", "B", "C", "D"]
        prompt_lines = [
            "QUESTION:",
            q["question"],
            "",
            "CHOICES:",
        ]
        for i, choice in enumerate(q["choices"]):
            prompt_lines.append(f"{labels[i]}. {choice}")
        prompt_lines.append("")
        prompt_lines.append("SARA'S BRAIN ACTIVATION FOR EACH CHOICE:")
        prompt_lines.append("")

        for i, (choice, activation) in enumerate(
            zip(q["choices"], choice_activations)
        ):
            prompt_lines.append(f"--- Choice {labels[i]} activation ---")
            prompt_lines.append(
                format_activation_for_cortex(activation, top_n=20)
            )
            prompt_lines.append("")

        prompt_lines.append(
            "Based on Sara's activation patterns, which choice (A/B/C/D) "
            "has the most relevant activation to the question?"
        )

        prompt = "\n".join(prompt_lines)

        # Ask the cortex
        response = call_ollama(prompt, SYSTEM, model)
        answer = None
        for char in response.strip().upper():
            if char in "ABCD":
                answer = char
                break

        is_correct = answer == correct_letter
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "WRONG"
        elapsed = _time.time() - bench_start
        avg = elapsed / (qi + 1)
        remaining = avg * (total - qi - 1)
        accuracy = correct / (qi + 1) * 100
        print(f"  [{qi+1}/{total}] Q{qid:2d}: {status:7s} cortex={answer} "
              f"correct={correct_letter} — {accuracy:.1f}% "
              f"— {avg:.1f}s/q (~{remaining/60:.0f}m left)",
              flush=True)

    print()
    total_time = _time.time() - bench_start
    print(f"  ={'='*55}")
    print(f"  Brain + Cortex: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Time: {total_time/60:.1f} min ({total_time/total:.1f}s/question)")
    print(f"  ={'='*55}")
    print()
    print(f"  Reference:")
    print(f"    Random:        25.0%")
    print(f"    Brain alone:   50.0% (on 10-question sample)")
    print(f"    3B alone:      58.4% (no Sara)")
    print(f"    GPT-3.5:       ~70%")
    print(f"    GPT-4:         ~86%")


if __name__ == "__main__":
    main()
