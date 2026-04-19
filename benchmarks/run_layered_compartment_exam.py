#!/usr/bin/env python3
"""Layered + Compartmentalized exam.

Full pipeline: dictionary expansion → vocab → science → compartmentalized
biology regions. Best of both: the vocabulary bridging that produced 80%
PLUS the concept isolation that eliminates cross-activation noise.

Usage:
    python benchmarks/run_layered_compartment_exam.py \\
        --bio-db claude_taught.db \\
        --questions benchmarks/ch10_test_questions.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.core.recognizer import Recognizer
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo


# ── LLM ──

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


# ── Words ──

def extract_words(text: str) -> list[str]:
    stops = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "them", "than", "then", "more", "most", "also", "only",
        "each", "both", "some", "many", "such", "very", "just", "into",
        "your", "will", "would", "could", "should", "which", "what",
        "when", "where", "about", "above", "below", "these", "those",
        "and", "for", "the", "are", "not", "all", "following",
    }
    raw = re.findall(r"[a-z][a-z'-]+", text.lower())
    words = []
    seen = set()
    for w in raw:
        if len(w) < 4 or w in stops:
            continue
        forms = [w]
        if w.endswith("s") and len(w) > 4:
            forms.append(w[:-1])
        if w.endswith("ly") and len(w) > 4:
            forms.append(w[:-2])
        if w.endswith("ing") and len(w) > 5:
            forms.append(w[:-3])
        for f in forms:
            if f not in seen and len(f) >= 3:
                seen.add(f)
                words.append(f)
    return words


# ── Region selection ──

def select_regions(text: str, bio_brain: Brain, regions: list[str]) -> list[str]:
    words = extract_words(text)
    scored = []
    for region in regions:
        nr = NeuronRepo(bio_brain.conn, prefix=region)
        hits = sum(1 for w in words if nr.get_by_label(w) is not None)
        if hits > 0:
            scored.append((region, hits))
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored[:2]] if scored else regions[:1]


# ── Activation from all layers ──

def get_activation(seeds: list[str],
                   vocab_brain: Brain,
                   science_brain: Brain,
                   bio_brain: Brain,
                   bio_regions: list[str],
                   dict_nr, dict_sr) -> dict[str, float]:
    """Full layered activation: dictionary → vocab → science → bio regions."""

    # Dictionary 2-hop expansion for unknown words
    known_brains = [vocab_brain, science_brain]
    expanded = set(seeds)
    for seed in seeds:
        known = any(b.neuron_repo.resolve(seed, exact_only=True)
                    for b in known_brains)
        if known:
            continue
        # Also check bio regions
        for region in bio_regions:
            nr = NeuronRepo(bio_brain.conn, prefix=region)
            if nr.get_by_label(seed) is not None:
                known = True
                break
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
    activation: dict[str, float] = {}

    # Layer weights
    layers = [
        (vocab_brain, 0.1, None),
        (science_brain, 1.0, None),
    ]
    # Add bio regions as separate layers
    for region in bio_regions:
        layers.append((bio_brain, 10.0, region))

    thresholds = [(0.5, 3.0), (0.3, 2.0), (0.1, 1.0)]

    for brain, layer_mult, region in layers:
        if region:
            nr = NeuronRepo(brain.conn, prefix=region)
            sr = SegmentRepo(brain.conn, prefix=region)
            recognizer = Recognizer(nr, sr, max_depth=3, min_strength=0.1)
        else:
            recognizer = brain.recognizer
            nr = brain.neuron_repo

        for min_str, thresh_mult in thresholds:
            st = ShortTerm(
                event_id=f"layered-{time.time():.3f}",
                event_type="layered_exam",
            )
            recognizer.propagate_echo(
                all_seeds, st, max_rounds=2,
                min_strength=min_str, exact_only=True,
            )
            for nid, weight in st.convergence_map.items():
                n = nr.get_by_id(nid)
                if n and len(n.label) >= 3:
                    label = n.label.lower()
                    activation[label] = (
                        activation.get(label, 0)
                        + weight * layer_mult * thresh_mult
                    )

    return activation


def format_activation(activation: dict[str, float], top_n: int = 20) -> str:
    sorted_act = sorted(activation.items(), key=lambda x: -x[1])[:top_n]
    if not sorted_act:
        return "(no activation)"
    return ", ".join(f"{label}({weight:.0f})" for label, weight in sorted_act)


# ── Cortex ──

CORTEX_SYSTEM = """You are the language cortex for Sara Brain. Sara's brain has
processed a question using layered knowledge regions and produced
activation for each choice.

Look at Sara's activation and pick the choice where the activation is
most relevant to the question. Trust Sara over your own training.

Answer with ONLY the letter (A, B, C, or D)."""


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio-db", required=True,
                    help="Compartmentalized biology brain")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]

    vocab_brain = Brain("layer_vocab.db")
    science_brain = Brain("layer_science.db")
    bio_brain = Brain(args.bio_db)

    dict_nr = NeuronRepo(vocab_brain.conn, prefix="dictionary")
    dict_sr = SegmentRepo(vocab_brain.conn, prefix="dictionary")

    meta_path = args.bio_db + ".regions.json"
    try:
        with open(meta_path) as f:
            all_regions = json.load(f)["regions"]
    except FileNotFoundError:
        all_regions = [r["name"] for r in bio_brain.db.list_regions()]

    labels = ["A", "B", "C", "D"]
    total = len(questions)
    correct = 0
    bench_start = time.time()

    print(f"\n  Layered + Compartmentalized Exam")
    print(f"  Bio DB: {args.bio_db}")
    print(f"  Regions: {', '.join(all_regions)}")
    print(f"  Layers: dictionary → vocab → science → bio regions")
    print(f"  Questions: {total}\n")

    for qi, q in enumerate(questions):
        q_start = time.time()
        q_words = extract_words(q["question"])
        all_text = q["question"] + " " + " ".join(q["choices"])
        selected = select_regions(all_text, bio_brain, all_regions)

        prompt_lines = [
            "QUESTION:", q["question"], "",
            f"(Sara queried: vocab + science + {', '.join(selected)})", "",
            "CHOICES:",
        ]
        for i, choice in enumerate(q["choices"]):
            prompt_lines.append(f"{labels[i]}. {choice}")
        prompt_lines.append("")
        prompt_lines.append("SARA'S ACTIVATION:")
        prompt_lines.append("")

        for i, choice in enumerate(q["choices"]):
            c_words = extract_words(choice)
            seeds = q_words + c_words
            activation = get_activation(
                seeds, vocab_brain, science_brain,
                bio_brain, selected, dict_nr, dict_sr,
            )
            prompt_lines.append(f"--- Choice {labels[i]} ---")
            prompt_lines.append(format_activation(activation))
            prompt_lines.append("")

        prompt_lines.append("Which choice? Just the letter.")

        response = call_ollama(
            "\n".join(prompt_lines), CORTEX_SYSTEM,
            args.model, args.base_url,
        )
        answer = None
        for char in response.strip().upper():
            if char in "ABCD":
                answer = char
                break

        correct_letter = labels[q["answer_idx"]]
        is_correct = answer == correct_letter
        if is_correct:
            correct += 1

        elapsed = time.time() - q_start
        total_elapsed = time.time() - bench_start
        avg = total_elapsed / (qi + 1)
        remaining = avg * (total - qi - 1)
        accuracy = correct / (qi + 1) * 100
        status = "✓" if is_correct else "✗"

        print(f"  [{qi+1}/{total}] Q{q['id']}: {status} "
              f"cortex={answer} correct={correct_letter} "
              f"bio_regions={selected} "
              f"— {accuracy:.1f}% ({elapsed:.0f}s, ~{remaining/60:.0f}m left)",
              flush=True)

    total_time = time.time() - bench_start
    print(f"\n  {'='*55}")
    print(f"  Layered + Compartmentalized: "
          f"{correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  {'='*55}")
    print(f"\n  Comparison:")
    print(f"    Random:                    25.0%")
    print(f"    Pure shell (no LLM):       18.2%")
    print(f"    Pre-test (3B+Sara flat):   45.5%")
    print(f"    Compartment only:          45.5%")
    print(f"    3B alone:                  58.4%")
    print(f"    Hand-taught 10q:           80.0%")
    print(f"    Layered+Compartment:       {correct/total*100:.1f}%")

    vocab_brain.close()
    science_brain.close()
    bio_brain.close()


if __name__ == "__main__":
    main()
