#!/usr/bin/env python3
"""Compartmentalized exam — query only the relevant region(s) per question.

Each question is routed to the brain region(s) that contain matching
concepts. Cross-activation noise is eliminated because regions are
isolated tables.

Usage:
    python benchmarks/run_compartment_exam.py --db compartment.db \\
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
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words


# ── Region selection ──

def select_regions(question: str, choices: list[str],
                   brain: Brain, regions: list[str]) -> list[str]:
    """Find which regions are relevant to this question + choices."""
    all_text = question + " " + " ".join(choices)
    words = extract_words(all_text)

    scored = []
    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        hits = 0
        for word in words:
            n = nr.get_by_label(word)
            if n is not None:
                hits += 1
        if hits > 0:
            scored.append((region, hits))

    scored.sort(key=lambda x: -x[1])
    # Return top 2 regions max — focused, not flooding
    return [r for r, _ in scored[:2]] if scored else regions[:1]


# ── Activation from specific regions ──

def get_regional_activation(brain: Brain, regions: list[str],
                            seeds: list[str]) -> dict[str, float]:
    """Run echo propagation through ONLY the selected regions."""
    activation: dict[str, float] = {}

    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        sr = SegmentRepo(brain.conn, prefix=region)
        recognizer = Recognizer(nr, sr, max_depth=3, min_strength=0.1)

        st = ShortTerm(
            event_id=f"exam-{time.time():.3f}",
            event_type="exam",
        )
        recognizer.propagate_echo(
            seeds, st, max_rounds=2, min_strength=0.1, exact_only=True,
        )

        for nid, weight in st.convergence_map.items():
            n = nr.get_by_id(nid)
            if n and len(n.label) >= 3:
                label = n.label.lower()
                activation[label] = activation.get(label, 0) + weight

    return activation


def format_activation(activation: dict[str, float], top_n: int = 15) -> str:
    sorted_act = sorted(activation.items(), key=lambda x: -x[1])[:top_n]
    if not sorted_act:
        return "(no activation)"
    return ", ".join(f"{label}({weight:.0f})" for label, weight in sorted_act)


# ── Cortex ──

CORTEX_SYSTEM = """You are the language cortex for Sara Brain. Sara's brain has
processed a question and produced activation for each choice from
specific knowledge regions.

Look at Sara's activation and pick the choice where the activation is
most relevant to the question. Trust Sara over your own training.

Answer with ONLY the letter (A, B, C, or D)."""


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--model", default="qwen2.5-coder:3b")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]

    brain = Brain(args.db)

    # Load region list
    meta_path = args.db + ".regions.json"
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        regions = meta["regions"]
    except FileNotFoundError:
        regions = [r["name"] for r in brain.db.list_regions()]

    print(f"\n  Compartmentalized Exam")
    print(f"  Brain: {args.db}")
    print(f"  Regions: {', '.join(regions)}")
    print(f"  Questions: {len(questions)}\n")

    correct = 0
    total = len(questions)
    bench_start = time.time()
    labels = ["A", "B", "C", "D"]

    for qi, q in enumerate(questions):
        q_start = time.time()
        qid = q["id"]

        # Select relevant regions
        selected = select_regions(
            q["question"], q["choices"], brain, regions
        )

        # Build activation per choice from selected regions only
        q_words = extract_words(q["question"])
        prompt_lines = [
            "QUESTION:", q["question"], "",
            f"(Sara queried regions: {', '.join(selected)})", "",
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
            activation = get_regional_activation(brain, selected, seeds)
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

        print(f"  [{qi+1}/{total}] Q{qid}: {status} "
              f"cortex={answer} correct={correct_letter} "
              f"regions={selected} "
              f"— {accuracy:.1f}% ({elapsed:.0f}s, ~{remaining/60:.0f}m left)",
              flush=True)

    total_time = time.time() - bench_start
    print(f"\n  {'='*55}")
    print(f"  Compartmentalized: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  {'='*55}")
    print(f"\n  Comparison:")
    print(f"    Pre-test (no study):        45.5%")
    print(f"    Flat brain (old parser):     48.5%")
    print(f"    Flat brain (new parser):     ~28%")
    print(f"    Compartmentalized:          {correct/total*100:.1f}%")

    brain.close()


if __name__ == "__main__":
    main()
