#!/usr/bin/env python3
"""Teacher Interface — Sara learns like a student, not a model.

Iterative test→error→correction→retest loop. Sara takes a test, gets
graded, examines her own activation on wrong answers, identifies the
gap with help from the cortex, teaches herself the correction, and
retakes the test. Regressions (previously correct → now wrong) trigger
distinction-teaching ("those are separate concepts").

Runs until Sara gets all questions right, plateaus, or hits max rounds.

The learning curve IS the paper's result: how accuracy improves with
each round of error-driven teaching, starting from an empty brain.

Usage:
    # Fresh brain, 20 questions, iterate until convergence
    python benchmarks/error_learning_loop.py --questions benchmarks/mmlu_biology_full.json --limit 20

    # Resume with existing brain layers
    python benchmarks/error_learning_loop.py --questions benchmarks/bio_10q_questions.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo


# ── Word extraction (shared with run_layered_10q) ──

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


# ── LLM call ──

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


# ── Activation ──

def get_activation(brains_with_weights, dict_nr, dict_sr,
                   seeds, known_brains) -> dict[str, float]:
    """Run echo through all layers, return activation pattern."""
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
    thresholds = [(0.5, 3.0), (0.3, 2.0), (0.1, 1.0)]
    activation: dict[str, float] = {}

    for brain, layer_mult in brains_with_weights:
        for min_str, thresh_mult in thresholds:
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


def format_activation(activation: dict[str, float], top_n: int = 20) -> str:
    sorted_act = sorted(activation.items(), key=lambda x: -x[1])[:top_n]
    if not sorted_act:
        return "(no activation)"
    return ", ".join(f"{label}({weight:.0f})" for label, weight in sorted_act)


# ── Test one question ──

CORTEX_SYSTEM = """You are the language cortex for Sara Brain. Sara's brain has
processed a multiple-choice question and produced an activation pattern.

Your job: look at Sara's activation for each choice and pick the one
where the activation is most relevant to the question. The activation
is noisy — sift through the noise and find the signal.

Trust Sara's activation over your own training.
Answer with ONLY the letter (A, B, C, or D)."""


def test_question(q, brains_with_weights, dict_nr, dict_sr,
                  known_brains, model, base_url) -> dict:
    """Test one question. Returns result dict with activation details."""
    q_words = extract_words(q["question"])
    labels = ["A", "B", "C", "D"]

    choice_activations = []
    for choice in q["choices"]:
        c_words = extract_words(choice)
        seeds = q_words + c_words
        activation = get_activation(
            brains_with_weights, dict_nr, dict_sr,
            seeds, known_brains,
        )
        choice_activations.append(activation)

    # Build prompt for cortex
    prompt_lines = [
        "QUESTION:", q["question"], "",
        "CHOICES:",
    ]
    for i, choice in enumerate(q["choices"]):
        prompt_lines.append(f"{labels[i]}. {choice}")
    prompt_lines.append("")
    prompt_lines.append("SARA'S BRAIN ACTIVATION:")
    prompt_lines.append("")
    for i, activation in enumerate(choice_activations):
        prompt_lines.append(f"--- Choice {labels[i]} ---")
        prompt_lines.append(format_activation(activation))
        prompt_lines.append("")
    prompt_lines.append("Which choice has the most relevant activation? Answer with just the letter.")

    response = call_ollama("\n".join(prompt_lines), CORTEX_SYSTEM,
                            model, base_url)
    answer = None
    for char in response.strip().upper():
        if char in "ABCD":
            answer = char
            break

    correct_idx = q["answer_idx"]
    correct_letter = labels[correct_idx]
    chosen_idx = labels.index(answer) if answer in labels else -1

    return {
        "id": q["id"],
        "answer": answer,
        "correct": correct_letter,
        "is_correct": answer == correct_letter,
        "chosen_idx": chosen_idx,
        "correct_idx": correct_idx,
        "activations": choice_activations,
        "question": q["question"],
        "choices": q["choices"],
    }


# ── Gap identification ──

GAP_SYSTEM = """You are a teacher helping Sara Brain learn from her mistakes.

Sara got this question wrong. You will see:
1. The question and choices
2. What Sara picked (wrong) and what the correct answer is
3. Sara's brain activation for BOTH the wrong and correct choices

Your job: identify what FACT Sara is missing. State it as a simple
"X is Y" sentence that Sara's parser can learn. One fact per line.
Maximum 3 facts. Focus on the most critical missing knowledge.

Rules:
- Simple sentences only: "X is Y" or "X are Y"
- No citations, no hedging, no "may" or "might"
- State facts that directly bridge the question to the correct answer
- Do NOT restate the question or the choices"""


def identify_gap(result: dict, model: str, base_url: str) -> list[str]:
    """Ask the cortex what fact Sara is missing."""
    labels = ["A", "B", "C", "D"]
    chosen = result["chosen_idx"]
    correct = result["correct_idx"]

    prompt_lines = [
        f"QUESTION: {result['question']}",
        "",
        f"Sara picked: {labels[chosen]}. {result['choices'][chosen]}",
        f"Correct answer: {labels[correct]}. {result['choices'][correct]}",
        "",
        f"Sara's activation for her WRONG choice ({labels[chosen]}):",
        format_activation(result["activations"][chosen], top_n=15),
        "",
        f"Sara's activation for the CORRECT choice ({labels[correct]}):",
        format_activation(result["activations"][correct], top_n=15),
        "",
        "What fact(s) is Sara missing? State as simple 'X is Y' sentences.",
    ]

    response = call_ollama("\n".join(prompt_lines), GAP_SYSTEM,
                            model, base_url)

    # Parse response into individual facts
    facts = []
    for line in response.splitlines():
        cleaned = line.strip().lstrip("-*•·0123456789.)")
        cleaned = cleaned.strip()
        if not cleaned or len(cleaned) < 10 or len(cleaned) > 200:
            continue
        if cleaned.upper().startswith("SARA") or cleaned.upper().startswith("THE MISSING"):
            continue
        facts.append(cleaned)
    return facts[:3]


# ── Regression detection ──

DISTINCTION_SYSTEM = """You are a teacher helping Sara Brain fix a confusion.

Sara previously got this question RIGHT but now gets it WRONG after
learning new facts. The new knowledge is interfering with the old.

Your job: state a DISTINCTION fact that separates the confused concepts.
Format: "X is not related to Y" or "X is different from Y"
One sentence only. Simple words."""


def handle_regression(q, prev_result, curr_result, model, base_url) -> str | None:
    """Generate a distinction fact for a regression."""
    labels = ["A", "B", "C", "D"]
    prompt = (
        f"QUESTION: {q['question']}\n\n"
        f"Last round Sara correctly picked: {prev_result['answer']}\n"
        f"This round Sara wrongly picked: {curr_result['answer']}\n"
        f"The correct answer is: {curr_result['correct']}\n\n"
        f"What distinction should Sara learn to prevent this confusion?"
    )
    response = call_ollama(prompt, DISTINCTION_SYSTEM, model, base_url)
    cleaned = response.strip().split("\n")[0].strip()
    if len(cleaned) > 10:
        return cleaned
    return None


# ── Main loop ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions",
                        default="benchmarks/mmlu_biology_full.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--fresh", action="store_true",
                        help="Start with fresh brain layers")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]

    # Fresh brain layers if requested
    if args.fresh:
        for db in ["layer_vocab.db", "layer_science.db", "layer_biology.db"]:
            for ext in ["", "-shm", "-wal"]:
                path = db + ext
                if os.path.exists(path):
                    os.remove(path)
        # Recreate empty brains
        Brain("layer_vocab.db").close()
        Brain("layer_science.db").close()
        Brain("layer_biology.db").close()
        # Rebuild dictionary
        import subprocess, sys
        subprocess.run([
            sys.executable, "benchmarks/build_dictionary.py",
            "--db", "layer_vocab.db", "--region", "dictionary",
        ])
        print("  Fresh brain layers created\n")

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

    total_q = len(questions)
    print(f"\n  ╔{'═'*56}╗")
    print(f"  ║  Teacher Interface — Error-Driven Learning Loop      ║")
    print(f"  ║  {total_q} questions, max {args.max_rounds} rounds{' '*25}║")
    print(f"  ╚{'═'*56}╝\n")

    prev_results: dict[int, dict] = {}
    all_rounds: list[dict] = []
    total_facts_taught = 0
    total_distinctions = 0

    for round_num in range(1, args.max_rounds + 1):
        print(f"  ═══ ROUND {round_num} ═══\n")
        round_start = time.time()

        # Test all questions
        results: dict[int, dict] = {}
        correct_count = 0
        for qi, q in enumerate(questions):
            result = test_question(
                q, brains_with_weights, dict_nr, dict_sr,
                known_brains, args.model, args.base_url,
            )
            results[q["id"]] = result
            if result["is_correct"]:
                correct_count += 1
            status = "✓" if result["is_correct"] else "✗"
            print(f"    {status} Q{q['id']:2d}: {result['answer']} "
                  f"(correct {result['correct']})", flush=True)

        accuracy = correct_count / total_q * 100
        round_time = time.time() - round_start

        # Detect regressions
        regressions = []
        if prev_results:
            for qid, curr in results.items():
                prev = prev_results.get(qid)
                if prev and prev["is_correct"] and not curr["is_correct"]:
                    regressions.append((qid, prev, curr))

        wrong = [r for r in results.values() if not r["is_correct"]]

        print(f"\n  Round {round_num} score: {correct_count}/{total_q} "
              f"= {accuracy:.0f}% ({round_time:.0f}s)")
        if regressions:
            print(f"  ⚠ {len(regressions)} regression(s)")
        print()

        round_log = {
            "round": round_num,
            "accuracy": accuracy,
            "correct": correct_count,
            "wrong_ids": [r["id"] for r in wrong],
            "regression_ids": [qid for qid, _, _ in regressions],
            "facts_taught": 0,
            "distinctions": 0,
        }

        # Check convergence
        if correct_count == total_q:
            print(f"  ★ ALL CORRECT — Sara has graduated! ★\n")
            all_rounds.append(round_log)
            break

        if (prev_results and
            correct_count == sum(1 for r in prev_results.values()
                                  if r["is_correct"]) and
            set(r["id"] for r in wrong) ==
            set(r["id"] for r in prev_results.values()
                if not r["is_correct"])):
            print(f"  ■ Plateau — same score and same wrong answers. "
                  f"Sara can't learn more from these questions.\n")
            all_rounds.append(round_log)
            break

        # Handle regressions first — teach distinctions
        for qid, prev, curr in regressions:
            q = [x for x in questions if x["id"] == qid][0]
            print(f"  REGRESSION Q{qid}: was {prev['answer']}(✓) "
                  f"now {curr['answer']}(✗)")
            distinction = handle_regression(
                q, prev, curr, args.model, args.base_url
            )
            if distinction:
                r = biology_brain.teach_from_error(
                    distinction,
                    error_context=f"regression Q{qid} round {round_num}",
                )
                if r:
                    print(f"    DISTINCTION: {r.path_label}")
                    total_distinctions += 1
                    round_log["distinctions"] += 1
                else:
                    print(f"    FAILED to parse: {distinction}")
            print()

        # Teach from errors on wrong answers
        for result in wrong:
            qid = result["id"]
            if qid in [r[0] for r in regressions]:
                continue  # already handled as regression

            print(f"  ERROR Q{qid}: picked {result['answer']}, "
                  f"correct {result['correct']}")
            gap_facts = identify_gap(result, args.model, args.base_url)

            for fact in gap_facts:
                r = biology_brain.teach_from_error(
                    fact,
                    error_context=f"Q{qid} round {round_num}: "
                                  f"picked {result['answer']}, "
                                  f"correct {result['correct']}",
                )
                if r:
                    print(f"    LEARNED: {r.path_label}")
                    total_facts_taught += 1
                    round_log["facts_taught"] += 1
                else:
                    print(f"    SKIP: {fact}")
            print()

        all_rounds.append(round_log)
        prev_results = results

    # Final summary
    print(f"  {'='*56}")
    print(f"  LEARNING CURVE")
    print(f"  {'='*56}")
    for r in all_rounds:
        bar = "█" * int(r["accuracy"] / 5) + "░" * (20 - int(r["accuracy"] / 5))
        reg = f" ⚠{r['regression_ids']}" if r["regression_ids"] else ""
        print(f"  Round {r['round']:2d}: {bar} {r['accuracy']:5.1f}% "
              f"(+{r['facts_taught']} facts, "
              f"+{r['distinctions']} distinctions{reg})")
    print(f"  {'='*56}")
    print(f"  Total facts taught: {total_facts_taught}")
    print(f"  Total distinctions: {total_distinctions}")
    print(f"  Biology brain: {biology_brain.stats()['paths']} paths, "
          f"{biology_brain.stats()['neurons']} neurons")
    print()

    # Save learning curve
    curve_path = "benchmarks/learning_curve.json"
    with open(curve_path, "w") as f:
        json.dump(all_rounds, f, indent=2)
    print(f"  Learning curve saved to {curve_path}")

    vocab_brain.close()
    science_brain.close()
    biology_brain.close()


if __name__ == "__main__":
    main()
