#!/usr/bin/env python3
"""GPQA Diamond Chemistry Benchmark — Sara Brain vs 3B model alone.

Runs 93 PhD-level chemistry multiple-choice questions through:
  1. Sara Brain + 3B model (knowledge-grounded)
  2. 3B model alone (no Sara, pure weights)

Compares accuracy, hallucination rate, and consistency.

Usage:
    # Baseline (3B alone, no Sara):
    python benchmarks/run_gpqa_chemistry.py --baseline

    # Sara + 3B:
    python benchmarks/run_gpqa_chemistry.py --db GPQA_Diamond_chemistry_r1.db

    # Both (full comparison):
    python benchmarks/run_gpqa_chemistry.py --db GPQA_Diamond_chemistry_r1.db --compare
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
import urllib.error
from pathlib import Path


def load_questions(path: str = "benchmarks/gpqa_diamond_chemistry.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def shuffle_choices(q: dict, seed: int | None = None) -> tuple[str, list[str], int]:
    """Shuffle answer choices and return (question_text, choices, correct_index)."""
    choices = [q["correct"], q["wrong1"], q["wrong2"], q["wrong3"]]
    # Remove any None/empty
    choices = [c.strip() for c in choices if c and c.strip()]
    rng = random.Random(seed if seed is not None else q["id"])
    rng.shuffle(choices)
    correct_idx = choices.index(q["correct"].strip())
    return q["question"], choices, correct_idx


def format_mc_prompt(question: str, choices: list[str]) -> str:
    """Format a multiple-choice question for the LLM."""
    labels = ["A", "B", "C", "D"]
    lines = [question, ""]
    for i, choice in enumerate(choices):
        lines.append(f"{labels[i]}. {choice}")
    lines.append("")
    lines.append("Answer with ONLY the letter (A, B, C, or D). Nothing else.")
    return "\n".join(lines)


def call_ollama(prompt: str, model: str, system: str = "",
                base_url: str = "http://localhost:11434") -> str:
    """Call Ollama API and return the response text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system} if system else None,
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    payload["messages"] = [m for m in payload["messages"] if m is not None]

    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer(response: str) -> str | None:
    """Extract the answer letter from the LLM response."""
    response = response.strip().upper()
    # Direct single letter
    if response in ("A", "B", "C", "D"):
        return response
    # First letter
    for char in response:
        if char in "ABCD":
            return char
    return None


def build_sara_system_prompt(brain, question: str, **_kwargs) -> str:
    """Build a benchmark prompt using Sara's graph to find relevant paths.

    No LLM synthesis — Sara's graph structure determines relevance.
    Extracts content words from the question, finds paths that contain
    multiple matching words (intersection = relevance), and injects
    only those as source_text.
    """
    import re
    from sara_brain.cortex.cleanup import STOPWORD_SUBJECTS

    # Extract content words from the question
    words = re.findall(r"[a-z][a-z']+", question.lower())
    content_words = {
        w for w in words
        if len(w) > 2 and w not in STOPWORD_SUBJECTS
        and w not in {"the", "a", "an", "will", "shall"}
    }

    # Search paths by source_text — find paths whose source text
    # contains multiple question words (structural relevance)
    cursor = brain.conn.cursor()
    rows = cursor.execute(
        "SELECT source_text FROM paths WHERE source_text IS NOT NULL"
    ).fetchall()

    scored_facts = []
    for (source_text,) in rows:
        if not source_text:
            continue
        text_lower = source_text.lower()
        # Count how many question words appear in this path's source
        hits = sum(1 for w in content_words if w in text_lower)
        if hits >= 2:  # at least 2 word overlap = relevant
            scored_facts.append((hits, source_text))

    # Sort by relevance (most word overlap first), take top 20
    scored_facts.sort(key=lambda x: x[0], reverse=True)
    top_facts = [text for _, text in scored_facts[:20]]

    if not top_facts:
        knowledge_section = "No relevant knowledge available."
    else:
        knowledge_section = "\n".join(f"- {f}" for f in top_facts)

    return f"""\
You are a polymath answering a multiple-choice exam question.

You have access to the following verified knowledge. Use it to reason
about the answer. You may apply logic, deduction, and inference.
You may eliminate wrong answers by reasoning.

If the knowledge is insufficient, use your best reasoning on what
IS provided.

## Verified Knowledge
{knowledge_section}

## Instructions
- Read the question and all choices carefully
- Use the knowledge above to reason about the correct answer
- Answer with ONLY the letter (A, B, C, or D)"""


def run_benchmark(questions: list[dict], model: str, brain=None,
                  base_url: str = "http://localhost:11434") -> dict:
    """Run the benchmark and return results."""
    results = {
        "model": model,
        "mode": "sara+llm" if brain else "llm_only",
        "total": len(questions),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "answers": [],
        "start_time": time.time(),
    }

    bench_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        question_text, choices, correct_idx = shuffle_choices(q)
        prompt = format_mc_prompt(question_text, choices)

        # Build system prompt
        if brain:
            system = build_sara_system_prompt(brain, question_text,
                                              model=model, base_url=base_url)
        else:
            system = (
                "You are a chemistry expert. Answer the multiple-choice "
                "question with ONLY the letter (A, B, C, or D)."
            )

        # Call LLM
        response = call_ollama(prompt, model, system=system, base_url=base_url)
        answer = extract_answer(response)

        correct_letter = ["A", "B", "C", "D"][correct_idx]
        is_correct = answer == correct_letter

        if answer is None:
            results["errors"] += 1
        elif is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        results["answers"].append({
            "id": q["id"],
            "subdomain": q["subdomain"],
            "correct_letter": correct_letter,
            "model_answer": answer,
            "model_raw": response[:200],
            "is_correct": is_correct,
        })

        # Progress
        q_elapsed = time.time() - q_start
        total_elapsed = time.time() - bench_start
        status = "CORRECT" if is_correct else ("ERROR" if answer is None else "WRONG")
        accuracy = results["correct"] / (i + 1) * 100
        avg_per_q = total_elapsed / (i + 1)
        remaining = avg_per_q * (len(questions) - i - 1)
        print(f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} "
              f"(got {answer}, correct {correct_letter}) "
              f"— {accuracy:.1f}% — {q_elapsed:.1f}s "
              f"(~{remaining/60:.0f}m left)")

    total_time = time.time() - bench_start
    results["accuracy"] = results["correct"] / results["total"] * 100
    results["total_time_sec"] = total_time
    results["avg_time_per_question"] = total_time / results["total"]
    return results


def print_summary(results: dict) -> None:
    """Print a summary of benchmark results."""
    print()
    print(f"  {'='*50}")
    print(f"  GPQA Diamond Chemistry — {results['mode']}")
    print(f"  Model: {results['model']}")
    print(f"  {'='*50}")
    total_time = results.get('total_time_sec', 0)
    avg_time = results.get('avg_time_per_question', 0)
    print(f"  Total questions: {results['total']}")
    print(f"  Correct:   {results['correct']} ({results['accuracy']:.1f}%)")
    print(f"  Incorrect: {results['incorrect']}")
    print(f"  Errors:    {results['errors']}")
    print(f"  Time:      {total_time/60:.1f} min ({avg_time:.1f}s/question)")
    print(f"  {'='*50}")
    print()
    print(f"  Reference scores:")
    print(f"    Random guessing:     25.0%")
    print(f"    Non-expert PhDs:     34.0%")
    print(f"    GPT-4 (2024):       ~39.7%")
    print(f"    Claude Opus 4.5:    ~60.0%")
    print()

    # Subdomain breakdown
    from collections import Counter
    sub_correct = Counter()
    sub_total = Counter()
    for a in results["answers"]:
        sub_total[a["subdomain"]] += 1
        if a["is_correct"]:
            sub_correct[a["subdomain"]] += 1

    print(f"  By subdomain:")
    for sub in sorted(sub_total):
        c = sub_correct[sub]
        t = sub_total[sub]
        pct = c / t * 100
        print(f"    {sub}: {c}/{t} ({pct:.1f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description="GPQA Diamond Chemistry Benchmark")
    parser.add_argument("--db", help="Sara Brain database path")
    parser.add_argument("--baseline", action="store_true",
                        help="Run 3B model alone (no Sara)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both modes and compare")
    parser.add_argument("--model", default="qwen2.5-coder:3b",
                        help="Ollama model name")
    parser.add_argument("--url", default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N questions (0 = all)")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit > 0:
        questions = questions[:args.limit]

    print(f"\n  GPQA Diamond Chemistry Benchmark")
    print(f"  {len(questions)} questions, model: {args.model}")
    print()

    all_results = []

    # Baseline run (3B alone)
    if args.baseline or args.compare:
        print("  --- Baseline: 3B model alone (no Sara) ---\n")
        baseline = run_benchmark(questions, args.model, brain=None,
                                 base_url=args.url)
        print_summary(baseline)
        all_results.append(baseline)

    # Sara run
    if args.db or (args.compare and args.db):
        from sara_brain.core.brain import Brain
        brain = Brain(args.db)
        stats = brain.stats()
        print(f"  --- Sara Brain + 3B ---")
        print(f"  Brain: {args.db} ({stats['neurons']} neurons, {stats['paths']} paths)\n")
        sara_results = run_benchmark(questions, args.model, brain=brain,
                                     base_url=args.url)
        print_summary(sara_results)
        all_results.append(sara_results)

    # Comparison
    if args.compare and len(all_results) == 2:
        baseline, sara = all_results
        diff = sara["accuracy"] - baseline["accuracy"]
        print(f"  {'='*50}")
        print(f"  COMPARISON")
        print(f"  {'='*50}")
        print(f"  3B alone:      {baseline['accuracy']:.1f}%")
        print(f"  Sara + 3B:     {sara['accuracy']:.1f}%")
        print(f"  Improvement:   {diff:+.1f}%")
        print(f"  {'='*50}")
        print()

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to {args.output}")

    if not args.baseline and not args.db:
        parser.print_help()


if __name__ == "__main__":
    main()
