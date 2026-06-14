#!/usr/bin/env python3
"""MMLU Biology benchmark for the 115M Reader Pipeline.

Uses the SaraPipeline (115M Extractor + Wavefront + 115M Synthesizer)
to answer questions without any external LLM.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add scripts to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from sara_pipeline import SaraPipeline

def load_questions() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    questions = []
    for i, q in enumerate(ds):
        questions.append({
            'id': i,
            'question': q['question'],
            'choices': q['choices'],
            'answer_idx': q['answer'],
        })
    return questions

def extract_letter(response: str) -> str | None:
    # Synthesizer returns prose. We need to see if it mentions one of the choices or just the answer text.
    # For now, we'll look for the first A, B, C, or D if it's isolated.
    response = response.strip()
    if len(response) == 1 and response in "ABCD":
        return response
    # Often it might say "The answer is A."
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1)
    return None

def main():
    import argparse
    import re
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--limit", type=int, default=33)
    ap.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    args = ap.parse_args()

    print(f"\n  MMLU 115M Pipeline Benchmark")
    print(f"  Brain: {args.db}, Device: {args.device}\n")

    pipeline = SaraPipeline(args.db, device=args.device)
    questions = load_questions()[:args.limit]

    correct = 0
    t0 = time.time()

    for i, q in enumerate(questions):
        # We need the choices in the question for the pipeline
        labels = ["A", "B", "C", "D"]
        full_q = q["question"] + "\n" + "\n".join([f"{labels[j]}. {q['choices'][j]}" for j in range(4)])
        
        answer_prose = pipeline.ask(full_q)
        got = extract_letter(answer_prose)
        
        correct_letter = labels[q["answer_idx"]]
        is_correct = (got == correct_letter)
        if is_correct:
            correct += 1
        
        status = "CORRECT" if is_correct else "WRONG"
        print(f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} (got {got}, correct {correct_letter})")
        # print(f"    Prose: {answer_prose}")

    elapsed = time.time() - t0
    print(f"\n  Final Score: {correct}/{len(questions)} ({correct/len(questions)*100:.1f}%)")
    print(f"  Total Time: {elapsed:.1f}s")

if __name__ == "__main__":
    main()
