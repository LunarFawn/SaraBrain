#!/usr/bin/env python3
"""MMLU High School Biology Benchmark — Sara Brain's sweet spot.

310 multiple-choice questions testing factual biology recall.
This tests what Sara actually does well: knowledge lookup, not
multi-step reasoning.

Usage:
    # Baseline (3B alone):
    python benchmarks/run_mmlu_biology.py --baseline

    # Sara + 3B:
    python benchmarks/run_mmlu_biology.py --db biology_brain.db

    # Both:
    python benchmarks/run_mmlu_biology.py --db biology_brain.db --compare
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
import urllib.error


def load_questions() -> list[dict]:
    """Load MMLU high school biology test set."""
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


def format_mc_prompt(question: str, choices: list[str]) -> str:
    labels = ['A', 'B', 'C', 'D']
    lines = [question, '']
    for i, choice in enumerate(choices):
        lines.append(f'{labels[i]}. {choice}')
    lines.append('')
    lines.append('Answer with ONLY the letter (A, B, C, or D). Nothing else.')
    return '\n'.join(lines)


def call_ollama(prompt: str, model: str, system: str,
                base_url: str) -> str:
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ],
        'stream': False,
        'options': {'temperature': 0},
    }
    url = f'{base_url}/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode('utf-8'))
            return body['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f'ERROR: {e}'


def extract_answer(response: str) -> str | None:
    response = response.strip().upper()
    if response in ('A', 'B', 'C', 'D'):
        return response
    for char in response:
        if char in 'ABCD':
            return char
    return None


def build_sara_system_prompt(brain, question: str) -> str:
    """Build a polymath-style prompt using Sara's graph for relevance."""
    import re
    from sara_brain.cortex.cleanup import STOPWORD_SUBJECTS

    words = re.findall(r"[a-z][a-z']+", question.lower())
    content_words = {
        w for w in words
        if len(w) > 2 and w not in STOPWORD_SUBJECTS
        and w not in {'the', 'a', 'an', 'will', 'shall'}
    }

    cursor = brain.conn.cursor()
    rows = cursor.execute(
        'SELECT source_text FROM paths WHERE source_text IS NOT NULL'
    ).fetchall()

    scored_facts = []
    for (source_text,) in rows:
        if not source_text:
            continue
        text_lower = source_text.lower()
        hits = sum(1 for w in content_words if w in text_lower)
        if hits >= 2:
            scored_facts.append((hits, source_text))

    scored_facts.sort(key=lambda x: x[0], reverse=True)
    top_facts = [text for _, text in scored_facts[:20]]

    if not top_facts:
        knowledge_section = 'No relevant knowledge available.'
    else:
        knowledge_section = '\n'.join(f'- {f}' for f in top_facts)

    return f"""\
You are a polymath answering a multiple-choice exam question.

You have access to the following verified knowledge. Use it to reason
about the answer. You may apply logic, deduction, and inference.

If the knowledge is insufficient, use your best reasoning on what
IS provided.

## Verified Knowledge
{knowledge_section}

## Instructions
- Read the question and all choices carefully
- Use the knowledge above to reason about the correct answer
- Answer with ONLY the letter (A, B, C, or D)"""


def run_benchmark(questions: list[dict], model: str, brain=None,
                  base_url: str = 'http://localhost:11434') -> dict:
    results = {
        'model': model,
        'mode': 'sara+llm' if brain else 'llm_only',
        'total': len(questions),
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'answers': [],
    }

    bench_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        prompt = format_mc_prompt(q['question'], q['choices'])

        if brain:
            system = build_sara_system_prompt(brain, q['question'])
        else:
            system = (
                'You are an expert answering a multiple-choice question. '
                'Answer with ONLY the letter (A, B, C, or D).'
            )

        response = call_ollama(prompt, model, system, base_url)
        answer = extract_answer(response)

        correct_letter = ['A', 'B', 'C', 'D'][q['answer_idx']]
        is_correct = answer == correct_letter

        if answer is None:
            results['errors'] += 1
        elif is_correct:
            results['correct'] += 1
        else:
            results['incorrect'] += 1

        results['answers'].append({
            'id': q['id'],
            'correct_letter': correct_letter,
            'model_answer': answer,
            'is_correct': is_correct,
        })

        q_elapsed = time.time() - q_start
        total_elapsed = time.time() - bench_start
        status = 'CORRECT' if is_correct else ('ERROR' if answer is None else 'WRONG')
        accuracy = results['correct'] / (i + 1) * 100
        avg = total_elapsed / (i + 1)
        remaining = avg * (len(questions) - i - 1)
        print(f'  [{i+1}/{len(questions)}] Q{q["id"]}: {status} '
              f'(got {answer}, correct {correct_letter}) — {accuracy:.1f}% — '
              f'{q_elapsed:.1f}s (~{remaining/60:.0f}m left)', flush=True)

    total_time = time.time() - bench_start
    results['accuracy'] = results['correct'] / results['total'] * 100
    results['total_time_sec'] = total_time
    return results


def print_summary(results: dict) -> None:
    print()
    print(f"  {'='*50}")
    print(f"  MMLU High School Biology — {results['mode']}")
    print(f"  Model: {results['model']}")
    print(f"  {'='*50}")
    print(f"  Total: {results['total']}")
    print(f"  Correct:   {results['correct']} ({results['accuracy']:.1f}%)")
    print(f"  Incorrect: {results['incorrect']}")
    print(f"  Errors:    {results['errors']}")
    print(f"  Time: {results['total_time_sec']/60:.1f} min")
    print(f"  {'='*50}")
    print()
    print(f'  Reference scores on MMLU biology (all MMLU):')
    print(f'    Random:           25.0%')
    print(f'    GPT-3.5:          ~70%')
    print(f'    GPT-4:            ~86%')
    print(f'    Claude Opus 4.5:  ~92%')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', help='Sara Brain database path')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--model', default='qwen2.5-coder:3b')
    parser.add_argument('--url', default='http://localhost:11434')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output')
    args = parser.parse_args()

    questions = load_questions()
    if args.limit > 0:
        questions = questions[:args.limit]

    print(f'\n  MMLU High School Biology Benchmark')
    print(f'  {len(questions)} questions, model: {args.model}\n')

    all_results = []

    if args.baseline or args.compare:
        print('  --- Baseline: 3B model alone ---\n')
        baseline = run_benchmark(questions, args.model, brain=None,
                                 base_url=args.url)
        print_summary(baseline)
        all_results.append(baseline)

    if args.db:
        from sara_brain.core.brain import Brain
        brain = Brain(args.db)
        stats = brain.stats()
        print(f'  --- Sara Brain + 3B ---')
        print(f'  Brain: {args.db} ({stats["neurons"]} neurons, {stats["paths"]} paths)\n')
        sara_results = run_benchmark(questions, args.model, brain=brain,
                                     base_url=args.url)
        print_summary(sara_results)
        all_results.append(sara_results)

    if args.compare and len(all_results) == 2:
        baseline, sara = all_results
        diff = sara['accuracy'] - baseline['accuracy']
        print(f"  {'='*50}")
        print(f'  COMPARISON')
        print(f"  {'='*50}")
        print(f"  3B alone:    {baseline['accuracy']:.1f}%")
        print(f"  Sara + 3B:   {sara['accuracy']:.1f}%")
        print(f"  Improvement: {diff:+.1f}%")
        print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'  Results saved to {args.output}')


if __name__ == '__main__':
    main()
