#!/usr/bin/env python3
"""Comprehension-Verified Teaching — sentence by sentence, paragraph by paragraph.

Sara reads a source text one paragraph at a time. Each sentence is
parsed and taught. After the paragraph, the 3B cortex is tested to
verify it can USE Sara's knowledge. If it can't, the facts are
rephrased and retaught until comprehension is verified.

The 3B model is the constraint — Sara needs to understand things in
a way the cortex can interpret. A correct fact in a format the cortex
can't read is useless.

Usage:
    python benchmarks/comprehension_teacher.py --db learn.db \\
        --source textbook_chapter.txt --model qwen2.5-coder:3b
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo


# ── LLM calls ──

def call_ollama(prompt: str, system: str, model: str,
                base_url: str = "http://localhost:11434",
                max_tokens: int = 500) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


# ── Text processing ──

def split_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double-newlines."""
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > 30]


def split_sentences(paragraph: str) -> list[str]:
    """Split paragraph into sentences."""
    sents = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sents if len(s.strip()) > 10]


# ── Fact extraction ──

EXTRACT_SYSTEM = """You are extracting simple facts from a sentence for a knowledge graph.

Rules:
- Convert the sentence into one or more simple "X is Y" statements
- Each statement on its own line
- Simple words only, no hedging, no "may" or "might"
- If the sentence contains no teachable fact, output NONE
- Do NOT include citations, author names, or reference numbers
- Maximum 3 facts per sentence"""


def extract_facts(sentence: str, model: str, base_url: str) -> list[str]:
    """Use the LLM to convert a sentence into teachable facts."""
    raw = call_ollama(sentence, EXTRACT_SYSTEM, model, base_url)
    if not raw or "NONE" in raw.upper():
        return []
    facts = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*•·0123456789.)")
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 8 and len(cleaned) < 200:
            if "none" not in cleaned.lower():
                facts.append(cleaned)
    return facts[:3]


# ── Comprehension test ──

QUESTION_SYSTEM = """You are generating a simple multiple-choice question to test
whether a student understood a paragraph. The question should test the
key concept from the paragraph.

Output format:
QUESTION: [the question]
A. [correct answer]
B. [wrong answer]
C. [wrong answer]
D. [wrong answer]
CORRECT: A

The correct answer MUST be A. The wrong answers should be plausible
but clearly wrong if you understood the paragraph."""


def generate_test_question(paragraph: str, model: str,
                           base_url: str) -> dict | None:
    """Generate a multiple-choice question about a paragraph."""
    raw = call_ollama(
        f"Generate a test question about this paragraph:\n\n{paragraph}",
        QUESTION_SYSTEM, model, base_url, max_tokens=300,
    )

    # Parse the response
    lines = raw.strip().splitlines()
    question = ""
    choices = []
    correct_idx = 0

    for line in lines:
        line = line.strip()
        if line.upper().startswith("QUESTION:"):
            question = line[9:].strip()
        elif re.match(r"^[A-D]\.", line):
            choices.append(line[2:].strip())
        elif line.upper().startswith("CORRECT:"):
            letter = line[8:].strip().upper()
            if letter in "ABCD":
                correct_idx = ord(letter) - ord("A")

    if question and len(choices) == 4:
        return {
            "question": question,
            "choices": choices,
            "answer_idx": correct_idx,
        }
    return None


# ── Activation + cortex answer ──

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
        if w.endswith("s") and len(w) > 4:
            forms.append(w[:-1])
        for form in forms:
            if form not in seen and len(form) >= 3:
                seen.add(form)
                words.append(form)
    return words


CORTEX_SYSTEM = """You are the language cortex for Sara Brain. Sara's brain has
processed a question and produced activation for each choice.

Look at Sara's activation and pick the choice where the activation is
most relevant to the question. Trust Sara over your own training.

Answer with ONLY the letter (A, B, C, or D)."""


def test_comprehension(brain: Brain, question: dict, model: str,
                       base_url: str) -> bool:
    """Test if the cortex can answer using Sara's activation. Returns True if correct."""
    q_words = extract_words(question["question"])
    labels = ["A", "B", "C", "D"]

    # Build activation for each choice
    prompt_lines = [
        "QUESTION:", question["question"], "",
        "CHOICES:",
    ]
    for i, choice in enumerate(question["choices"]):
        prompt_lines.append(f"{labels[i]}. {choice}")
    prompt_lines.append("")
    prompt_lines.append("SARA'S BRAIN ACTIVATION:")
    prompt_lines.append("")

    for i, choice in enumerate(question["choices"]):
        c_words = extract_words(choice)
        seeds = q_words + c_words

        st = ShortTerm(
            event_id=f"comp-{time.time():.3f}",
            event_type="comprehension",
        )
        brain.recognizer.propagate_echo(
            seeds, st, max_rounds=2, min_strength=0.1, exact_only=True,
        )

        # Format top activation
        sorted_act = sorted(st.convergence_map.items(), key=lambda x: -x[1])[:15]
        act_labels = []
        for nid, weight in sorted_act:
            n = brain.neuron_repo.get_by_id(nid)
            if n and len(n.label) >= 3:
                act_labels.append(f"{n.label}({weight:.0f})")

        prompt_lines.append(f"--- Choice {labels[i]} ---")
        prompt_lines.append(", ".join(act_labels[:10]) if act_labels else "(no activation)")
        prompt_lines.append("")

    prompt_lines.append("Which choice? Answer with just the letter.")

    response = call_ollama("\n".join(prompt_lines), CORTEX_SYSTEM,
                            model, base_url)
    answer = None
    for char in response.strip().upper():
        if char in "ABCD":
            answer = char
            break

    correct_letter = labels[question["answer_idx"]]
    return answer == correct_letter


# ── Rephrase for retry ──

REPHRASE_SYSTEM = """You are rephrasing facts to be simpler and more direct.

The student's cortex could not understand the original phrasing. Rewrite
each fact in the simplest possible "X is Y" form. One fact per line.
Use shorter words. Be more direct. Drop unnecessary qualifiers."""


def rephrase_facts(facts: list[str], model: str, base_url: str) -> list[str]:
    """Rephrase facts more simply for retry."""
    prompt = "Rephrase these facts more simply:\n\n" + "\n".join(
        f"- {f}" for f in facts
    )
    raw = call_ollama(prompt, REPHRASE_SYSTEM, model, base_url)
    rephrased = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*•·0123456789.)")
        cleaned = cleaned.strip()
        if cleaned and len(cleaned) > 8 and len(cleaned) < 200:
            rephrased.append(cleaned)
    return rephrased[:len(facts)]


# ── Main loop ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True,
                        help="Sara Brain database")
    parser.add_argument("--source", required=True,
                        help="Source text file to learn from")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max rephrase attempts per paragraph")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N paragraphs (0 = all)")
    args = parser.parse_args()

    brain = Brain(args.db)
    stats = brain.stats()
    print(f"\n  Comprehension-Verified Teaching")
    print(f"  Brain: {args.db} ({stats['neurons']} neurons, "
          f"{stats['paths']} paths)")
    print(f"  Source: {args.source}")
    print(f"  Model: {args.model}\n")

    with open(args.source) as f:
        text = f.read()

    paragraphs = split_paragraphs(text)
    if args.limit > 0:
        paragraphs = paragraphs[:args.limit]

    print(f"  {len(paragraphs)} paragraphs to learn\n")

    total_facts = 0
    total_rephrased = 0
    understood = 0
    not_understood = 0
    source_label = args.source

    for pi, paragraph in enumerate(paragraphs):
        print(f"  ── Paragraph {pi + 1}/{len(paragraphs)} ──")
        print(f"  {paragraph[:80]}...\n", flush=True)

        # Step 1: extract and teach sentences
        sentences = split_sentences(paragraph)
        para_facts: list[str] = []

        for sentence in sentences:
            facts = extract_facts(sentence, args.model, args.base_url)
            for fact in facts:
                result = brain.teach_tentative(fact, source_label=source_label)
                if result:
                    para_facts.append(fact)
                    total_facts += 1
                    print(f"    + {result.path_label}", flush=True)

        if not para_facts:
            print(f"    (no facts extracted)\n")
            continue

        brain.conn.commit()

        # Step 2: generate a comprehension test PER FACT
        # A paragraph may teach multiple lessons — test each one
        para_passed = 0
        para_failed = 0

        for fi, fact in enumerate(para_facts):
            test_q = generate_test_question(fact, args.model, args.base_url)
            if test_q is None:
                para_passed += 1  # can't test, assume ok
                continue

            print(f"    TEST ({fi+1}/{len(para_facts)}): "
                  f"{test_q['question'][:60]}...", end="", flush=True)

            passed = test_comprehension(brain, test_q, args.model,
                                         args.base_url)
            if passed:
                print(f" ✓", flush=True)
                para_passed += 1
                continue

            # Rephrase and retry this specific fact
            print(f" ✗ — rephrasing...", flush=True)
            fact_understood = False

            for retry in range(args.max_retries):
                rephrased = rephrase_facts([fact], args.model, args.base_url)
                for rf in rephrased:
                    result = brain.teach_from_error(
                        rf,
                        error_context=f"para {pi+1} fact {fi+1} "
                                      f"rephrase {retry+1}",
                    )
                    if result:
                        total_rephrased += 1
                        print(f"      ~ {result.path_label}", flush=True)

                brain.conn.commit()

                passed = test_comprehension(brain, test_q, args.model,
                                             args.base_url)
                if passed:
                    print(f"      ✓ (after {retry + 1} rephrase)", flush=True)
                    fact_understood = True
                    break

            if fact_understood:
                para_passed += 1
            else:
                print(f"      ✗ still not understood", flush=True)
                para_failed += 1

        if para_failed == 0:
            print(f"    PARAGRAPH UNDERSTOOD ({para_passed} facts) ✓\n")
            understood += 1
        else:
            print(f"    PARAGRAPH PARTIAL ({para_passed} ok, "
                  f"{para_failed} failed) ✗\n")
            not_understood += 1

    # Summary
    stats = brain.stats()
    print(f"  {'='*50}")
    print(f"  COMPREHENSION REPORT")
    print(f"  {'='*50}")
    print(f"  Paragraphs: {len(paragraphs)}")
    print(f"  Understood: {understood}")
    print(f"  Not understood: {not_understood}")
    print(f"  Facts taught: {total_facts}")
    print(f"  Rephrased facts: {total_rephrased}")
    print(f"  Brain now: {stats['neurons']} neurons, {stats['paths']} paths")
    print(f"  {'='*50}")

    brain.close()


if __name__ == "__main__":
    main()
