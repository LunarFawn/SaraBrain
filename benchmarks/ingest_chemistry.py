#!/usr/bin/env python3
"""Ingest all chemistry chunk files into a Sara Brain — drip by drip.

Two-pass learning:
  Pass 1: Cold read — extract what the LLM can from each chunk
  Pass 2: Reinforcement — re-read each chunk with Sara's learned
           topics as context, catching facts missed the first time

Usage:
    python benchmarks/ingest_chemistry.py --db GPQA_Diamond_chemistry_r1.db
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request


def _call_ollama(prompt: str, system: str, model: str,
                 base_url: str) -> str | None:
    """Direct Ollama call for the reinforcement pass."""
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
    headers = {"Content-Type": "application/json"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def pass1_ingest(brain, chunks_dir: str, chunks: list[str],
                 progress_file: str) -> int:
    """Pass 1: Cold read — extract facts from each chunk."""
    already_done = set()
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            already_done = set(line.strip() for line in pf if line.strip())
        print(f"  Resuming pass 1 — {len(already_done)} chunks already done\n")

    total = len(chunks)
    total_facts = 0
    start_time = time.time()

    for i, filename in enumerate(chunks):
        if filename in already_done:
            continue

        filepath = os.path.join(chunks_dir, filename)
        text = open(filepath).read()
        if not text.strip():
            continue

        try:
            result = brain.ingest(text, source=filename)
            facts = result.total_taught
            total_facts += facts
            elapsed = time.time() - start_time
            done = i + 1 - len(already_done)
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0

            print(f"  [{i+1}/{total}] {filename}: "
                  f"+{facts} facts (total: {total_facts}, "
                  f"{rate:.1f}/min, ~{remaining:.0f}m left)", flush=True)
            with open(progress_file, "a") as pf:
                pf.write(filename + "\n")
        except Exception as e:
            print(f"  [{i+1}/{total}] {filename}: ERROR — {e}", flush=True)

    return total_facts


def pass2_reinforce(brain, chunks_dir: str, chunks: list[str],
                    progress_file: str, model: str, base_url: str) -> int:
    """Pass 2: Re-read chunks with Sara's learned topics as context.

    Sara now knows concepts from pass 1. We tell the LLM what Sara
    already knows and ask it to find MORE facts about those topics
    that it missed the first time.
    """
    already_done = set()
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            already_done = set(line.strip() for line in pf if line.strip())
        print(f"  Resuming pass 2 — {len(already_done)} chunks already done\n")

    # Get Sara's current concept labels for context
    all_neurons = brain.neuron_repo.list_all()
    concepts = [n.label for n in all_neurons
                if n.neuron_type.value == "concept"][:50]
    concept_list = ", ".join(concepts)

    from sara_brain.innate.primitives import get_structural
    primitives = ", ".join(sorted(get_structural()))

    system = (
        "You are a reading cortex doing a SECOND pass on a document. "
        "The brain already knows about these topics:\n"
        f"  {concept_list}\n\n"
        "Read the text again and extract facts you missed the first time. "
        "Focus on:\n"
        "- Specific numbers, dates, measurements\n"
        "- Relationships between known concepts\n"
        "- Properties and mechanisms\n"
        "- Conditions under which reactions occur\n"
        "One fact per line. Format: <subject> is/has/requires <property>\n"
        "Do not skip dates, numbers, or names.\n"
        "If you find nothing new, say NONE.\n"
        "Simple words only. No explanations."
    )

    total = len(chunks)
    total_facts = 0
    start_time = time.time()

    for i, filename in enumerate(chunks):
        if filename in already_done:
            continue

        filepath = os.path.join(chunks_dir, filename)
        text = open(filepath).read()
        if not text.strip():
            continue

        try:
            raw = _call_ollama(text, system, model, base_url)
            if raw is None or raw.strip().upper() == "NONE":
                print(f"  [{i+1}/{total}] {filename}: no new facts", flush=True)
                with open(progress_file, "a") as pf:
                    pf.write(filename + "\n")
                continue

            # Parse and teach each new fact
            facts = 0
            for line in raw.splitlines():
                cleaned = line.strip().lstrip("-*•·0123456789.)")
                cleaned = cleaned.strip()
                if not cleaned or cleaned.upper() == "NONE" or len(cleaned) > 200:
                    continue
                result = brain.teach(cleaned)
                if result is not None:
                    facts += 1
            brain.conn.commit()

            total_facts += facts
            elapsed = time.time() - start_time
            done = i + 1 - len(already_done)
            rate = done / elapsed * 60 if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0

            print(f"  [{i+1}/{total}] {filename}: "
                  f"+{facts} new facts (total: {total_facts}, "
                  f"{rate:.1f}/min, ~{remaining:.0f}m left)", flush=True)
            with open(progress_file, "a") as pf:
                pf.write(filename + "\n")
        except Exception as e:
            print(f"  [{i+1}/{total}] {filename}: ERROR — {e}", flush=True)

    return total_facts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Sara Brain database path")
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--chunks-dir", default="benchmarks/chemistry_chunks")
    parser.add_argument("--pass2-only", action="store_true",
                        help="Skip pass 1, only do reinforcement")
    args = parser.parse_args()

    from sara_brain.core.brain import Brain

    brain = Brain(args.db)
    stats = brain.stats()
    print(f"\n  Brain: {args.db}")
    print(f"  Before: {stats['neurons']} neurons, {stats['paths']} paths")
    print()

    chunks = sorted(
        f for f in os.listdir(args.chunks_dir) if f.endswith(".txt")
    )
    print(f"  {len(chunks)} chunk files\n")

    # Configure LLM if not already set
    if not brain.settings_repo.get("llm_model"):
        brain.settings_repo.set("llm_provider", "ollama")
        brain.settings_repo.set("llm_model", args.model)
        brain.settings_repo.set("llm_api_url", args.url)
        brain.conn.commit()

    p1_progress = args.db + ".p1_progress"
    p2_progress = args.db + ".p2_progress"

    if not args.pass2_only:
        # ── Pass 1: Cold read ──
        print("  ═══ PASS 1: Cold read ═══\n")
        p1_facts = pass1_ingest(brain, args.chunks_dir, chunks, p1_progress)
        stats = brain.stats()
        print(f"\n  Pass 1 done: +{p1_facts} facts")
        print(f"  Brain: {stats['neurons']} neurons, {stats['paths']} paths\n")

    # ── Pass 2: Reinforcement ──
    print("  ═══ PASS 2: Reinforcement with learned topics ═══\n")
    p2_facts = pass2_reinforce(brain, args.chunks_dir, chunks, p2_progress,
                                args.model, args.url)

    stats = brain.stats()
    print(f"\n  Pass 2 done: +{p2_facts} new facts")
    print(f"  Final brain: {stats['neurons']} neurons, {stats['paths']} paths")
    brain.close()


if __name__ == "__main__":
    main()
