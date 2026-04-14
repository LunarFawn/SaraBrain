#!/usr/bin/env python3
"""Ingest chemistry chunks using Claude as the extractor — smarter cortex.

The 3B model made extraction errors like "methane has 4 carbon atoms".
Claude reads the same sources and extracts cleanly.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python benchmarks/ingest_chemistry_claude.py --db phd_chemistry_v2.db
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.error


DEFAULT_MODEL = "claude-sonnet-4-6"  # fast + smart extractor


def call_claude(prompt: str, system: str, model: str,
                api_key: str) -> str | None:
    """Call Anthropic API and return the response text."""
    payload = {
        "model": model,
        "max_tokens": 2000,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["content"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        print(f"  API error {e.code}: {err[:200]}", flush=True)
        return None
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        return None


def parse_statements(raw: str) -> list[str]:
    """Parse LLM output into clean statement lines."""
    statements = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*•·0123456789.)")
        cleaned = cleaned.strip()
        if not cleaned:
            continue
        if cleaned.upper() == "NONE":
            continue
        if len(cleaned) > 200:
            continue
        if any(kw in cleaned.lower() for kw in ("http", "www", "doi:")):
            continue
        statements.append(cleaned)
    # Deduplicate
    seen = set()
    result = []
    for s in statements:
        low = s.lower()
        if low not in seen:
            seen.add(low)
            result.append(s)
    return result


SYSTEM_PROMPT = """You are a careful fact extractor for a knowledge graph.

You read chemistry text and extract ONLY factually correct, verifiable statements. Your output will be stored permanently as ground truth, so accuracy matters more than completeness.

Rules:
- Each statement must be factually correct as written. If unsure, omit it.
- One fact per line. Simple format: <subject> is/has/requires <property>.
- Use simple subject-predicate-object sentences. No compound sentences.
- Include specific numbers, dates, named reactions, mechanisms.
- Do NOT extract: author names, DOIs, citation markers, bibliography entries, URLs, "see also" references.
- Do NOT extract: vague statements like "there are many types" without specifics.
- Do NOT extract: anything you cannot verify from the text provided.
- If a passage is bibliography/references/navigation, output NONE.
- If you find nothing factually precise, output NONE.

Simple words only. No explanations, no commentary. Just facts."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Sara Brain database path")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Anthropic model name")
    parser.add_argument("--chunks-dir", default="benchmarks/chemistry_chunks")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.api_key:
        print("ERROR: Set ANTHROPIC_API_KEY or pass --api-key")
        return 1

    from sara_brain.core.brain import Brain

    brain = Brain(args.db)
    stats = brain.stats()
    print(f"\n  Brain: {args.db}")
    print(f"  Before: {stats['neurons']} neurons, {stats['paths']} paths")
    print(f"  Extractor: {args.model}\n")

    chunks = sorted(
        f for f in os.listdir(args.chunks_dir) if f.endswith(".txt")
    )
    if args.limit > 0:
        chunks = chunks[:args.limit]
    total = len(chunks)
    print(f"  {total} chunks to ingest\n")

    progress_file = args.db + ".claude_progress"
    already_done = set()
    if os.path.exists(progress_file):
        with open(progress_file) as pf:
            already_done = set(line.strip() for line in pf if line.strip())
        print(f"  Resuming — {len(already_done)} chunks already done\n")

    total_facts = 0
    start = time.time()

    for i, filename in enumerate(chunks):
        if filename in already_done:
            continue

        text = open(os.path.join(args.chunks_dir, filename)).read()
        if not text.strip():
            continue

        raw = call_claude(text, SYSTEM_PROMPT, args.model, args.api_key)
        if raw is None:
            print(f"  [{i+1}/{total}] {filename}: API failure", flush=True)
            continue

        statements = parse_statements(raw)
        facts = 0
        for stmt in statements:
            result = brain.teach(stmt)
            if result is not None:
                facts += 1
        brain.conn.commit()

        total_facts += facts
        elapsed = time.time() - start
        done = i + 1 - len(already_done)
        rate = done / elapsed * 60 if elapsed > 0 else 0
        remaining = (total - i - 1) / rate if rate > 0 else 0

        print(f"  [{i+1}/{total}] {filename}: "
              f"+{facts} facts (total: {total_facts}, "
              f"{rate:.1f}/min, ~{remaining:.0f}m left)", flush=True)

        with open(progress_file, "a") as pf:
            pf.write(filename + "\n")

    elapsed = time.time() - start
    stats = brain.stats()
    print(f"\n  Done in {elapsed/60:.1f} minutes")
    print(f"  Facts learned: {total_facts}")
    print(f"  Brain now: {stats['neurons']} neurons, {stats['paths']} paths")
    brain.close()
    return 0


if __name__ == "__main__":
    exit(main())
