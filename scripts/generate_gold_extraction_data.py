#!/usr/bin/env python3
"""Generate gold-standard multi-triple extraction training data using a local LLM.

Uses llama3.1:8b to extract multiple (subject, relation, object) triples
from real biology sentences. The output is formatted for training the 115M
extractor model.

The key insight: complex sentences contain multiple facts. Training the
extractor on single-triple examples teaches it to stop after one.
This script generates multi-triple targets from rich sentences.

Usage:
    python scripts/generate_gold_extraction_data.py \
        --out training_data/extractor_gold_multitiple.jsonl \
        --num-sentences 2000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

SYSTEM_PROMPT = """You are a fact extractor. Given a biology sentence, extract ALL factual relationships as triples.

Rules:
- Output format: subject | relation | object (one per line)
- Subjects and objects must be 1-3 words (noun phrases only)
- Relations must be one of: is_a, contains, produces, requires, involves, causes, prevents, occurs_in, part_of, enables, interacts_with, transforms_into, regulates, provides, activates, inhibits, depends_on, results_in, leads_to, attaches_to, composed_of
- Extract ALL facts from the sentence (usually 2-4 per sentence)
- Only extract facts explicitly stated in the sentence
- Do NOT include articles (a, an, the) in subjects or objects
- Do NOT include verbs as subjects or objects — only nouns/noun phrases

Example input: "The mitochondria produces ATP through oxidative phosphorylation in eukaryotic cells."
Example output:
mitochondria | produces | ATP
mitochondria | occurs_in | eukaryotic cells
oxidative phosphorylation | produces | ATP
ATP | results_in | oxidative phosphorylation"""

VALID_RELATIONS = {
    "is_a", "contains", "produces", "requires", "involves", "causes",
    "prevents", "occurs_in", "part_of", "enables", "interacts_with",
    "transforms_into", "regulates", "provides", "activates", "inhibits",
    "depends_on", "results_in", "leads_to", "attaches_to", "composed_of",
}


def call_ollama(sentence: str, model: str = "llama3.1:8b",
                base_url: str = "http://localhost:11434") -> str:
    """Call Ollama to extract triples from a sentence."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all facts from: \"{sentence}\""},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_triples(llm_output: str) -> list[tuple[str, str, str]]:
    """Parse LLM output into structured triples."""
    triples = []
    for line in llm_output.strip().split("\n"):
        line = line.strip()
        if "|" not in line:
            continue
        parts = [p.strip().lower() for p in line.split("|")]
        if len(parts) != 3:
            continue
        subj, rel, obj = parts
        # Validate relation
        rel = rel.replace(" ", "_")
        if rel not in VALID_RELATIONS:
            continue
        # Validate subject/object (1-3 words, no articles)
        subj = re.sub(r'^(the|a|an)\s+', '', subj).strip()
        obj = re.sub(r'^(the|a|an)\s+', '', obj).strip()
        if not subj or not obj or len(subj.split()) > 4 or len(obj.split()) > 4:
            continue
        if subj == obj:
            continue
        triples.append((subj, rel, obj))
    return triples


def format_output(triples: list[tuple[str, str, str]]) -> str:
    """Format triples into t_start/t_rel/t_obj/t_end format."""
    parts = []
    for subj, rel, obj in triples:
        parts.append(f"t_start {subj} t_rel {rel} t_obj {obj} t_end")
    return "\n".join(parts)


def load_biology_sentences() -> list[str]:
    """Load rich biology sentences (prefer longer ones with multiple facts)."""
    sentences = []
    bio_dir = Path("data/biology_english")
    if not bio_dir.exists():
        print("ERROR: data/biology_english not found", file=sys.stderr)
        sys.exit(1)

    for f in sorted(bio_dir.glob("ch*_facts.txt")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sent in re.split(r'(?<=[.!?])\s+', line):
                sent = sent.strip()
                # Prefer longer sentences (more likely to have multiple facts)
                if 40 < len(sent) < 300:
                    sentences.append(sent)

    return sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--num-sentences", type=int, default=2000)
    ap.add_argument("--model", default="llama3.1:8b")
    ap.add_argument("--url", default="http://localhost:11434")
    args = ap.parse_args()

    sentences = load_biology_sentences()
    print(f"Loaded {len(sentences)} sentences.", file=sys.stderr)

    # Shuffle and take the requested number
    import random
    rng = random.Random(2026)
    rng.shuffle(sentences)
    sentences = sentences[:args.num_sentences]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    generated = 0
    skipped = 0
    total_triples = 0
    t0 = time.time()

    for i, sent in enumerate(sentences):
        llm_output = call_ollama(sent, model=args.model, base_url=args.url)
        if llm_output.startswith("ERROR:"):
            skipped += 1
            continue

        triples = parse_triples(llm_output)
        if not triples:
            skipped += 1
            continue

        output = format_output(triples)
        example = {"paragraph": sent, "output": output}
        out_f.write(json.dumps(example) + "\n")
        generated += 1
        total_triples += len(triples)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (args.num_sentences - i - 1) / rate
            avg_triples = total_triples / generated if generated else 0
            print(f"  [{i+1}/{args.num_sentences}] generated={generated} "
                  f"skipped={skipped} avg_triples={avg_triples:.1f} "
                  f"({elapsed:.0f}s, ~{remaining/60:.0f}m left)",
                  file=sys.stderr)

    out_f.close()
    elapsed = time.time() - t0
    avg_triples = total_triples / generated if generated else 0
    print(f"\nDone in {elapsed:.0f}s. Generated {generated} examples "
          f"({total_triples} triples, avg {avg_triples:.1f}/sentence). "
          f"Skipped {skipped}. Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
