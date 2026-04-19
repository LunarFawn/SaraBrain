#!/usr/bin/env python3
"""Two-Mode Study Tool — Memorize then Recite, through Short-Term Memory.

Pass 1 (Memorize): Read paragraphs → extract facts → store in short-term
Pass 2 (Recite): Compare short-term to original text → verify → fix
Consolidate: Only verified facts reach long-term. Bad stuff decays.

Like a student who reads with a highlighter (pass 1), closes the book
and writes what they remember (pass 2), then checks their notes (consolidate).
Bad notes get thrown out. Good notes go in the binder.

Usage:
    python benchmarks/study_tool.py --db layer_biology.db \\
        --source training_material/ch10_cell_reproduction.txt \\
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.parsing.statement_parser import StatementParser
from sara_brain.parsing.taxonomy import Taxonomy


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
        "options": {"temperature": 0, "num_predict": 500},
    }
    url = f"{base_url}/v1/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"].strip()


# ── Text ──

def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]


def split_sentences(paragraph: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", paragraph)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def extract_concepts(text: str) -> list[str]:
    """Pull content words from text for concept matching."""
    stops = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "them", "than", "then", "more", "most", "also", "only",
        "each", "both", "some", "many", "such", "very", "just", "into",
        "your", "will", "would", "could", "should", "which", "what",
        "when", "where", "about", "above", "below", "these", "those",
        "and", "for", "the", "are", "not", "but", "its", "can",
    }
    words = re.findall(r"[a-z][a-z'-]+", text.lower())
    return [w for w in words if len(w) >= 4 and w not in stops]


# ── Fact extraction ──

EXTRACT_SYSTEM = """You are extracting facts from a sentence for a knowledge graph.

Rules:
- Convert into simple statements. Formats accepted:
  "X is Y" (attributes)
  "X causes Y" (causation)
  "X precedes Y" or "X follows Y" (sequence)
  "X prevents Y" (prevention)
  "X contains Y" or "X includes Y" (composition)
  "X requires Y" (dependency)
- One fact per line
- Simple words only
- No citations, author names, or reference numbers
- If no teachable fact exists, output NONE
- Maximum 3 facts per sentence"""


def extract_facts(sentence: str, model: str, base_url: str) -> list[str]:
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


# ── Observation (fact + metadata in short-term) ──

class Observation:
    """A fact held in short-term memory with its parse result and status."""
    def __init__(self, raw_fact: str, source_sentence: str,
                 paragraph_idx: int):
        self.raw_fact = raw_fact
        self.source_sentence = source_sentence
        self.paragraph_idx = paragraph_idx
        self.parsed = None       # ParsedStatement or None
        self.status = "pending"  # pending, verified, mangled, rejected
        self.subject = ""
        self.obj = ""
        self.relation = ""

    def __repr__(self):
        return f"[{self.status}] {self.raw_fact[:60]}"


# ── Pass 1: Memorize ──

def pass1_memorize(paragraphs: list[str], parser: StatementParser,
                   model: str, base_url: str) -> list[Observation]:
    """Read paragraphs, extract facts, store in short-term as observations."""
    observations: list[Observation] = []
    total_sentences = 0
    total_extracted = 0
    total_parsed = 0
    total_rejected = 0

    for pi, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        total_sentences += len(sentences)
        print(f"  ── Paragraph {pi+1}/{len(paragraphs)} "
              f"({len(sentences)} sentences) ──", flush=True)

        for sentence in sentences:
            facts = extract_facts(sentence, model, base_url)
            total_extracted += len(facts)

            for fact in facts:
                obs = Observation(fact, sentence, pi)

                # Try to parse
                parsed = parser.parse(fact)
                if parsed is None:
                    obs.status = "rejected"
                    total_rejected += 1
                    print(f"    ✗ REJECT: {fact[:60]}", flush=True)
                    observations.append(obs)
                    continue

                obs.parsed = parsed
                obs.subject = parsed.subject
                obs.obj = parsed.obj
                obs.relation = parsed.relation

                # Retention check: is the label clean?
                if len(obs.subject) < 2 or len(obs.obj) < 2:
                    obs.status = "mangled"
                    print(f"    ~ MANGLED: {fact[:60]}", flush=True)
                elif obs.subject.endswith("_attribute"):
                    obs.status = "mangled"
                    print(f"    ~ MANGLED: {fact[:60]}", flush=True)
                else:
                    obs.status = "pending"
                    total_parsed += 1
                    print(f"    + {obs.subject} → {obs.relation} → {obs.obj}",
                          flush=True)

                observations.append(obs)

    print(f"\n  Pass 1 summary: {total_sentences} sentences, "
          f"{total_extracted} facts extracted, {total_parsed} parsed, "
          f"{total_rejected} rejected\n", flush=True)
    return observations


# ── Pass 2: Recite ──

def pass2_recite(paragraphs: list[str], observations: list[Observation],
                 model: str, base_url: str) -> list[Observation]:
    """Compare observations against original text. Verify and fix."""
    verified = 0
    missed = 0
    fixed = 0

    for pi, paragraph in enumerate(paragraphs):
        para_obs = [o for o in observations
                    if o.paragraph_idx == pi and o.status == "pending"]
        concepts = extract_concepts(paragraph)

        print(f"  ── Recite paragraph {pi+1}/{len(paragraphs)} "
              f"({len(para_obs)} observations, {len(concepts)} concepts) ──",
              flush=True)

        # Check each observation: does its subject/object appear in
        # the original paragraph?
        for obs in para_obs:
            subj_in_para = obs.subject in paragraph.lower()
            obj_words = obs.obj.split()
            # At least half the object words should appear in paragraph
            obj_hits = sum(1 for w in obj_words
                          if w.lower() in paragraph.lower())
            obj_coverage = obj_hits / len(obj_words) if obj_words else 0

            if subj_in_para and obj_coverage >= 0.3:
                obs.status = "verified"
                verified += 1
                print(f"    ✓ {obs.subject} → {obs.relation} → {obs.obj[:40]}",
                      flush=True)
            else:
                obs.status = "mangled"
                print(f"    ~ MISMATCH: '{obs.subject}' "
                      f"(in para: {subj_in_para}, "
                      f"obj coverage: {obj_coverage:.0%})", flush=True)

        # Check for concepts in the paragraph that have NO observations
        covered_concepts = set()
        for obs in observations:
            if obs.paragraph_idx == pi and obs.status == "verified":
                covered_concepts.update(obs.subject.split())
                covered_concepts.update(obs.obj.split()[:3])

        missing_concepts = [c for c in concepts
                            if c not in covered_concepts and len(c) >= 5]
        if missing_concepts[:3]:
            print(f"    ? MISSING concepts: {missing_concepts[:5]}",
                  flush=True)
            # Try to extract facts about missing concepts
            for concept in missing_concepts[:3]:
                # Find the sentence mentioning this concept
                for sent in split_sentences(paragraph):
                    if concept in sent.lower():
                        new_facts = extract_facts(sent, model, base_url)
                        for fact in new_facts:
                            new_obs = Observation(fact, sent, pi)
                            from sara_brain.parsing.statement_parser import StatementParser
                            from sara_brain.parsing.taxonomy import Taxonomy
                            p = StatementParser(Taxonomy())
                            parsed = p.parse(fact)
                            if parsed:
                                new_obs.parsed = parsed
                                new_obs.subject = parsed.subject
                                new_obs.obj = parsed.obj
                                new_obs.relation = parsed.relation
                                new_obs.status = "verified"
                                observations.append(new_obs)
                                fixed += 1
                                print(f"    + RECOVERED: {new_obs.subject} → "
                                      f"{new_obs.relation} → {new_obs.obj[:40]}",
                                      flush=True)
                        break  # only try first matching sentence

    print(f"\n  Pass 2 summary: {verified} verified, "
          f"{fixed} recovered, "
          f"{sum(1 for o in observations if o.status == 'mangled')} mangled\n",
          flush=True)
    return observations


# ── Consolidate ──

def consolidate(brain: Brain, observations: list[Observation],
                source_label: str) -> int:
    """Write verified observations to long-term. Discard the rest."""
    consolidated = 0
    discarded = 0

    for obs in observations:
        if obs.status != "verified":
            discarded += 1
            continue

        result = brain.teach_tentative(
            obs.raw_fact, source_label=source_label
        )
        if result:
            consolidated += 1
        else:
            discarded += 1

    brain.conn.commit()
    return consolidated


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:3b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N paragraphs")
    args = parser.parse_args()

    brain = Brain(args.db)
    stmt_parser = StatementParser(Taxonomy())
    stats = brain.stats()

    print(f"\n  ╔{'═'*50}╗")
    print(f"  ║  Study Tool — Memorize → Recite → Consolidate  ║")
    print(f"  ╚{'═'*50}╝")
    print(f"  Brain: {args.db} ({stats['neurons']} neurons, "
          f"{stats['paths']} paths)")
    print(f"  Source: {args.source}\n")

    with open(args.source) as f:
        text = f.read()

    paragraphs = split_paragraphs(text)
    if args.limit > 0:
        paragraphs = paragraphs[:args.limit]
    print(f"  {len(paragraphs)} paragraphs to study\n")

    # ── Pass 1: Memorize ──
    print("  ═══ PASS 1: MEMORIZE (into short-term) ═══\n")
    start = time.time()
    observations = pass1_memorize(
        paragraphs, stmt_parser, args.model, args.base_url
    )

    # ── Pass 2: Recite ──
    print("  ═══ PASS 2: RECITE (verify against original) ═══\n")
    observations = pass2_recite(
        paragraphs, observations, args.model, args.base_url
    )

    # ── Consolidate ──
    print("  ═══ CONSOLIDATE (verified → long-term) ═══\n")
    source_label = args.source
    consolidated = consolidate(brain, observations, source_label)
    discarded = len(observations) - consolidated

    elapsed = time.time() - start
    stats = brain.stats()

    # ── Report ──
    status_counts = {}
    for obs in observations:
        status_counts[obs.status] = status_counts.get(obs.status, 0) + 1

    print(f"  {'='*50}")
    print(f"  STUDY REPORT")
    print(f"  {'='*50}")
    print(f"  Paragraphs studied:   {len(paragraphs)}")
    print(f"  Total observations:   {len(observations)}")
    for status, count in sorted(status_counts.items()):
        print(f"    {status:20s}: {count}")
    print(f"  Consolidated to LT:   {consolidated}")
    print(f"  Discarded:            {discarded}")
    print(f"  Time:                 {elapsed/60:.1f} min")
    print(f"  Brain now:            {stats['neurons']} neurons, "
          f"{stats['paths']} paths")
    print(f"  {'='*50}\n")

    brain.close()


if __name__ == "__main__":
    main()
