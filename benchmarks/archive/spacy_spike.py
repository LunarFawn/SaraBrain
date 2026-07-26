"""Spike: compare spaCy's grammar parse against Sara's existing tokenizer
on a handful of biology facts and questions.

Purpose: decide whether spaCy is worth wiring into src/sara_brain/sensory/
as a grammar-extraction step (and whether it feeds the missing
temporal/sequence detector cleanly). Throwaway script — no brain writes,
no edits to sensory/parsing modules.

Run: .venv/bin/python benchmarks/spacy_spike.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import spacy

from sara_brain.sensory.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
FACTS = ROOT / "biology_facts.txt"
QUESTIONS = ROOT / "bio_10q_questions.json"

# Signals a temporal/sequence detector would look for.
TEMPORAL_PREPS = {"during", "after", "before", "while", "until", "since", "when", "as"}
CAUSAL_MARKERS = {"because", "since", "therefore", "thus", "so", "causes", "produces", "results"}


def temporal_signal(doc) -> dict:
    """What a temporal/sequence detector would extract from a spaCy parse."""
    preps = [t.text.lower() for t in doc
             if t.text.lower() in TEMPORAL_PREPS]
    causals = [t.text.lower() for t in doc
               if t.lemma_.lower() in CAUSAL_MARKERS]
    tenses = sorted({t.morph.get("Tense")[0] for t in doc
                     if t.pos_ == "VERB" and t.morph.get("Tense")})
    aspects = sorted({t.morph.get("Aspect")[0] for t in doc
                      if t.pos_ == "VERB" and t.morph.get("Aspect")})
    return {
        "temporal_preps": preps,
        "causal_markers": causals,
        "verb_tenses": tenses,
        "verb_aspects": aspects,
    }


def svo(doc) -> list[tuple[str, str, str]]:
    """Extract (subject, verb, object) triples from a spaCy dep parse."""
    triples = []
    for token in doc:
        if token.pos_ != "VERB" and token.dep_ != "ROOT":
            continue
        if token.pos_ not in {"VERB", "AUX"}:
            continue
        subj = next((c.text for c in token.children
                     if c.dep_ in {"nsubj", "nsubjpass"}), None)
        obj = next((c.text for c in token.children
                    if c.dep_ in {"dobj", "attr", "pobj", "acomp"}), None)
        if subj or obj:
            triples.append((subj or "?", token.lemma_, obj or "?"))
    return triples


def print_compare(label: str, text: str, nlp, sara_tok: Tokenizer) -> None:
    print(f"\n{'=' * 78}")
    print(f"{label}: {text}")
    print("=" * 78)

    # Sara's current tokenizer (no brain attached — just the pure tokenization).
    # Tokenizer needs a brain for phrase lookup; we fake it with a null brain
    # so we see at least the word-split + stopword behavior.
    words = [w for w in __import__("re").findall(r"[a-z0-9]+", text.lower())
             if w not in __import__("sara_brain.sensory.tokenizer",
                                    fromlist=["_STOP"])._STOP]
    print(f"\n[Sara tokenizer]  words-after-stop: {words}")

    # spaCy parse
    doc = nlp(text)
    print("\n[spaCy tokens]")
    for t in doc:
        print(f"  {t.text:<18} POS={t.pos_:<6} dep={t.dep_:<10} "
              f"lemma={t.lemma_:<14} head={t.head.text}")
    print(f"\n[spaCy SVO triples]   {svo(doc)}")
    print(f"[spaCy temporal/seq]  {temporal_signal(doc)}")
    ents = [(e.text, e.label_) for e in doc.ents]
    print(f"[spaCy entities]      {ents}")


def main() -> int:
    nlp = spacy.load("en_core_web_sm")
    sara_tok = Tokenizer(brain=None)  # type: ignore[arg-type]

    # Grab ~8 diverse facts (skip comments and section headers).
    facts = []
    for line in FACTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        facts.append(line)
        if len(facts) >= 8:
            break

    # Grab 3 questions.
    questions = json.loads(QUESTIONS.read_text())[:3]

    print("\n" + "#" * 78)
    print("# FACTS")
    print("#" * 78)
    for f in facts:
        print_compare("FACT", f, nlp, sara_tok)

    print("\n" + "#" * 78)
    print("# QUESTIONS")
    print("#" * 78)
    for q in questions:
        print_compare(f"Q{q['id']}", q["question"], nlp, sara_tok)

    return 0


if __name__ == "__main__":
    sys.exit(main())
