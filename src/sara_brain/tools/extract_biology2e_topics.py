#!/usr/bin/env python3
"""Extract topic candidates for Biology 2e into the hierarchical brain.

Topics come from two deterministic sources, merged by slug:

1. **Noun lemmas from the per-chapter fact logs.** Lemmatise every
   fact with spaCy, count NOUN/PROPN lemmas, keep those appearing in
   ≥ MIN distinct facts. This surfaces real topics like 'mitosis',
   'meiosis', 'photosynthesis', 'heterotroph', 'chromosome' — the
   nouns Sara has actually seen repeatedly, not parser noise.

2. **Numbered section headings from the raw textbook TOC.** Lines of
   the form '10.2 Mitosis' in Biology2e-WEB.txt become coarse topics
   that catch facts not pulled into any noun-driven concept.

Trigger lemmas for each topic are its content lemmas (what you'd
say to mean 'this is about X'). These go into concept_lemmas and
are what `route_teach` matches against at teach time.

The output is written directly into `brain_root/` via
HierarchicalBackend. The tool does NOT move any facts — that's
`migrate_to_hierarchy.py`'s job. Separating topic discovery from
fact migration keeps each step inspectable.

Usage:
    .venv/bin/python -m sara_brain.tools.extract_biology2e_topics \\
        --facts-dir benchmarks/biology2e_facts \\
        --toc /path/to/Biology2e-WEB.txt \\
        --subject biology \\
        --dest brain_root/ \\
        --min-freq 5
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import spacy

from sara_brain.storage.hierarchical_backend import (
    HierarchicalBackend, slugify_concept,
)


# Content-lemma filter (roughly — we don't load spaCy just for lemmas
# here; tokens are used as-is since concept labels are already
# normalised by the parser).
_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "for", "with", "from", "by",
    "at", "on", "as", "about", "into", "onto", "upon", "within",
    "without", "through", "between", "among", "and", "or", "but",
    "that", "which", "who", "whom", "whose", "this", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "may", "might", "must", "should", "would", "could",
    "will", "shall", "not", "no",
})


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_lemmas(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in _STOPWORDS and len(t) >= 3]


def _parse_toc_headings(toc_path: Path) -> list[tuple[str, str]]:
    """Return [(chapter_num, title)] pairs for 'N.M Title' lines.

    'N.M' is the section number; trailing page numbers are stripped.
    """
    out: list[tuple[str, str]] = []
    section_re = re.compile(r"^(\d+\.\d+)\s+(.+?)\s{2,}\d+\s*$")
    alt_re = re.compile(r"^(\d+\.\d+)\s+(.+)$")
    for line in toc_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines():
        m = section_re.match(line)
        if m is None:
            m = alt_re.match(line)
        if m is None:
            continue
        num, title = m.group(1), m.group(2).strip()
        # Title must have at least one letter and be reasonable length
        if not re.search(r"[A-Za-z]", title):
            continue
        if len(title) > 120 or len(title) < 3:
            continue
        out.append((num, title))
    return out


def _mine_noun_topics(facts_dir: Path, min_freq: int) -> list[dict]:
    """Extract noun-lemma topic candidates from the per-chapter fact
    logs. Each candidate carries its lemma (used as both slug root
    and trigger) plus the count of distinct facts that mention it.

    Multi-word noun phrases (compound nouns) are kept as-is — spaCy's
    noun_chunks give us phrases like 'cell cycle', 'dna replication',
    'mitotic spindle'. Single-word noun lemmas are kept for the
    classic one-word topics ('mitosis', 'photosynthesis',
    'heterotroph').
    """
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Count distinct facts per candidate (not raw token frequency), so
    # a word mentioned 50 times in one fact doesn't dominate.
    facts_per_lemma: dict[str, int] = Counter()
    # Track cooccurring lemmas per candidate for trigger-lemma seeding.
    cooccurs: dict[str, Counter] = defaultdict(Counter)

    fact_files = sorted(facts_dir.glob("ch*_facts.txt"))
    total_facts = 0
    for i, fp in enumerate(fact_files, 1):
        text = fp.read_text(encoding="utf-8", errors="replace")
        # Drop comment header
        facts = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        total_facts += len(facts)
        # spaCy pipe for speed
        for doc in nlp.pipe(facts, batch_size=64):
            this_fact: set[str] = set()
            # 1. Single-noun lemmas
            for tok in doc:
                if tok.pos_ not in ("NOUN", "PROPN"):
                    continue
                if tok.is_stop:
                    continue
                lemma = tok.lemma_.lower().strip()
                if len(lemma) < 4:
                    continue
                if not re.match(r"^[a-z][a-z0-9\-]*$", lemma):
                    continue
                this_fact.add(lemma)
            # 2. Noun-chunk phrases (head + its compounds/amods)
            for nc in doc.noun_chunks:
                head = nc.root
                if head.pos_ not in ("NOUN", "PROPN"):
                    continue
                # Build lowercased lemma phrase from compound + head
                words: list[str] = []
                for tok in nc:
                    if tok.is_stop or tok.is_punct:
                        continue
                    if tok.dep_ not in ("compound", "amod", "nmod", "nn"):
                        if tok is head:
                            words.append(tok.lemma_.lower())
                        continue
                    words.append(tok.lemma_.lower())
                if head.lemma_.lower() not in words:
                    words.append(head.lemma_.lower())
                phrase = " ".join(w for w in words if w).strip()
                if " " not in phrase:
                    continue  # single word already covered
                if len(phrase) > 60 or len(phrase) < 5:
                    continue
                if not re.match(r"^[a-z][a-z0-9 \-]*$", phrase):
                    continue
                this_fact.add(phrase)
            for cand in this_fact:
                facts_per_lemma[cand] += 1
                for other in this_fact:
                    if other != cand:
                        cooccurs[cand][other] += 1
        print(f"  processed {i}/{len(fact_files)} ({fp.name}): "
              f"{len(facts_per_lemma)} candidates so far")

    print(f"  total facts seen: {total_facts}")
    topics: list[dict] = []
    for cand, freq in facts_per_lemma.items():
        if freq < min_freq:
            continue
        # Top cooccurring lemmas — capped — form extra triggers
        extras = [w for w, _c in cooccurs[cand].most_common(8)
                  if _c >= min_freq]
        lemmas = _content_lemmas(cand) + extras
        lemmas = list(dict.fromkeys(lemmas))[:10]  # dedupe, cap
        slug = slugify_concept(cand)
        if not slug or slug == "unnamed":
            continue
        topics.append({
            "label": cand,
            "slug": slug,
            "fact_count": freq,
            "lemmas": lemmas,
        })
    topics.sort(key=lambda t: -t["fact_count"])
    return topics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts-dir", required=True,
                    help="Directory of per-chapter fact logs "
                         "(benchmarks/biology2e_facts).")
    ap.add_argument("--toc", default=None,
                    help="Optional raw Biology2e-WEB.txt to mine "
                         "section headings.")
    ap.add_argument("--subject", default="biology",
                    help="Subject name under brain_root/subjects/.")
    ap.add_argument("--dest", required=True,
                    help="brain_root/ directory.")
    ap.add_argument("--min-freq", type=int, default=5,
                    help="Minimum distinct-fact frequency for a "
                         "noun/noun-phrase to become a topic.")
    ap.add_argument("--description", default="",
                    help="Subject description (free text).")
    args = ap.parse_args()

    facts_dir = Path(args.facts_dir)
    if not facts_dir.exists():
        print(f"error: facts dir not found: {facts_dir}", file=sys.stderr)
        return 2

    print(f"Mining noun topics from {facts_dir} "
          f"(min_freq={args.min_freq}) …")
    mined = _mine_noun_topics(facts_dir, args.min_freq)
    print(f"  {len(mined)} noun-topic candidates")

    toc_topics: list[tuple[str, str]] = []
    if args.toc:
        toc_path = Path(args.toc)
        if toc_path.exists():
            print(f"Parsing TOC section headings from {toc_path} …")
            toc_topics = _parse_toc_headings(toc_path)
            print(f"  {len(toc_topics)} section headings found")
        else:
            print(f"  warning: TOC path not found, skipping: {toc_path}")

    backend = HierarchicalBackend(args.dest)
    backend.register_subject(args.subject, description=args.description)

    # Register noun-topic concepts first (they carry fact-frequency
    # evidence directly)
    concept_slugs: set[str] = set()
    for t in mined:
        slug = backend.register_concept(
            subject=args.subject,
            concept=t["label"],
            source_kind="noun_topic",
            description=(
                f"Seen in {t['fact_count']} facts across fact logs"
            ),
            trigger_lemmas=t["lemmas"] or [t["slug"]],
        )
        concept_slugs.add(slug)

    # Then section headings — skip ones that already exist as a slug
    # (the concept_neuron version is strictly more specific)
    toc_added = 0
    for num, title in toc_topics:
        slug_candidate = slugify_concept(title)
        if slug_candidate in concept_slugs:
            continue
        lemmas = _content_lemmas(title) or [slug_candidate]
        slug = backend.register_concept(
            subject=args.subject,
            concept=title,
            source_kind="section_header",
            description=f"Section {num}: {title}",
            trigger_lemmas=lemmas,
        )
        concept_slugs.add(slug)
        toc_added += 1

    # Unclassified bucket — guarantees fact migration never drops
    backend.register_concept(
        subject=args.subject,
        concept="_unclassified",
        source_kind="system_bucket",
        description="Facts that matched no topic trigger at migration "
                    "time. Inspect to tune topic triggers.",
        trigger_lemmas=[],
    )

    backend.commit()
    backend.close()

    total = len(concept_slugs) + 1  # +1 for _unclassified
    print()
    print(f"Subject: {args.subject}")
    print(f"  noun topics:                 {len(mined)}")
    print(f"  section-header topics added: {toc_added}")
    print(f"  _unclassified bucket:        1")
    print(f"  total concepts:              {total}")
    print()
    print(f"Brain root: {args.dest}")
    print(f"  brain.db             → subjects index")
    print(f"  subjects/{args.subject}.db   → concepts + triggers + bridges")
    print(f"  concepts/{args.subject}/*.db → per-topic storage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
