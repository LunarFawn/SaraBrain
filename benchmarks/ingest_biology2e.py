#!/usr/bin/env python3
"""Teach Sara the entire Biology 2e (OpenStax) textbook.

Pipeline:
  1. Read the pdftotext output of Biology2e-WEB.pdf.
  2. Split into chapters using the textbook's "CHAPTER N" markers.
  3. Per chapter: walk sentences, keep only clean declarative facts
     (not captions, not figure refs, not questions, not bullet lists
     of learning objectives), ALL via pure spaCy + heuristic — no LLM.
  4. Write one facts file per chapter, then teach each into its own
     region using batch_teach.py's Learner.

Fact selection rules (no LLM involved):
  - Sentence must end with a period.
  - Length between 3 and 30 words (short fragments and long
    complex sentences both parse badly).
  - Sentence must contain at least one NOUN/PROPN subject and one
    NOUN/PROPN/VERB content in the rest.
  - Discard if starts with "Figure", "Table", "See", "Note:", a digit
    followed by period ("1.", "10.2", etc.).
  - Discard if contains "?" (questions) or "!" (exclamations).
  - Discard if it's a learning-objective bullet (starts with
    "Describe", "Distinguish", "Explain", "Identify" at sentence
    start — these are imperatives, not facts).

Usage:
    .venv/bin/python benchmarks/ingest_biology2e.py \\
        --source /Users/grizzlyengineer/repo/training_material/Biology2e-WEB.txt \\
        --db biology2e.db \\
        --facts-dir benchmarks/biology2e_facts
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import spacy

from sara_brain.core.brain import Brain


_CHAPTER_RE = re.compile(r"^\s*CHAPTER\s+(\d+)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^\s*(\d+)\.(\d+)\s+(.+?)\s*$", re.MULTILINE)
_FIGURE_START = re.compile(r"^\s*(FIGURE|TABLE|Figure|Table)\b", re.IGNORECASE)
_IMPERATIVE_START = re.compile(
    r"^\s*(Describe|Distinguish|Explain|Identify|Compare|Contrast|Define|"
    r"List|Name|State|Show|Discuss|Summarize|Explore)\b",
    re.IGNORECASE,
)
_LEADING_DIGIT = re.compile(r"^\s*\d+(\.\d+)*\.?\s")
_FIG_REF_INLINE = re.compile(r"\(?\bFigure\s+\d+\.\d+", re.IGNORECASE)
_CREDIT_START = re.compile(r"^\s*(credit|source|adapted|modified)\b", re.IGNORECASE)


def split_chapters(text: str) -> list[tuple[int, str, str]]:
    """Return list of (chapter_number, title_guess, body_text).

    Uses 'CHAPTER N' markers. Title is best-effort from the line
    after the marker.
    """
    chapters: list[tuple[int, str, str]] = []
    matches = list(_CHAPTER_RE.finditer(text))
    for i, m in enumerate(matches):
        n = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        # Title: first non-empty line after the CHAPTER marker
        title = ""
        for line in body.splitlines():
            line = line.strip()
            if line:
                title = line[:60]
                break
        chapters.append((n, title, body))
    return chapters


def is_fact_candidate(sent_text: str) -> bool:
    """Pure-heuristic fact filter. No LLM."""
    s = sent_text.strip()
    if not s:
        return False
    if not s.endswith("."):
        return False
    if "?" in s or "!" in s:
        return False
    n_words = len(s.split())
    if n_words < 3 or n_words > 30:
        return False
    if _FIGURE_START.match(s):
        return False
    if _IMPERATIVE_START.match(s):
        return False
    if _LEADING_DIGIT.match(s):
        return False
    if _FIG_REF_INLINE.search(s):
        return False
    if _CREDIT_START.match(s):
        return False
    # Must contain at least one alphabetic word of length >= 3
    if not any(len(w) >= 3 and w.isalpha() for w in s.split()):
        return False
    return True


_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_BRACKET_REF_RE = re.compile(r"\s*\[[^]]*\]")
_INLINE_FIG_REF = re.compile(
    r"\s*\(?\b(?:see\s+)?(?:Figure|Table)\s+\d+\.\d+\)?\b",
    re.IGNORECASE,
)


def simplify_for_parser(sent_text: str) -> list[str]:
    """Rephrase a textbook sentence into Sara-parser-friendly forms.

    The teacher's craft: strip parenthetical asides, remove inline
    figure references, and split compound sentences on ";" or
    ", and" boundaries so each resulting piece is a simple
    subject-verb-object statement Sara's parser can accept.

    Returns a list because one textbook sentence may yield multiple
    taught facts. Always returns the simplified SINGLE form too so
    nothing is lost.
    """
    s = sent_text.strip()
    s = _PAREN_RE.sub("", s)
    s = _BRACKET_REF_RE.sub("", s)
    s = _INLINE_FIG_REF.sub("", s)
    # Normalise curly quotes to straight — some parsers stumble on these.
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []

    variants: list[str] = [s]
    # Split compound sentences on semicolons — each side is usually
    # its own independent clause.
    if ";" in s:
        parts = [p.strip(" .") for p in s.split(";") if p.strip(" .")]
        for p in parts:
            if p and p not in variants:
                variants.append(p + "." if not p.endswith(".") else p)
    # Split on ", and" / ", or" / ", while" — the comma + coordinator
    # often separates two independent clauses.
    for connector in (", and ", ", or ", ", while ", ", whereas "):
        if connector in s:
            parts = s.split(connector)
            for p in parts:
                p = p.strip(" .")
                if p and p not in variants:
                    variants.append(p + "." if not p.endswith(".") else p)
    return variants


def _spacy_is_declarative(sent) -> bool:
    """Grammar filter via spaCy. Every ROOT verb/aux with an nsubj
    and at least one content complement is declarative. Rejects pure
    fragments, imperatives, and headers that reached here."""
    root = None
    for tok in sent:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return False
    if root.pos_ not in {"VERB", "AUX", "NOUN", "ADJ"}:
        return False
    nsubj = None
    for child in root.children:
        if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {
            "NOUN", "PROPN", "PRON"
        }:
            nsubj = child
            break
    if nsubj is None:
        return False
    return any(
        c.dep_ in {"dobj", "attr", "acomp", "prep", "pobj", "xcomp",
                   "ccomp", "agent", "advmod"}
        for c in root.children
    )


def extract_facts_from_chapter(body: str, nlp) -> list[str]:
    """Walk chapter body, split into sentences via spaCy, simplify each
    sentence into parser-friendly variants, then keep variants that
    spaCy agrees are declarative. ALL declarative content is taught;
    the only filtering is rejection of non-facts (headers, captions,
    fragments) and simplification for the parser — not selection.
    """
    facts: list[str] = []
    seen: set[str] = set()
    paragraphs = re.split(r"\n\s*\n", body)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para = re.sub(r"\s+", " ", para)
        try:
            doc = nlp(para)
        except Exception:
            continue
        for sent in doc.sents:
            s = sent.text.strip()
            if not is_fact_candidate(s):
                continue
            # Present each simplified variant to spaCy for grammar
            # judgment; keep the ones that pass.
            for variant in simplify_for_parser(s):
                if not is_fact_candidate(variant):
                    continue
                try:
                    v_doc = nlp(variant)
                except Exception:
                    continue
                if not all(_spacy_is_declarative(sn) for sn in v_doc.sents):
                    continue
                key = variant.lower()
                if key in seen:
                    continue
                seen.add(key)
                facts.append(variant)
    return facts


def region_name_for_chapter(n: int) -> str:
    return f"ch{n}"


def write_facts_file(path: Path, chapter_num: int, title: str,
                     facts: list[str]) -> None:
    lines = [
        f"# Biology2e Chapter {chapter_num} — {title}",
        f"# Auto-extracted from Biology2e-WEB.txt via ingest_biology2e.py",
        f"# Fact count: {len(facts)}",
        "",
    ]
    lines.extend(facts)
    path.write_text("\n".join(lines))


def teach_chapter(brain: Brain, region: str, facts: list[str],
                  verbose: bool = False) -> tuple[int, int, int]:
    """Teach a list of facts into a compartmentalised region.

    Returns (taught, skipped_parser, failed).
    """
    from sara_brain.storage.neuron_repo import NeuronRepo
    from sara_brain.storage.segment_repo import SegmentRepo
    from sara_brain.storage.path_repo import PathRepo
    from sara_brain.core.learner import Learner
    from sara_brain.parsing.statement_parser import StatementParser
    from sara_brain.parsing.taxonomy import Taxonomy

    brain.db.create_region(region)
    nr = NeuronRepo(brain.conn, prefix=region)
    sr = SegmentRepo(brain.conn, prefix=region)
    pr = PathRepo(brain.conn, prefix=region)
    parser = StatementParser(taxonomy=Taxonomy())
    learner = Learner(parser, nr, sr, pr)

    taught = skipped = failed = 0
    for fact in facts:
        try:
            result = learner.learn(fact, apply_filter=False)
        except Exception:
            failed += 1
            continue
        if result is None:
            skipped += 1
        else:
            taught += 1
    brain.conn.commit()
    return taught, skipped, failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Path to Biology2e-WEB.txt (pdftotext output).")
    ap.add_argument("--db", required=True,
                    help="Target brain DB path.")
    ap.add_argument("--facts-dir", required=True,
                    help="Directory to write per-chapter facts files.")
    ap.add_argument("--chapters",
                    help="Comma-separated list of chapter numbers to ingest "
                         "(e.g. '1,2,3' or '10,11'). Default: all.")
    ap.add_argument("--extract-only", action="store_true",
                    help="Only extract facts to files; do NOT teach.")
    args = ap.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Source not found: {source_path}")
        return 1
    facts_dir = Path(args.facts_dir)
    facts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading spaCy…")
    nlp = spacy.load("en_core_web_sm")
    # Disable components we don't need — speeds up the parse considerably.
    # But sentence segmentation needs the parser, so keep it on.

    print(f"Reading {source_path} ({source_path.stat().st_size // 1024} KB)…")
    text = source_path.read_text(errors="replace")

    print("Splitting chapters…")
    chapters = split_chapters(text)
    print(f"Found {len(chapters)} chapter markers.")

    selected: set[int] | None = None
    if args.chapters:
        selected = {int(c.strip()) for c in args.chapters.split(",")}

    brain = None
    if not args.extract_only:
        brain = Brain(args.db)

    total_facts = 0
    total_taught = 0
    total_skipped = 0
    total_failed = 0

    for (n, title, body) in chapters:
        if selected is not None and n not in selected:
            continue
        print(f"\nChapter {n}: {title}")
        facts = extract_facts_from_chapter(body, nlp)
        total_facts += len(facts)
        out_path = facts_dir / f"ch{n:02d}_facts.txt"
        write_facts_file(out_path, n, title, facts)
        print(f"  Extracted: {len(facts)} facts → {out_path.name}")

        if args.extract_only or brain is None:
            continue

        region = region_name_for_chapter(n)
        taught, skipped, failed = teach_chapter(brain, region, facts)
        total_taught += taught
        total_skipped += skipped
        total_failed += failed
        print(f"  Taught {taught}, parser-skipped {skipped}, "
              f"failed {failed} into region {region}")

    if brain is not None:
        brain.close()

    print(f"\n{'=' * 60}")
    print(f"  Chapters processed: "
          f"{len(selected) if selected else len(chapters)}")
    print(f"  Candidate facts extracted: {total_facts}")
    if not args.extract_only:
        print(f"  Taught: {total_taught}")
        print(f"  Parser-skipped: {total_skipped}")
        print(f"  Failed: {total_failed}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
