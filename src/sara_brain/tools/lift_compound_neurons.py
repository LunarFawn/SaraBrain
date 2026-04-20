#!/usr/bin/env python3
"""Lift flat multi-word neurons into an IS-A hierarchy.

For each 2-3 word neuron label in brain.db, run spaCy's dependency
parser. If the label is a clean noun-noun compound (modifier →
head, both NOUN/PROPN/ADJ), create an IS-A segment from the
compound to its head. Create the head neuron if absent.

This is a post-migration sweep that gives `nerve_cell —is_a→ cell`,
`daughter_cell —is_a→ cell`, `red_blood_cell —is_a→ blood_cell` etc.
without re-ingesting source texts.

Usage:
    .venv/bin/python -m sara_brain.tools.lift_compound_neurons --db brain.db
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import spacy

IS_A = "is_a"


_ACCEPTABLE_POS = {"NOUN", "PROPN", "ADJ"}
_SKIP_SUFFIXES = ("_attribute",)


def _is_candidate(label: str) -> bool:
    if not label:
        return False
    if any(label.endswith(s) for s in _SKIP_SUFFIXES):
        return False
    if any(ch in label for ch in '.,;:"()[]{}!?/'):
        return False
    toks = label.split()
    if not (2 <= len(toks) <= 3):
        return False
    for t in toks:
        stripped = t.replace("-", "").replace("'", "")
        if not stripped.isalpha():
            return False
    return True


def _parse_compound(nlp, label: str) -> str | None:
    """Return the head noun if label is a clean noun compound, else None."""
    doc = nlp(label)
    toks = [t for t in doc if not t.is_space]
    if len(toks) < 2:
        return None
    # All tokens must be NOUN / PROPN / ADJ
    if not all(t.pos_ in _ACCEPTABLE_POS for t in toks):
        return None
    # Head must be the rightmost token and a NOUN/PROPN
    head = toks[-1]
    if head.pos_ not in {"NOUN", "PROPN"}:
        return None
    # Modifier(s) should attach to the head via compound/amod dep
    for mod in toks[:-1]:
        if mod.dep_ not in {"compound", "amod", "nmod", "ROOT"}:
            return None
    # Reject if head equals whole label (single token case — guarded
    # by length check above anyway)
    head_lemma = head.lemma_.lower().strip()
    if not head_lemma or head_lemma == label.strip().lower():
        return None
    return head_lemma


def lift(db_path: Path, dry_run: bool = False) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    cur = conn.cursor()

    # Load spaCy (used elsewhere in project — en_core_web_sm)
    print("Loading spaCy…")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # Build label → neuron_id map
    label_to_id: dict[str, int] = {}
    for nid, label in cur.execute("SELECT id, label FROM neurons").fetchall():
        label_to_id[label.strip().lower()] = nid

    # Candidates
    candidates = [
        (nid, label) for label, nid in label_to_id.items()
        if _is_candidate(label)
    ]
    print(f"Total neurons: {len(label_to_id)}")
    print(f"Compound candidates: {len(candidates)}")

    created_head_neurons = 0
    created_is_a_edges = 0
    skipped_non_compound = 0
    skipped_self_loop = 0
    already_linked = 0

    t0 = time.time()
    # Batch-process in spaCy pipe for speed
    labels = [label for _, label in candidates]
    ids = [nid for nid, _ in candidates]
    for i, doc in enumerate(nlp.pipe(labels, batch_size=500)):
        compound_id = ids[i]
        label = labels[i]
        toks = [t for t in doc if not t.is_space]
        if len(toks) < 2:
            skipped_non_compound += 1
            continue
        if not all(t.pos_ in _ACCEPTABLE_POS for t in toks):
            skipped_non_compound += 1
            continue
        head = toks[-1]
        if head.pos_ not in {"NOUN", "PROPN"}:
            skipped_non_compound += 1
            continue
        modifiers_ok = all(
            mod.dep_ in {"compound", "amod", "nmod", "ROOT"}
            for mod in toks[:-1]
        )
        if not modifiers_ok:
            skipped_non_compound += 1
            continue
        head_lemma = head.lemma_.lower().strip()
        if not head_lemma or head_lemma == label.strip().lower():
            skipped_self_loop += 1
            continue

        # Get or create head neuron
        head_id = label_to_id.get(head_lemma)
        if head_id is None:
            if dry_run:
                head_id = -1
            else:
                cur.execute(
                    "INSERT INTO neurons (label, neuron_type, created_at) "
                    "VALUES (?, ?, ?)",
                    (head_lemma, "concept", time.time()),
                )
                head_id = cur.lastrowid
                label_to_id[head_lemma] = head_id
                created_head_neurons += 1

        if head_id == compound_id:
            skipped_self_loop += 1
            continue

        # Add IS-A edge (compound → head); UNIQUE dedups
        if dry_run:
            created_is_a_edges += 1
            continue
        try:
            cur.execute(
                "INSERT INTO segments (source_id, target_id, relation, "
                "strength, created_at) VALUES (?, ?, ?, ?, ?)",
                (compound_id, head_id, IS_A, 1.0, time.time()),
            )
            created_is_a_edges += 1
        except sqlite3.IntegrityError:
            already_linked += 1

        if (i + 1) % 1000 == 0:
            conn.commit()

    conn.commit()

    print()
    print(f"  spaCy-parsed:              {len(candidates)}")
    print(f"  skipped non-compound:      {skipped_non_compound}")
    print(f"  skipped self-loop:         {skipped_self_loop}")
    print(f"  new head neurons:          {created_head_neurons}")
    print(f"  IS-A edges created:        {created_is_a_edges}")
    print(f"  already linked:            {already_linked}")
    print(f"  elapsed:                   {time.time()-t0:.1f}s")

    if not dry_run:
        # Index for fast IS-A filtering at query time
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_seg_relation "
            "ON segments(relation)"
        )
        conn.commit()
    conn.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    lift(args.db, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
