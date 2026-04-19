#!/usr/bin/env python3
"""Migrate a monolithic Sara DB into the hierarchical per-concept layout.

For every path in the source DB:
  1. Compute its content lemmas.
  2. Call backend.route_teach() — returns all concept slugs whose
     trigger lemmas intersect the path's lemmas.
  3. Write the path (neurons → segments → path → path_steps →
     segment_sources) into each returned concept DB, translating
     IDs to the target DB's ID namespace.
  4. Paths with no concept hit go to _unclassified.
  5. After all paths are written, rebuild concept_vocab for every
     concept that received facts.
  6. Carry cross-region bridges from the source DB to the subject DB.

Each fact may live in multiple concept DBs — overlap is expected.
IDs are local to each concept DB and are never shared across files.

Usage:
    .venv/bin/python -m sara_brain.tools.migrate_to_hierarchy \\
        --source biology2e.db \\
        --subject biology \\
        --dest brain_root/ \\
        [--verify]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from sara_brain.storage.hierarchical_backend import HierarchicalBackend

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "to", "for", "with", "from", "by",
    "at", "on", "as", "into", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "have", "has", "do", "does", "not", "no",
    "its", "their", "this", "that", "these", "those",
})

_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_lemmas(text: str) -> list[str]:
    return [
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and len(w) >= 3
    ]


def _path_lemmas(path_row, step_labels: list[tuple[str, str]]) -> list[str]:
    """Collect content lemmas from a path's source_text + all step
    neuron labels. More signal for routing than source_text alone."""
    words: set[str] = set()
    if path_row["source_text"]:
        for w in _content_lemmas(path_row["source_text"]):
            words.add(w)
    for src_label, tgt_label in step_labels:
        for w in _content_lemmas(src_label):
            words.add(w)
        for w in _content_lemmas(tgt_label):
            words.add(w)
    return list(words)


def _copy_path(src_conn: sqlite3.Connection,
               dst_conn: sqlite3.Connection,
               region: str,
               path_id: int) -> bool:
    """Copy one path (and its neurons/segments/steps/segment_sources)
    from the source region into the (already-open) destination concept
    DB.  Returns True on success.

    Neurons and segments are upserted by label / (source, target,
    relation) so that overlap writes don't create duplicates across
    concepts that share facts.  ID remapping is done locally — each
    concept DB has its own ID namespace.
    """
    prefix = region

    # ── Load path + steps from source ──
    p = src_conn.execute(
        f"SELECT id, origin_id, terminus_id, source_text, created_at "
        f"FROM {prefix}_paths WHERE id = ?",
        (path_id,),
    ).fetchone()
    if p is None:
        return False

    steps = src_conn.execute(
        f"SELECT step_order, segment_id FROM {prefix}_path_steps "
        f"WHERE path_id = ? ORDER BY step_order",
        (path_id,),
    ).fetchall()

    # Collect all segment IDs, then all neuron IDs touched
    seg_ids = [s[1] for s in steps]
    if not seg_ids:
        return False

    segs = {}
    for sid in seg_ids:
        row = src_conn.execute(
            f"SELECT id, source_id, target_id, relation, strength, "
            f"traversals, refutations, created_at, last_used, "
            f"operation_tag FROM {prefix}_segments WHERE id = ?",
            (sid,),
        ).fetchone()
        if row:
            segs[sid] = row

    neuron_ids = set()
    for row in segs.values():
        neuron_ids.add(row[1])  # source_id
        neuron_ids.add(row[2])  # target_id
    neuron_ids.add(p[1])  # origin_id
    neuron_ids.add(p[2])  # terminus_id

    neurons = {}
    for nid in neuron_ids:
        row = src_conn.execute(
            f"SELECT id, label, neuron_type, created_at, metadata "
            f"FROM {prefix}_neurons WHERE id = ?",
            (nid,),
        ).fetchone()
        if row:
            neurons[nid] = row

    # ── Upsert into destination (no prefix) ──
    def upsert_neuron(src_id: int) -> int | None:
        n = neurons.get(src_id)
        if n is None:
            return None
        label, ntype, created_at, metadata = n[1], n[2], n[3], n[4]
        dst_conn.execute(
            "INSERT OR IGNORE INTO neurons "
            "(label, neuron_type, created_at, metadata) "
            "VALUES (?, ?, ?, ?)",
            (label, ntype, created_at, metadata),
        )
        row = dst_conn.execute(
            "SELECT id FROM neurons WHERE label = ?", (label,),
        ).fetchone()
        return row[0] if row else None

    def upsert_segment(src_seg_id: int,
                       new_src_id: int,
                       new_tgt_id: int) -> int | None:
        s = segs.get(src_seg_id)
        if s is None:
            return None
        relation = s[3]
        strength = s[4]
        traversals = s[5]
        refutations = s[6]
        created_at = s[7]
        last_used = s[8]
        op_tag = s[9]
        dst_conn.execute(
            "INSERT OR IGNORE INTO segments "
            "(source_id, target_id, relation, strength, traversals, "
            "refutations, created_at, last_used, operation_tag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_src_id, new_tgt_id, relation, strength, traversals,
             refutations, created_at, last_used, op_tag),
        )
        row = dst_conn.execute(
            "SELECT id FROM segments "
            "WHERE source_id = ? AND target_id = ? AND relation = ?",
            (new_src_id, new_tgt_id, relation),
        ).fetchone()
        return row[0] if row else None

    # Map source neuron IDs → destination IDs
    id_map: dict[int, int] = {}
    for src_nid in neuron_ids:
        dst_nid = upsert_neuron(src_nid)
        if dst_nid is None:
            return False
        id_map[src_nid] = dst_nid

    # Map source segment IDs → destination IDs
    seg_id_map: dict[int, int] = {}
    for src_sid in seg_ids:
        s = segs.get(src_sid)
        if s is None:
            continue
        new_src = id_map.get(s[1])
        new_tgt = id_map.get(s[2])
        if new_src is None or new_tgt is None:
            return False
        dst_sid = upsert_segment(src_sid, new_src, new_tgt)
        if dst_sid is None:
            return False
        seg_id_map[src_sid] = dst_sid

    # Insert path
    new_origin = id_map.get(p[1])
    new_term = id_map.get(p[2])
    if new_origin is None or new_term is None:
        return False
    cur = dst_conn.execute(
        "INSERT INTO paths (origin_id, terminus_id, source_text, "
        "created_at) VALUES (?, ?, ?, ?)",
        (new_origin, new_term, p[3], p[4]),
    )
    new_path_id = cur.lastrowid

    # Insert path steps
    for step_order, src_sid in steps:
        dst_sid = seg_id_map.get(src_sid)
        if dst_sid is None:
            continue
        dst_conn.execute(
            "INSERT OR IGNORE INTO path_steps "
            "(path_id, step_order, segment_id) VALUES (?, ?, ?)",
            (new_path_id, step_order, dst_sid),
        )

    # Copy segment_sources
    for src_sid, dst_sid in seg_id_map.items():
        src_rows = src_conn.execute(
            f"SELECT source_label, created_at "
            f"FROM {prefix}_segment_sources WHERE segment_id = ?",
            (src_sid,),
        ).fetchall()
        for src_label, created_at in src_rows:
            dst_conn.execute(
                "INSERT OR IGNORE INTO segment_sources "
                "(segment_id, source_label, created_at) VALUES (?, ?, ?)",
                (dst_sid, src_label, created_at),
            )

    return True


def _build_concept_vocab(dst_conn: sqlite3.Connection) -> dict[str, int]:
    """Compute lemma → document-frequency over all paths in a concept DB.

    'document' = one path. df = number of distinct paths a lemma
    appears in (path.source_text + all step neuron labels).
    """
    rows = dst_conn.execute(
        "SELECT p.id, p.source_text, "
        "   GROUP_CONCAT(ns.label, ' ') AS src_labels, "
        "   GROUP_CONCAT(nt.label, ' ') AS tgt_labels "
        "FROM paths p "
        "LEFT JOIN path_steps ps ON ps.path_id = p.id "
        "LEFT JOIN segments s ON s.id = ps.segment_id "
        "LEFT JOIN neurons ns ON ns.id = s.source_id "
        "LEFT JOIN neurons nt ON nt.id = s.target_id "
        "GROUP BY p.id"
    ).fetchall()
    df: Counter = Counter()
    for _pid, src_text, src_labels, tgt_labels in rows:
        doc_lemmas: set[str] = set()
        for part in (src_text or "", src_labels or "", tgt_labels or ""):
            for w in _content_lemmas(part):
                doc_lemmas.add(w)
        for lemma in doc_lemmas:
            df[lemma] += 1
    return dict(df)


def migrate(source_db: Path, subject: str, dest: Path,
            verify: bool = False) -> None:
    src = sqlite3.connect(str(source_db))
    src.row_factory = sqlite3.Row

    regions = [r[0] for r in src.execute(
        "SELECT name FROM regions ORDER BY name"
    ).fetchall()]
    print(f"Source: {source_db} — {len(regions)} regions")

    backend = HierarchicalBackend(str(dest))

    # Register subject (idempotent — extractor may have done this)
    backend.register_subject(subject, description="Biology 2e full book")
    # Ensure _unclassified exists
    backend.register_concept(
        subject, "_unclassified",
        source_kind="system_bucket",
        description="Facts that matched no topic trigger.",
        trigger_lemmas=[],
    )

    total_paths = 0
    routed_paths = 0      # paths that hit ≥1 concept
    unclassified_paths = 0
    overlap_hist: Counter = Counter()  # concept-count → path count
    concept_written: Counter = Counter()  # concept → paths written

    for region in regions:
        path_ids = [
            r[0] for r in src.execute(
                f"SELECT id FROM {region}_paths ORDER BY id"
            ).fetchall()
        ]
        if not path_ids:
            continue

        print(f"  {region}: {len(path_ids)} paths …")
        commit_every = 500

        for i, path_id in enumerate(path_ids):
            total_paths += 1
            # Gather labels from all steps for better lemma signal
            step_rows = src.execute(
                f"SELECT ns.label, nt.label "
                f"FROM {region}_path_steps ps "
                f"JOIN {region}_segments s ON s.id = ps.segment_id "
                f"JOIN {region}_neurons ns ON ns.id = s.source_id "
                f"JOIN {region}_neurons nt ON nt.id = s.target_id "
                f"WHERE ps.path_id = ?",
                (path_id,),
            ).fetchall()
            p_row = src.execute(
                f"SELECT source_text FROM {region}_paths WHERE id = ?",
                (path_id,),
            ).fetchone()
            source_text = p_row["source_text"] if p_row else None
            lemmas = _path_lemmas(
                {"source_text": source_text},
                [(r[0], r[1]) for r in step_rows],
            )
            concepts = backend.route_teach(subject, lemmas)
            if not concepts:
                concepts = ["_unclassified"]
                unclassified_paths += 1
            else:
                routed_paths += 1

            overlap_hist[len(concepts)] += 1
            for concept in concepts:
                dst_conn = backend.concept_conn(subject, concept)
                ok = _copy_path(src, dst_conn, region, path_id)
                if ok:
                    concept_written[concept] += 1
                if (i + 1) % commit_every == 0:
                    dst_conn.commit()

        # Commit after each region
        backend.commit()
        print(f"    done — cumulative paths: {total_paths}")

    # Final commit
    backend.commit()

    # ── Rebuild concept_vocab for every concept that got paths ──
    print(f"\nRebuilding concept_vocab for "
          f"{len(concept_written)} concepts …")
    for concept in concept_written:
        dst_conn = backend.concept_conn(subject, concept)
        vocab = _build_concept_vocab(dst_conn)
        backend.update_concept_vocab(subject, concept, vocab)
    backend.commit()

    # ── Carry bridges ──
    try:
        bridge_rows = src.execute(
            "SELECT source_region, source_neuron_id, "
            "       target_region, target_neuron_id, relation "
            "FROM bridges"
        ).fetchall()
        print(f"Carrying {len(bridge_rows)} bridges to subject DB …")
        sc = backend.subject_conn(subject)
        src.row_factory = sqlite3.Row
        inserted = 0
        for br in bridge_rows:
            src_region, src_nid, tgt_region, tgt_nid, relation = br
            src_label_row = src.execute(
                f"SELECT label FROM {src_region}_neurons WHERE id = ?",
                (src_nid,),
            ).fetchone()
            tgt_label_row = src.execute(
                f"SELECT label FROM {tgt_region}_neurons WHERE id = ?",
                (tgt_nid,),
            ).fetchone()
            if src_label_row is None or tgt_label_row is None:
                continue
            sc.execute(
                "INSERT OR IGNORE INTO bridges "
                "(source_concept, source_label, "
                " target_concept, target_label, relation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (src_region, str(src_label_row[0]),
                 tgt_region, str(tgt_label_row[0]),
                 relation, time.time()),
            )
            inserted += 1
            if inserted % 50_000 == 0:
                sc.commit()
                print(f"  … {inserted} bridges written")
        sc.commit()
        print(f"  {inserted} bridges written.")
    except sqlite3.OperationalError:
        print("  (no bridges table in source DB — skipping)")

    src.close()
    backend.close()

    # ── Report ──
    print()
    print("=" * 58)
    print(f"Migration complete")
    print(f"  Source paths:          {total_paths}")
    print(f"  Routed to concepts:    {routed_paths} "
          f"({100*routed_paths//total_paths if total_paths else 0}%)")
    print(f"  Unclassified:          {unclassified_paths} "
          f"({100*unclassified_paths//total_paths if total_paths else 0}%)")
    print(f"  Concepts with paths:   {len(concept_written)}")
    print()
    print("Overlap distribution (concept DBs per path):")
    for n_concepts in sorted(overlap_hist):
        print(f"  {n_concepts} concepts: {overlap_hist[n_concepts]} paths")
    print("=" * 58)

    if verify:
        print("\n--- VERIFY ---")
        print("Top 20 concepts by path count:")
        for concept, cnt in concept_written.most_common(20):
            print(f"  {concept}: {cnt}")
        unc = concept_written.get("_unclassified", 0)
        pct_unc = 100 * unc // total_paths if total_paths else 0
        pass_fail = "PASS" if pct_unc < 20 else "FAIL"
        print(f"\nUnclassified rate: {pct_unc}% — {pass_fail} "
              f"(target < 20%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--subject", default="biology")
    ap.add_argument("--dest", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    if not source.exists():
        print(f"error: source DB not found: {source}", file=sys.stderr)
        return 2

    migrate(source, args.subject, dest, verify=args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
