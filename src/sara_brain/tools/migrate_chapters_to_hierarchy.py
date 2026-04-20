#!/usr/bin/env python3
"""Chapter-grain migration: one concept DB per source-chapter region.

Unlike `migrate_to_hierarchy` (topic-grain with fan-out), this tool
takes each region from the monolithic DB and copies its 6 tables
verbatim into a separate concept DB, stripping the region prefix.
No routing, no overlap — one chapter, one file.

Usage:
    .venv/bin/python -m sara_brain.tools.migrate_chapters_to_hierarchy \\
        --source biology2e.db \\
        --subject biology \\
        --dest brain_chapters/
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from collections import Counter
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
        w for w in _WORD_RE.findall((text or "").lower())
        if w not in _STOPWORDS and len(w) >= 3
    ]


_CHAPTER_TABLES = (
    "neurons",
    "segments",
    "paths",
    "path_steps",
    "segment_sources",
)


def _copy_region(src: sqlite3.Connection, dst: sqlite3.Connection,
                 region: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    dst.executescript("""
        CREATE TABLE IF NOT EXISTS neurons (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL UNIQUE,
            kind TEXT,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            relation TEXT,
            weight REAL DEFAULT 1.0,
            created_at REAL,
            UNIQUE(source_id, target_id, relation)
        );
        CREATE TABLE IF NOT EXISTS paths (
            id INTEGER PRIMARY KEY,
            source_text TEXT,
            terminus_id INTEGER,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS path_steps (
            path_id INTEGER NOT NULL,
            step_order INTEGER NOT NULL,
            segment_id INTEGER NOT NULL,
            PRIMARY KEY (path_id, step_order)
        );
        CREATE TABLE IF NOT EXISTS segment_sources (
            segment_id INTEGER NOT NULL,
            source_text TEXT,
            PRIMARY KEY (segment_id, source_text)
        );
    """)
    for tbl in _CHAPTER_TABLES:
        src_tbl = f"{region}_{tbl}"
        exists = src.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (src_tbl,),
        ).fetchone()
        if not exists:
            counts[tbl] = 0
            continue
        cols = [r[1] for r in src.execute(
            f"PRAGMA table_info({src_tbl})"
        ).fetchall()]
        col_list = ",".join(cols)
        placeholders = ",".join("?" * len(cols))
        dst.execute(f"DELETE FROM {tbl}")
        rows = src.execute(f"SELECT {col_list} FROM {src_tbl}").fetchall()
        if rows:
            dst.executemany(
                f"INSERT INTO {tbl} ({col_list}) VALUES ({placeholders})",
                rows,
            )
        counts[tbl] = len(rows)
    dst.commit()
    return counts


def _build_concept_vocab(dst: sqlite3.Connection) -> dict[str, int]:
    rows = dst.execute(
        "SELECT p.id, p.source_text, "
        "   GROUP_CONCAT(ns.label, ' '), GROUP_CONCAT(nt.label, ' ') "
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


def migrate(source_db: Path, subject: str, dest: Path) -> None:
    src = sqlite3.connect(str(source_db))
    src.row_factory = sqlite3.Row

    regions = [r[0] for r in src.execute(
        "SELECT name FROM regions ORDER BY name"
    ).fetchall()]
    print(f"Source: {source_db} — {len(regions)} regions (chapters)")

    backend = HierarchicalBackend(str(dest))
    backend.register_subject(subject, description=f"{subject} — chapter-grain")

    # Carry bridges verbatim to subject DB (chapter = concept)
    sc = backend.subject_conn(subject)

    t0 = time.time()
    for i, region in enumerate(regions, 1):
        slug = backend.register_concept(
            subject, region,
            source_kind="chapter",
            description=f"Chapter {region}",
            trigger_lemmas=[region],
        )
        conn = backend.concept_conn(subject, slug)
        counts = _copy_region(src, conn, region)
        vocab = _build_concept_vocab(conn)
        backend.update_concept_vocab(subject, slug, vocab)
        print(f"  [{i}/{len(regions)}] {region} → {slug}: "
              f"{counts.get('paths',0)} paths, {len(vocab)} vocab")

    backend.commit()

    # ── Bridges: source_region/target_region become source/target concepts ──
    try:
        bridge_rows = src.execute(
            "SELECT source_region, source_neuron_id, "
            "       target_region, target_neuron_id, relation FROM bridges"
        ).fetchall()
        print(f"Carrying {len(bridge_rows)} bridges …")
        inserted = 0
        for br in bridge_rows:
            src_region, src_nid, tgt_region, tgt_nid, relation = br
            sl = src.execute(
                f"SELECT label FROM {src_region}_neurons WHERE id=?",
                (src_nid,),
            ).fetchone()
            tl = src.execute(
                f"SELECT label FROM {tgt_region}_neurons WHERE id=?",
                (tgt_nid,),
            ).fetchone()
            if not sl or not tl:
                continue
            sc.execute(
                "INSERT OR IGNORE INTO bridges "
                "(source_concept, source_label, target_concept, "
                " target_label, relation, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (src_region, sl[0], tgt_region, tl[0],
                 relation or "same_as", time.time()),
            )
            inserted += 1
        sc.commit()
        print(f"  bridges inserted: {inserted}")
    except sqlite3.OperationalError as e:
        print(f"  (no bridges table or skipped: {e})")

    print(f"\nDone in {time.time()-t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--subject", required=True)
    p.add_argument("--dest", required=True, type=Path)
    args = p.parse_args()
    migrate(args.source, args.subject, args.dest)


if __name__ == "__main__":
    main()
