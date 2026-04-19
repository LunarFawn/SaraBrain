"""Hierarchical per-concept SQL backend.

Three tiers of SQLite files:

    brain_root/
        brain.db                             # subjects index
        subjects/
            biology.db                       # concepts index + bridges + vocab
        concepts/
            biology/
                mitosis.db                   # self-contained neurons/segments/paths
                dna_replication.db
                ...
                _unclassified.db             # debug bucket for unrouted facts

The subject DB's `concept_vocab` maps lemma → list of concept DBs, so
routing a query is a single SQL lookup instead of a scan of every
concept file. `concept_lemmas` is the *trigger set* used at teach time
to decide which concept DBs a new fact belongs in; overlap is allowed
and expected (a fact about anaphase lives in mitosis.db, meiosis.db,
anaphase.db, …).

The backend does NOT provide the full Brain API — it's the storage
layer that the migration tool, the scorer, and (later) Brain itself
sit on top of. Writes happen via raw concept-DB connections returned
by `concept_conn()`; the standard `NeuronRepo(conn, prefix="")`,
`SegmentRepo(conn, prefix="")`, `PathRepo(conn, prefix="")` work
unchanged against those connections.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from .database import Database


# Concept DBs outnumber OS file-descriptor-friendly concurrent opens
# once you pass a few thousand. WAL mode opens 3 FDs per DB (main + wal
# + shm), so a cap of 256 keeps us well under every default limit.
_MAX_OPEN_CONCEPTS = 256


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify_concept(name: str) -> str:
    """Concept names go on the filesystem. Normalise to a safe slug.

    'DNA replication' → 'dna_replication'
    'gram-negative bacterium' → 'gram_negative_bacterium'
    """
    s = _SLUG_RE.sub("_", name.strip().lower()).strip("_")
    return s or "unnamed"


_BRAIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS subjects (
    name        TEXT PRIMARY KEY,
    db_path     TEXT NOT NULL,
    description TEXT,
    created_at  REAL
);
"""


_SUBJECT_SCHEMA = """
CREATE TABLE IF NOT EXISTS concepts (
    name         TEXT PRIMARY KEY,
    db_path      TEXT NOT NULL,
    source_kind  TEXT NOT NULL,
    description  TEXT,
    created_at   REAL
);

CREATE TABLE IF NOT EXISTS concept_vocab (
    concept   TEXT NOT NULL,
    lemma     TEXT NOT NULL,
    df        INTEGER NOT NULL,
    PRIMARY KEY (concept, lemma)
);
CREATE INDEX IF NOT EXISTS idx_concept_vocab_lemma
    ON concept_vocab(lemma);

CREATE TABLE IF NOT EXISTS concept_lemmas (
    concept   TEXT NOT NULL,
    lemma     TEXT NOT NULL,
    PRIMARY KEY (concept, lemma)
);
CREATE INDEX IF NOT EXISTS idx_concept_lemmas_lemma
    ON concept_lemmas(lemma);

CREATE TABLE IF NOT EXISTS bridges (
    id              INTEGER PRIMARY KEY,
    source_concept  TEXT NOT NULL,
    source_label    TEXT NOT NULL,
    target_concept  TEXT NOT NULL,
    target_label    TEXT NOT NULL,
    relation        TEXT NOT NULL DEFAULT 'same_as',
    created_at      REAL,
    UNIQUE(source_concept, source_label, target_concept, target_label)
);
CREATE INDEX IF NOT EXISTS idx_bridges_source
    ON bridges(source_concept, source_label);
CREATE INDEX IF NOT EXISTS idx_bridges_target
    ON bridges(target_concept, target_label);
"""


class HierarchicalBackend:
    """Three-tier SQLite backend rooted at `brain_root/`.

    Usage:
        backend = HierarchicalBackend("brain_root/")
        backend.register_subject("biology", description="...")
        backend.register_concept(
            subject="biology",
            concept="mitosis",
            source_kind="glossary_term",
            description="the division of somatic cells",
            trigger_lemmas=["mitosis", "mitotic", "prophase", ...],
        )
        conn = backend.concept_conn("biology", "mitosis")
        # Standard repos work against conn with prefix="":
        nr = NeuronRepo(conn)
        ...

    Fan-out teach is the caller's responsibility: compute fact lemmas,
    call `route_teach` to get the list of concepts, open each and
    write. This keeps the backend free of parser/learner coupling.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "subjects").mkdir(exist_ok=True)
        (self.root / "concepts").mkdir(exist_ok=True)

        self._brain_conn = sqlite3.connect(str(self.root / "brain.db"))
        self._brain_conn.execute("PRAGMA journal_mode=WAL")
        self._brain_conn.execute("PRAGMA foreign_keys=ON")
        self._brain_conn.executescript(_BRAIN_SCHEMA)

        self._subject_conns: dict[str, sqlite3.Connection] = {}
        # LRU cache of opened concept Databases keyed by (subject, slug).
        # Insertion order = access order; oldest is evicted when the
        # cache exceeds `_MAX_OPEN_CONCEPTS`.
        self._concept_dbs: OrderedDict[tuple[str, str], Database] = (
            OrderedDict()
        )

    # ── Tier 1: subjects ────────────────────────────────────────────

    def register_subject(self, name: str, description: str = "") -> None:
        """Idempotently register a subject in brain.db.

        Creates `subjects/{name}.db` on disk with the subject-level
        schema if it doesn't already exist.
        """
        name = name.strip().lower()
        db_path = f"subjects/{name}.db"
        self._brain_conn.execute(
            "INSERT OR IGNORE INTO subjects (name, db_path, description, "
            "created_at) VALUES (?, ?, ?, ?)",
            (name, db_path, description, time.time()),
        )
        self._brain_conn.commit()
        # Eagerly open + init the subject DB
        self._subject_conn(name)

    def list_subjects(self) -> list[dict]:
        rows = self._brain_conn.execute(
            "SELECT name, db_path, description, created_at "
            "FROM subjects ORDER BY name"
        ).fetchall()
        return [
            {"name": r[0], "db_path": r[1],
             "description": r[2], "created_at": r[3]}
            for r in rows
        ]

    # ── Tier 2: subject index ───────────────────────────────────────

    def _subject_conn(self, subject: str) -> sqlite3.Connection:
        subject = subject.strip().lower()
        if subject in self._subject_conns:
            return self._subject_conns[subject]
        path = self.root / "subjects" / f"{subject}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SUBJECT_SCHEMA)
        self._subject_conns[subject] = conn
        return conn

    def subject_conn(self, subject: str) -> sqlite3.Connection:
        """Public accessor for the subject-level DB connection."""
        return self._subject_conn(subject)

    def register_concept(self, subject: str, concept: str,
                         source_kind: str, description: str = "",
                         trigger_lemmas: Iterable[str] = ()) -> str:
        """Idempotently register a concept under a subject.

        - Creates `concepts/{subject}/{slug}.db` with the standard
          Sara schema (neurons/segments/paths/path_steps/segment_sources).
        - Inserts into `concepts` with the slug as name.
        - Seeds `concept_lemmas` with the trigger set.
        Returns the slug (canonical concept name on disk).
        """
        subject = subject.strip().lower()
        slug = slugify_concept(concept)
        if slug.startswith("_") and slug != "_unclassified":
            # Reserve leading-underscore names for system buckets
            slug = "c" + slug
        self.register_subject(subject)

        sc = self._subject_conn(subject)
        db_rel = f"concepts/{subject}/{slug}.db"
        sc.execute(
            "INSERT OR IGNORE INTO concepts "
            "(name, db_path, source_kind, description, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (slug, db_rel, source_kind, description, time.time()),
        )
        # Seed trigger lemmas — additive, never clobbers
        for lemma in trigger_lemmas:
            lemma = lemma.strip().lower()
            if not lemma:
                continue
            sc.execute(
                "INSERT OR IGNORE INTO concept_lemmas (concept, lemma) "
                "VALUES (?, ?)",
                (slug, lemma),
            )
        sc.commit()
        # Ensure the concept DB file exists with schema applied
        self._concept_db(subject, slug)
        return slug

    def list_concepts(self, subject: str) -> list[dict]:
        sc = self._subject_conn(subject)
        rows = sc.execute(
            "SELECT name, db_path, source_kind, description, created_at "
            "FROM concepts ORDER BY name"
        ).fetchall()
        return [
            {"name": r[0], "db_path": r[1], "source_kind": r[2],
             "description": r[3], "created_at": r[4]}
            for r in rows
        ]

    def concept_exists(self, subject: str, concept: str) -> bool:
        slug = slugify_concept(concept)
        sc = self._subject_conn(subject)
        row = sc.execute(
            "SELECT 1 FROM concepts WHERE name = ?", (slug,),
        ).fetchone()
        return row is not None

    # ── Tier 3: concept DB ──────────────────────────────────────────

    def _concept_db(self, subject: str, concept: str) -> Database:
        """Open (or reuse) the Database for a concept. Creates the file
        + standard schema on first touch. LRU-cached — oldest open
        concept DB is closed once the cache exceeds
        `_MAX_OPEN_CONCEPTS` so we don't blow the OS file-descriptor
        limit when ~7000 concept DBs exist."""
        subject = subject.strip().lower()
        slug = slugify_concept(concept)
        key = (subject, slug)
        cached = self._concept_dbs.get(key)
        if cached is not None:
            self._concept_dbs.move_to_end(key)
            return cached
        concept_dir = self.root / "concepts" / subject
        concept_dir.mkdir(parents=True, exist_ok=True)
        path = concept_dir / f"{slug}.db"
        db = Database(str(path))
        self._concept_dbs[key] = db
        while len(self._concept_dbs) > _MAX_OPEN_CONCEPTS:
            _, evict_db = self._concept_dbs.popitem(last=False)
            try:
                evict_db.conn.commit()
            except Exception:
                pass
            evict_db.close()
        return db

    def close_concept(self, subject: str, concept: str) -> None:
        """Explicitly close a concept DB. Useful when the caller knows
        it is done with one and wants to free the FD immediately."""
        subject = subject.strip().lower()
        slug = slugify_concept(concept)
        key = (subject, slug)
        db = self._concept_dbs.pop(key, None)
        if db is not None:
            try:
                db.conn.commit()
            except Exception:
                pass
            db.close()

    def concept_conn(self, subject: str, concept: str) -> sqlite3.Connection:
        """Raw connection to a concept DB. Standard repos work against
        it with prefix=''."""
        return self._concept_db(subject, concept).conn

    # ── Routing ─────────────────────────────────────────────────────

    def route_teach(self, subject: str,
                    fact_lemmas: Iterable[str]) -> list[str]:
        """Return the concept slugs whose trigger lemmas intersect the
        fact's content lemmas.

        Empty list means the fact has no known home — caller should
        route it to `_unclassified` so nothing is silently lost.
        """
        sc = self._subject_conn(subject)
        wanted = {w.strip().lower() for w in fact_lemmas if w and w.strip()}
        if not wanted:
            return []
        q = ("SELECT DISTINCT concept FROM concept_lemmas "
             "WHERE lemma IN (%s)" % ",".join("?" * len(wanted)))
        rows = sc.execute(q, tuple(wanted)).fetchall()
        return [r[0] for r in rows]

    def route_query(self, subject: str,
                    question_lemmas: Iterable[str],
                    top_k: int = 5) -> list[str]:
        """Top-k concepts whose observed vocabulary (concept_vocab)
        covers the most of the question's lemmas.

        Uses the df-weighted vocab, not the trigger set, because at
        query time we care about where the content actually lives.
        Ties broken by sum of df (a concept that mentions the lemma
        in many paths is a stronger match than one that mentions it
        in a single path).
        """
        sc = self._subject_conn(subject)
        wanted = {w.strip().lower() for w in question_lemmas if w and w.strip()}
        if not wanted:
            return []
        q = (
            "SELECT concept, COUNT(*) AS hits, SUM(df) AS total_df "
            "FROM concept_vocab WHERE lemma IN (%s) "
            "GROUP BY concept ORDER BY hits DESC, total_df DESC LIMIT ?"
        ) % ",".join("?" * len(wanted))
        rows = sc.execute(q, (*tuple(wanted), top_k)).fetchall()
        return [r[0] for r in rows]

    # ── Vocab maintenance ──────────────────────────────────────────

    def update_concept_vocab(self, subject: str, concept: str,
                             lemma_df: dict[str, int]) -> None:
        """Replace a concept's observed vocabulary (concept_vocab) with
        the given lemma → df mapping. Called by the migration tool
        after all paths are written into the concept DB."""
        slug = slugify_concept(concept)
        sc = self._subject_conn(subject)
        sc.execute("DELETE FROM concept_vocab WHERE concept = ?", (slug,))
        sc.executemany(
            "INSERT INTO concept_vocab (concept, lemma, df) VALUES (?, ?, ?)",
            [(slug, lemma, df) for lemma, df in lemma_df.items() if lemma],
        )
        sc.commit()

    # ── Bridges ────────────────────────────────────────────────────

    def add_bridge(self, subject: str,
                   source_concept: str, source_label: str,
                   target_concept: str, target_label: str,
                   relation: str = "same_as") -> None:
        sc = self._subject_conn(subject)
        sc.execute(
            "INSERT OR IGNORE INTO bridges "
            "(source_concept, source_label, target_concept, target_label, "
            "relation, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (slugify_concept(source_concept), source_label.lower(),
             slugify_concept(target_concept), target_label.lower(),
             relation, time.time()),
        )

    # ── Lifecycle ──────────────────────────────────────────────────

    def commit(self) -> None:
        self._brain_conn.commit()
        for c in self._subject_conns.values():
            c.commit()
        for db in self._concept_dbs.values():
            db.conn.commit()

    def close(self) -> None:
        self.commit()
        for db in self._concept_dbs.values():
            db.close()
        self._concept_dbs.clear()
        for c in self._subject_conns.values():
            c.close()
        self._subject_conns.clear()
        self._brain_conn.close()

    def __enter__(self) -> "HierarchicalBackend":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
