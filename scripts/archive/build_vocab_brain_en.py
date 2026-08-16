"""Build the default English vocab brain for v040 predicate slots.

Walks `synthesizer._TEMPLATES` and `synthesizer._ATTR_TEMPLATES`,
extracts the predicate phrase from each template (the part between
or surrounding `{src}` / `{tgt}`), and writes `(relation_name,
english_form, phrase)` triples to a fresh Sara brain.db.

Per v040: this brain becomes L3 of the synthesizer stack — the
substrate from which the inference adapter looks up the English
phrase for any given relation. Uses the same SQLite schema as
content brains so existing inspection tooling works.

Per v041 position 1: source is v032 templates ONLY. Real-brain
relation names (e.g. `forms`, `role_in`) that aren't in the v032
templates fall back at inference time to `relation.replace("_", " ")`.

We bypass `Brain.teach_triple` and write directly to the schema —
the chain-learning machinery (the `_attribute` convention) is the
wrong abstraction for a vocab lookup. The vocab brain stores raw
`relation -> english_form -> phrase` triples without any chain
inference.

Usage:
    .venv/bin/python scripts/build_vocab_brain_en.py
    # writes src/sara_brain/cortex/vocab/vocab_en.db
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

from sara_brain.cortex.transformer.synthesizer import (
    _ATTR_TEMPLATES,
    _TEMPLATES,
)


_PLACEHOLDER_RE = re.compile(r"\{(?:src|tgt)\}")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS neurons (
    id          INTEGER PRIMARY KEY,
    label       TEXT NOT NULL UNIQUE,
    neuron_type TEXT NOT NULL,
    created_at  REAL,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY,
    source_id     INTEGER NOT NULL REFERENCES neurons(id),
    target_id     INTEGER NOT NULL REFERENCES neurons(id),
    relation      TEXT NOT NULL,
    strength      REAL NOT NULL DEFAULT 1.0,
    traversals    INTEGER NOT NULL DEFAULT 0,
    refutations   INTEGER NOT NULL DEFAULT 0,
    created_at    REAL,
    last_used     REAL,
    operation_tag TEXT,
    UNIQUE(source_id, target_id, relation)
);
"""


def _extract_predicate_phrase(template: str) -> str:
    """Strip `{src}` / `{tgt}` placeholders, return the predicate
    phrase. `"{tgt} is a {src}"` -> `"is a"`. Collapses whitespace."""
    phrase = _PLACEHOLDER_RE.sub("", template).strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def _ensure_neuron(conn: sqlite3.Connection, label: str) -> int:
    """Find or create a neuron with `label`, return its id."""
    row = conn.execute("SELECT id FROM neurons WHERE label = ?", (label,)).fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute(
        "INSERT INTO neurons (label, neuron_type, created_at) VALUES (?, ?, ?)",
        (label, "vocab", time.time()),
    )
    return cur.lastrowid


def _add_segment(conn: sqlite3.Connection, src: str, rel: str, tgt: str) -> None:
    """Add a (src, rel, tgt) segment using direct neuron labels.
    Idempotent: the UNIQUE(source, target, relation) constraint
    silently no-ops on retry."""
    src_id = _ensure_neuron(conn, src)
    tgt_id = _ensure_neuron(conn, tgt)
    conn.execute(
        "INSERT OR IGNORE INTO segments "
        "(source_id, target_id, relation, created_at) "
        "VALUES (?, ?, ?, ?)",
        (src_id, tgt_id, rel, time.time()),
    )


def build_vocab_brain_en(out_path: Path) -> dict:
    """Construct the vocab_en brain. Returns a small summary dict."""
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} exists; remove or rename before regenerating"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(out_path))
    conn.executescript(_SCHEMA)

    seen: set[tuple[str, str]] = set()

    # _TEMPLATES: src-first; predicate is whatever sits between the
    # placeholders. _ATTR_TEMPLATES: typically tgt-first; same
    # extraction. Arg-order is recorded as a separate segment for
    # inspection / future tooling, but the model learns the order
    # from training-prose order via the existing `<attr>` flag.
    for table_name, table in (("normal", _TEMPLATES), ("attr", _ATTR_TEMPLATES)):
        for relation, template in table.items():
            phrase = _extract_predicate_phrase(template)
            if not phrase:
                continue
            key = (relation, phrase)
            if key in seen:
                continue
            seen.add(key)
            _add_segment(conn, relation, "english_form", phrase)

            order = "tgt_first" if (
                table_name == "attr" and template.lstrip().startswith("{tgt}")
            ) else "src_first"
            _add_segment(conn, relation, "arg_order", order)

    conn.commit()

    n_neurons = conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    n_segments = conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    n_relations = len({r for r, _ in seen})
    conn.close()

    summary = {
        "out_path": str(out_path.resolve()),
        "relations_taught": n_relations,
        "english_forms_taught": len(seen),
        "neurons_in_brain": n_neurons,
        "segments_in_brain": n_segments,
    }
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--out", type=Path,
        default=Path("src/sara_brain/cortex/vocab/vocab_en.db"),
        help="Output vocab brain path (default: "
             "src/sara_brain/cortex/vocab/vocab_en.db)",
    )
    args = p.parse_args()

    info = build_vocab_brain_en(args.out)
    print("vocab brain built.")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
