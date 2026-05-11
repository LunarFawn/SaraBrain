"""Bootstrap helpers — make a fresh brain useful by default.

A blank brain.db is a path graph with no synonym scaffolding. Question
words like "tallest" can't bridge to substrate labels like "directional
selection" without the synonym edges that Moby Thesaurus II provides.
Historically this was a separate `build_dictionary.py` step; that made
it easy to forget and made the standard recipe long.

This module exposes `ensure_dictionary(brain)` — idempotent. Loads the
Moby Thesaurus II synonym groups into the brain's `synonym_of`
relations. Skips if dictionary edges already exist.

Call from any CLI / pipeline that wants a usable-out-of-the-box brain.
The dictionary load is one-shot (~13s for 30k entries / 800k edges);
subsequent calls are no-ops.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .models.neuron import NeuronType

if TYPE_CHECKING:
    from .core.brain import Brain


# Default Moby Thesaurus location, relative to repo root. Override with
# the SARA_MOBY_THESAURUS env var.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_THESAURUS_PATH = _REPO_ROOT / "data" / "moby_thesaurus.txt"


def _thesaurus_path() -> Path:
    return Path(os.environ.get(
        "SARA_MOBY_THESAURUS", str(_DEFAULT_THESAURUS_PATH),
    ))


def _dictionary_already_loaded(brain: "Brain") -> bool:
    """True if the brain already has synonym_of segments (= dictionary
    has been loaded before). Cheap COUNT query."""
    cur = brain.conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM segments WHERE relation = ? LIMIT 1",
        ("synonym_of",),
    )
    row = cur.fetchone()
    return bool(row and row[0] > 0)


def ensure_dictionary(
    brain: "Brain",
    *,
    max_synonyms: int = 15,
    limit: int = 0,
    verbose: bool = True,
) -> dict:
    """Load Moby Thesaurus II into the brain if not already present.

    Idempotent: returns early without touching the brain when synonym_of
    segments already exist. Use `force=True` is not provided — if you
    need to re-load, delete the synonym_of segments first.

    Args:
      brain: the Brain instance to populate.
      max_synonyms: cap per root word (Moby entries can have 100+).
      limit: 0 = all entries, else stop after this many.
      verbose: print progress to stderr.

    Returns a stats dict: {"status": "loaded"|"already_present"|"missing",
                            "neurons": N, "segments": N, "entries": N,
                            "elapsed_s": float}.
    """
    if _dictionary_already_loaded(brain):
        return {
            "status": "already_present",
            "neurons": 0, "segments": 0, "entries": 0, "elapsed_s": 0.0,
        }

    path = _thesaurus_path()
    if not path.exists():
        if verbose:
            print(
                f"[bootstrap] WARN moby_thesaurus.txt not found at {path} — "
                f"skipping dictionary bootstrap. Synonym bridges will be "
                f"unavailable for wavefront convergence.",
                file=sys.stderr,
            )
        return {
            "status": "missing",
            "neurons": 0, "segments": 0, "entries": 0, "elapsed_s": 0.0,
        }

    if verbose:
        print(
            f"[bootstrap] loading Moby Thesaurus II into brain "
            f"({path.name}, ~30k entries) — one-time cost ~10-15s...",
            file=sys.stderr,
        )

    start = time.time()
    entries = 0
    total_neurons = 0
    total_segments = 0
    neuron_repo = brain.neuron_repo
    segment_repo = brain.segment_repo

    with open(path) as f:
        for line in f:
            if limit and entries >= limit:
                break
            parts = [w.strip().lower() for w in line.strip().split(",")]
            if len(parts) < 2:
                continue
            root = parts[0]
            synonyms = parts[1:max_synonyms + 1]
            root_n, created = neuron_repo.get_or_create(
                root, NeuronType.CONCEPT,
            )
            if created:
                total_neurons += 1
            for syn in synonyms:
                if not syn or syn == root:
                    continue
                syn_n, created = neuron_repo.get_or_create(
                    syn, NeuronType.CONCEPT,
                )
                if created:
                    total_neurons += 1
                _, created = segment_repo.get_or_create(
                    root_n.id, syn_n.id, "synonym_of",
                )
                if created:
                    total_segments += 1
                _, created = segment_repo.get_or_create(
                    syn_n.id, root_n.id, "synonym_of",
                )
                if created:
                    total_segments += 1
            entries += 1
            if entries % 5000 == 0:
                brain.conn.commit()
                if verbose:
                    print(
                        f"[bootstrap]   {entries} entries  "
                        f"{total_neurons} neurons  {total_segments} edges",
                        file=sys.stderr,
                    )

    brain.conn.commit()
    elapsed = time.time() - start
    if verbose:
        print(
            f"[bootstrap] dictionary loaded in {elapsed:.1f}s: "
            f"{entries} entries, {total_neurons} new neurons, "
            f"{total_segments} synonym edges.",
            file=sys.stderr,
        )
    return {
        "status": "loaded",
        "neurons": total_neurons,
        "segments": total_segments,
        "entries": entries,
        "elapsed_s": elapsed,
    }


__all__ = ["ensure_dictionary"]
