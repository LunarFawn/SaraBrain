#!/usr/bin/env python3
"""Create cross-region bridges in biology2e.db.

A bridge is an explicit link between a neuron in one region and a
neuron in another region with the same (or closely-related) label.
Without bridges, a question routed to one chapter cannot see concepts
taught in a different chapter, even when biology treats them as the
same thing. "DNA" in ch3 (Biological Macromolecules) should be
bridged to "DNA" in ch14 (DNA Structure and Function) and ch16 (Gene
Expression).

Algorithm:
  1. Walk every region's neuron table.
  2. Index neurons by their lowercased base label.
  3. For each label that appears in MORE THAN ONE region, emit a
     bridge for each pair — "same_as" relation.

The bridges table already exists in the schema (see
storage/database.py). This script only INSERTS, never deletes.

Usage:
    .venv/bin/python benchmarks/bridge_biology2e.py --db biology2e.db
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

from sara_brain.core.brain import Brain


def build_bridges(db_path: str, min_label_len: int = 4) -> dict:
    """Return stats. Skip labels shorter than min_label_len to avoid
    bridging every filler ("cell", "water", "is") across the whole
    book — those create excess cross-region traffic without helping
    disambiguation. Only content concepts (length >= 4) bridge.
    """
    brain = Brain(db_path)
    regions = [r[0] for r in brain.conn.execute(
        "SELECT name FROM regions ORDER BY name"
    ).fetchall()]

    # Collect: label → {region: neuron_id}
    label_index: dict[str, dict[str, int]] = defaultdict(dict)
    for region in regions:
        rows = brain.conn.execute(
            f"SELECT id, label FROM {region}_neurons"
        ).fetchall()
        for nid, label in rows:
            key = (label or "").strip().lower()
            if len(key) < min_label_len:
                continue
            # Skip attribute-node labels — they're structural helpers.
            if key.endswith("_attribute"):
                continue
            label_index[key][region] = nid

    # Build bridges for labels present in >= 2 regions.
    bridge_sql = (
        "INSERT OR IGNORE INTO bridges "
        "(source_region, source_neuron_id, target_region, "
        "target_neuron_id, relation, created_at) "
        "VALUES (?, ?, ?, ?, 'same_as', ?)"
    )
    created = 0
    now = time.time()
    multi_region_labels = 0
    for key, region_map in label_index.items():
        if len(region_map) < 2:
            continue
        multi_region_labels += 1
        items = list(region_map.items())
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                src_region, src_nid = items[i]
                tgt_region, tgt_nid = items[j]
                try:
                    cur = brain.conn.execute(bridge_sql, (
                        src_region, src_nid, tgt_region, tgt_nid, now,
                    ))
                    if cur.rowcount:
                        created += 1
                except Exception:
                    continue
    brain.conn.commit()

    total_bridges = brain.conn.execute(
        "SELECT COUNT(*) FROM bridges"
    ).fetchone()[0]
    brain.close()
    return {
        "multi_region_labels": multi_region_labels,
        "bridges_created": created,
        "bridges_total": total_bridges,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-label-len", type=int, default=4)
    args = ap.parse_args()
    stats = build_bridges(args.db, args.min_label_len)
    print(f"Labels shared across >=2 regions: {stats['multi_region_labels']}")
    print(f"Bridges newly created:              {stats['bridges_created']}")
    print(f"Bridges total in DB:                {stats['bridges_total']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
