#!/usr/bin/env python3
"""Flatten a region-prefixed monolithic DB into a single flat brain.db.

biology2e.db stores 48 chapters as region-prefixed table sets
(ch1_neurons, ch1_segments, ch1_paths, ch1_path_steps,
ch1_segment_sources, ch2_neurons, ...). This script merges all of
them into one flat set of tables, deduplicating neurons by label so
that "cell" appearing in every chapter collapses into a single
neuron id. Segments, paths, path_steps, and segment_sources are
remapped accordingly.

Two-witness provenance is preserved via segment_sources; if the same
fact was taught from multiple chapters, each becomes a distinct
source_label row.

Usage:
    .venv/bin/python -m sara_brain.tools.flatten_monolithic \\
        --source biology2e.db --dest brain.db
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from sara_brain.storage.database import Database


def _list_regions(src: sqlite3.Connection) -> list[str]:
    try:
        rows = src.execute(
            "SELECT name FROM regions ORDER BY name"
        ).fetchall()
        if rows:
            return [r[0] for r in rows]
    except sqlite3.OperationalError:
        pass
    # Fallback: infer from table names
    tables = [r[0] for r in src.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    prefixes = set()
    for t in tables:
        if t.endswith("_neurons"):
            prefixes.add(t[: -len("_neurons")])
    return sorted(prefixes)


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _copy_neurons(src: sqlite3.Connection, dst: sqlite3.Connection,
                  region: str, label_to_id: dict[str, int],
                  id_map: dict[int, int]) -> int:
    """Copy neurons from {region}_neurons, deduplicating by label."""
    tbl = f"{region}_neurons"
    if not _has_table(src, tbl):
        return 0
    cur = dst.cursor()
    added = 0
    for row in src.execute(
        f"SELECT id, label, neuron_type, created_at, metadata FROM {tbl}"
    ).fetchall():
        src_id, label, ntype, created, meta = row
        label = (label or "").strip()
        if not label:
            continue
        key = label.lower()
        if key in label_to_id:
            id_map[src_id] = label_to_id[key]
            continue
        cur.execute(
            "INSERT INTO neurons (label, neuron_type, created_at, metadata) "
            "VALUES (?, ?, ?, ?)",
            (label, ntype or "concept", created or time.time(), meta),
        )
        new_id = cur.lastrowid
        label_to_id[key] = new_id
        id_map[src_id] = new_id
        added += 1
    return added


def _copy_segments(src: sqlite3.Connection, dst: sqlite3.Connection,
                   region: str, id_map: dict[int, int],
                   seg_map: dict[int, int]) -> int:
    tbl = f"{region}_segments"
    if not _has_table(src, tbl):
        return 0
    cur = dst.cursor()
    added = 0
    # Detect columns present in the source
    cols = [r[1] for r in src.execute(
        f"PRAGMA table_info({tbl})"
    ).fetchall()]
    has_op = "operation_tag" in cols
    has_traversals = "traversals" in cols
    has_refutations = "refutations" in cols
    has_last_used = "last_used" in cols

    sel_cols = ["id", "source_id", "target_id", "relation", "strength",
                "created_at"]
    if has_traversals:
        sel_cols.append("traversals")
    if has_refutations:
        sel_cols.append("refutations")
    if has_last_used:
        sel_cols.append("last_used")
    if has_op:
        sel_cols.append("operation_tag")
    sel = ", ".join(sel_cols)

    for row in src.execute(f"SELECT {sel} FROM {tbl}").fetchall():
        d = dict(zip(sel_cols, row))
        src_id, src_n, tgt_n = d["id"], d["source_id"], d["target_id"]
        if src_n not in id_map or tgt_n not in id_map:
            continue
        new_src = id_map[src_n]
        new_tgt = id_map[tgt_n]
        relation = d["relation"]
        # Upsert on UNIQUE(source_id, target_id, relation)
        existing = cur.execute(
            "SELECT id, strength, traversals, refutations "
            "FROM segments WHERE source_id=? AND target_id=? AND relation=?",
            (new_src, new_tgt, relation),
        ).fetchone()
        if existing:
            seg_map[src_id] = existing[0]
            # Accumulate traversals and refutations; keep max strength
            new_strength = max(existing[1], d["strength"])
            new_traversals = existing[2] + d.get("traversals", 0)
            new_refutations = existing[3] + d.get("refutations", 0)
            cur.execute(
                "UPDATE segments SET strength=?, traversals=?, "
                "refutations=? WHERE id=?",
                (new_strength, new_traversals, new_refutations, existing[0]),
            )
        else:
            cur.execute(
                "INSERT INTO segments (source_id, target_id, relation, "
                "strength, traversals, refutations, created_at, "
                "last_used, operation_tag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (new_src, new_tgt, relation, d["strength"],
                 d.get("traversals", 0), d.get("refutations", 0),
                 d.get("created_at") or time.time(),
                 d.get("last_used"), d.get("operation_tag")),
            )
            seg_map[src_id] = cur.lastrowid
            added += 1
    return added


def _copy_paths(src: sqlite3.Connection, dst: sqlite3.Connection,
                region: str, id_map: dict[int, int],
                seg_map: dict[int, int]) -> tuple[int, int]:
    ptbl = f"{region}_paths"
    stbl = f"{region}_path_steps"
    if not _has_table(src, ptbl):
        return (0, 0)
    cur = dst.cursor()
    path_map: dict[int, int] = {}
    for row in src.execute(
        f"SELECT id, origin_id, terminus_id, source_text, created_at "
        f"FROM {ptbl}"
    ).fetchall():
        src_pid, origin, terminus, text, created = row
        if origin not in id_map or terminus not in id_map:
            continue
        cur.execute(
            "INSERT INTO paths (origin_id, terminus_id, source_text, "
            "created_at) VALUES (?, ?, ?, ?)",
            (id_map[origin], id_map[terminus], text, created or time.time()),
        )
        path_map[src_pid] = cur.lastrowid
    steps_added = 0
    if _has_table(src, stbl):
        for row in src.execute(
            f"SELECT path_id, step_order, segment_id FROM {stbl}"
        ).fetchall():
            src_pid, order, src_seg = row
            if src_pid not in path_map or src_seg not in seg_map:
                continue
            try:
                cur.execute(
                    "INSERT INTO path_steps (path_id, step_order, "
                    "segment_id) VALUES (?, ?, ?)",
                    (path_map[src_pid], order, seg_map[src_seg]),
                )
                steps_added += 1
            except sqlite3.IntegrityError:
                # duplicate (path_id, step_order) — skip
                pass
    return (len(path_map), steps_added)


def _copy_segment_sources(src: sqlite3.Connection, dst: sqlite3.Connection,
                          region: str, seg_map: dict[int, int]) -> int:
    tbl = f"{region}_segment_sources"
    if not _has_table(src, tbl):
        return 0
    cur = dst.cursor()
    added = 0
    cols = [r[1] for r in src.execute(
        f"PRAGMA table_info({tbl})"
    ).fetchall()]
    has_created = "created_at" in cols
    sel = "segment_id, source_label" + (", created_at" if has_created else "")
    for row in src.execute(f"SELECT {sel} FROM {tbl}").fetchall():
        src_seg, label = row[0], row[1]
        created = row[2] if has_created else time.time()
        if src_seg not in seg_map:
            continue
        try:
            cur.execute(
                "INSERT INTO segment_sources (segment_id, source_label, "
                "created_at) VALUES (?, ?, ?)",
                (seg_map[src_seg], label, created),
            )
            added += 1
        except sqlite3.IntegrityError:
            pass
    return added


def flatten(source_db: Path, dest_db: Path) -> None:
    if dest_db.exists():
        raise FileExistsError(
            f"Refusing to overwrite {dest_db} — move it aside first."
        )
    src = sqlite3.connect(str(source_db))
    src.execute("PRAGMA journal_mode=WAL")

    # Creates brain.db with the standard schema
    dst_db = Database(str(dest_db))
    dst = dst_db.conn

    regions = _list_regions(src)
    print(f"Source: {source_db}  →  Dest: {dest_db}")
    print(f"Regions: {len(regions)}")

    # Accumulators
    label_to_id: dict[str, int] = {}
    total_neurons = 0
    total_segments = 0
    total_paths = 0
    total_steps = 0
    total_sources = 0

    t0 = time.time()
    for i, region in enumerate(regions, 1):
        # Per-region ID maps (src_id → dst_id)
        id_map: dict[int, int] = {}
        seg_map: dict[int, int] = {}
        n_add = _copy_neurons(src, dst, region, label_to_id, id_map)
        s_add = _copy_segments(src, dst, region, id_map, seg_map)
        p_add, st_add = _copy_paths(src, dst, region, id_map, seg_map)
        src_add = _copy_segment_sources(src, dst, region, seg_map)
        dst.commit()
        total_neurons += n_add
        total_segments += s_add
        total_paths += p_add
        total_steps += st_add
        total_sources += src_add
        print(f"  [{i}/{len(regions)}] {region}: +{n_add}n +{s_add}s "
              f"+{p_add}p +{st_add}st +{src_add}src")

    # Carry regions table
    if _has_table(src, "regions"):
        try:
            for row in src.execute(
                "SELECT name, description, created_at FROM regions"
            ).fetchall():
                dst.execute(
                    "INSERT OR IGNORE INTO regions (name, description, "
                    "created_at) VALUES (?, ?, ?)",
                    row,
                )
        except sqlite3.OperationalError:
            pass

    # Carry bridges (remap neuron IDs via label lookup per region)
    if _has_table(src, "bridges"):
        bridge_rows = src.execute(
            "SELECT source_region, source_neuron_id, target_region, "
            "target_neuron_id, relation FROM bridges"
        ).fetchall()
        carried = 0
        for sr, snid, tr, tnid, rel in bridge_rows:
            sl = src.execute(
                f"SELECT label FROM {sr}_neurons WHERE id=?", (snid,)
            ).fetchone()
            tl = src.execute(
                f"SELECT label FROM {tr}_neurons WHERE id=?", (tnid,)
            ).fetchone()
            if not sl or not tl:
                continue
            src_key = sl[0].strip().lower()
            tgt_key = tl[0].strip().lower()
            if src_key not in label_to_id or tgt_key not in label_to_id:
                continue
            # Bridges become regular segments with the bridge's relation.
            # UNIQUE(source_id, target_id, relation) dedups automatically.
            try:
                dst.execute(
                    "INSERT OR IGNORE INTO segments (source_id, target_id, "
                    "relation, strength, created_at) VALUES (?, ?, ?, ?, ?)",
                    (label_to_id[src_key], label_to_id[tgt_key],
                     rel or "same_as", 1.0, time.time()),
                )
                carried += 1
            except sqlite3.IntegrityError:
                pass
        dst.commit()
        print(f"Bridges carried as segments: {carried}/{len(bridge_rows)}")

    # Carry categories if present
    if _has_table(src, "categories"):
        for row in src.execute(
            "SELECT label, category FROM categories"
        ).fetchall():
            try:
                dst.execute(
                    "INSERT OR IGNORE INTO categories (label, category) "
                    "VALUES (?, ?)", row,
                )
            except sqlite3.OperationalError:
                break
        dst.commit()

    print()
    print(f"  neurons:          {total_neurons}")
    print(f"  segments:         {total_segments}")
    print(f"  paths:            {total_paths}")
    print(f"  path_steps:       {total_steps}")
    print(f"  segment_sources:  {total_sources}")
    print(f"  unique labels:    {len(label_to_id)}")
    print(f"  elapsed:          {time.time()-t0:.1f}s")

    dst.commit()
    dst_db.close()
    src.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--dest", required=True, type=Path)
    args = p.parse_args()
    flatten(args.source, args.dest)


if __name__ == "__main__":
    main()
