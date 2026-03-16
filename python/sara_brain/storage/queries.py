from __future__ import annotations

import sqlite3


def traverse_from(conn: sqlite3.Connection, start_neuron_id: int, max_depth: int = 10) -> list[tuple]:
    """Recursive CTE: find all neurons reachable from start_neuron_id.

    Returns rows of (neuron_id, depth, path_text) where path_text is
    a comma-separated chain of neuron labels.
    """
    sql = """
    WITH RECURSIVE walk(neuron_id, depth, path_text, visited) AS (
        SELECT n.id, 0, n.label, ',' || CAST(n.id AS TEXT) || ','
        FROM neurons n
        WHERE n.id = ?

        UNION ALL

        SELECT seg.target_id,
               w.depth + 1,
               w.path_text || ' → ' || nt.label,
               w.visited || CAST(seg.target_id AS TEXT) || ','
        FROM walk w
        JOIN segments seg ON seg.source_id = w.neuron_id
        JOIN neurons nt ON nt.id = seg.target_id
        WHERE w.depth < ?
          AND w.visited NOT LIKE '%,' || CAST(seg.target_id AS TEXT) || ',%'
    )
    SELECT neuron_id, depth, path_text FROM walk WHERE depth > 0 ORDER BY depth
    """
    return conn.execute(sql, (start_neuron_id, max_depth)).fetchall()


def find_intersections(conn: sqlite3.Connection, start_ids: list[int], max_depth: int = 10) -> dict[int, list[int]]:
    """Find neurons reachable from multiple start points.

    Returns {neuron_id: [list of start_ids that reach it]}.
    Only includes neurons reached by 2+ start points.
    """
    reachable: dict[int, set[int]] = {}
    for sid in start_ids:
        rows = traverse_from(conn, sid, max_depth)
        for neuron_id, _depth, _path in rows:
            reachable.setdefault(neuron_id, set()).add(sid)

    return {nid: sorted(sources) for nid, sources in reachable.items() if len(sources) >= 2}
