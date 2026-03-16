"""Pyodide bootstrap: import sara_brain, create Brain instance, expose helpers."""

import sys
import json

# The sara_brain package is mounted at /home/pyodide/sara_brain by the JS loader
sys.path.insert(0, "/home/pyodide")

from sara_brain.core.brain import Brain
from sara_brain.repl.commands import (
    cmd_teach, cmd_recognize, cmd_why, cmd_trace,
    cmd_neurons, cmd_paths, cmd_stats, cmd_similar, cmd_analyze,
    cmd_define, cmd_describe, cmd_associations,
)
from sara_brain.visualization.text_tree import render_paths_from

# Global brain instance
brain = None
_last_recognition = None  # Cache last recognition results for animation


def init_brain():
    """Create a fresh in-memory brain."""
    global brain
    if brain is not None:
        brain.close()
    brain = Brain(":memory:")
    return "Brain initialized."


def run_command(command_line):
    """Execute a REPL command and return the output string."""
    global brain
    if brain is None:
        init_brain()

    line = command_line.strip()
    if not line:
        return ""

    parts = line.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    dispatch = {
        "teach": cmd_teach,
        "recognize": cmd_recognize,
        "why": cmd_why,
        "trace": cmd_trace,
        "neurons": cmd_neurons,
        "paths": cmd_paths,
        "stats": cmd_stats,
        "similar": cmd_similar,
        "analyze": cmd_analyze,
        "define": cmd_define,
        "describe": cmd_describe,
        "associations": cmd_associations,
    }

    if cmd == "tree":
        if not args.strip():
            return "  Usage: tree <neuron>"
        return render_paths_from(brain, args.strip())

    if cmd == "help":
        return (
            "  Commands:\n"
            "    teach <statement>       — Teach a fact (e.g., teach apples are red)\n"
            "    recognize <a>, <b>, ... — Recognize from inputs\n"
            "    trace <neuron>          — Outgoing paths from a neuron\n"
            "    why <concept>           — Incoming paths to a concept\n"
            "    similar <neuron>        — Find similar neurons\n"
            "    analyze                 — Run full similarity analysis\n"
            "    tree <neuron>           — ASCII path tree\n"
            "    neurons                 — List all neurons\n"
            "    paths                   — List all paths\n"
            "    stats                   — Brain statistics\n"
            "    define <name>           — Define a new association type\n"
            "    describe <a> as <p>,... — Register properties under association\n"
            "    associations            — List all associations\n"
            "    reset                   — Reset brain to empty\n"
            "    seed                    — Load demo data\n"
            "    help                    — Show this help"
        )

    if cmd == "reset":
        init_brain()
        return "  Brain reset to empty state."

    if cmd == "seed":
        return _seed_brain()

    handler = dispatch.get(cmd)
    if handler is None:
        return f'  Unknown command: "{cmd}". Type "help" for available commands.'

    # For recognize, cache the results for wavefront animation
    if cmd == "recognize" and args.strip():
        global _last_recognition
        labels = [l.strip() for l in args.split(",") if l.strip()]
        results = brain.recognizer.recognize(labels)
        brain.conn.commit()
        # Cache path data for animation
        output = []
        for r in results:
            paths_data = []
            for trace in r.converging_paths:
                paths_data.append([n.id for n in trace.neurons])
            output.append({
                "neuron_id": r.neuron.id,
                "label": r.neuron.label,
                "confidence": r.confidence,
                "paths": paths_data,
            })
        _last_recognition = output
        # Format the text output using the existing formatter
        from sara_brain.repl.formatters import format_recognition
        return format_recognition(results)

    return handler(brain, args)


def get_graph_data():
    """Return JSON string of neurons and segments for D3 visualization."""
    global brain
    if brain is None:
        return json.dumps({"nodes": [], "links": []})

    neurons = brain.neuron_repo.list_all()
    segments = brain.segment_repo.list_all()

    nodes = []
    for n in neurons:
        nodes.append({
            "id": n.id,
            "label": n.label,
            "type": n.neuron_type.value,
        })

    links = []
    for s in segments:
        links.append({
            "source": s.source_id,
            "target": s.target_id,
            "relation": s.relation,
            "strength": s.strength,
        })

    return json.dumps({"nodes": nodes, "links": links})


def get_last_recognition_paths():
    """Return JSON with cached recognition path data for animation.

    Must be called after run_command('recognize ...') — does NOT re-run recognition.
    """
    global _last_recognition
    if _last_recognition is None:
        return json.dumps({"results": []})
    result = json.dumps({"results": _last_recognition})
    _last_recognition = None
    return result


def _seed_brain():
    """Load the apple demo data."""
    global brain
    if brain is None:
        init_brain()

    teachings = [
        "apples are red",
        "apples are round",
        "apples are sweet",
        "circles are round",
        "bananas are yellow",
        "bananas are sweet",
        "lemons are yellow",
        "lemons are sour",
    ]

    lines = ["  Loading demo data..."]
    for stmt in teachings:
        result = brain.teach(stmt)
        if result:
            lines.append(f"    taught: {stmt}")
    lines.append(f"  Done! Loaded {len(teachings)} facts.")
    return "\n".join(lines)


def export_db():
    """Export the SQLite database as a JSON snapshot."""
    global brain
    if brain is None:
        return json.dumps({})

    neurons = brain.neuron_repo.list_all()
    segments = brain.segment_repo.list_all()
    all_paths = brain.path_repo.list_all()

    data = {
        "version": 1,
        "neurons": [
            {"id": n.id, "label": n.label, "neuron_type": n.neuron_type.value,
             "created_at": n.created_at}
            for n in neurons
        ],
        "segments": [
            {"id": s.id, "source_id": s.source_id, "target_id": s.target_id,
             "relation": s.relation, "strength": s.strength,
             "traversals": s.traversals, "created_at": s.created_at,
             "last_used": s.last_used}
            for s in segments
        ],
        "paths": [
            {"id": p.id, "origin_id": p.origin_id, "terminus_id": p.terminus_id,
             "source_text": p.source_text, "created_at": p.created_at}
            for p in all_paths
        ],
        "path_steps": [],
    }

    # Get path steps
    for p in all_paths:
        steps = brain.path_repo.get_steps(p.id)
        for step in steps:
            data["path_steps"].append({
                "id": step.id,
                "path_id": step.path_id,
                "step_order": step.step_order,
                "segment_id": step.segment_id,
            })

    return json.dumps(data)


def import_db(json_str):
    """Import a JSON snapshot into a fresh brain."""
    global brain
    data = json.loads(json_str)

    # Start fresh
    init_brain()

    conn = brain.conn

    # Disable foreign keys temporarily for bulk import
    conn.execute("PRAGMA foreign_keys=OFF")

    # Import neurons
    for n in data.get("neurons", []):
        conn.execute(
            "INSERT INTO neurons (id, label, neuron_type, created_at, metadata) VALUES (?, ?, ?, ?, ?)",
            (n["id"], n["label"], n["neuron_type"], n.get("created_at", 0), None),
        )

    # Import segments
    for s in data.get("segments", []):
        conn.execute(
            "INSERT INTO segments (id, source_id, target_id, relation, strength, traversals, created_at, last_used) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (s["id"], s["source_id"], s["target_id"], s["relation"],
             s["strength"], s["traversals"], s.get("created_at", 0), s.get("last_used", 0)),
        )

    # Import paths
    for p in data.get("paths", []):
        conn.execute(
            "INSERT INTO paths (id, origin_id, terminus_id, source_text, created_at) VALUES (?, ?, ?, ?, ?)",
            (p["id"], p["origin_id"], p["terminus_id"], p.get("source_text"), p.get("created_at", 0)),
        )

    # Import path steps
    for ps in data.get("path_steps", []):
        conn.execute(
            "INSERT INTO path_steps (id, path_id, step_order, segment_id) VALUES (?, ?, ?, ?)",
            (ps["id"], ps["path_id"], ps["step_order"], ps["segment_id"]),
        )

    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()

    stats = brain.stats()
    return f"  Imported: {stats['neurons']} neurons, {stats['segments']} segments, {stats['paths']} paths."
