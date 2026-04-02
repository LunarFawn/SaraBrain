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
    cmd_query, cmd_questions, cmd_categorize, cmd_categories,
)
from sara_brain.visualization.text_tree import render_paths_from

# Global brain instance
brain = None
_last_recognition = None  # Cache last recognition results for animation
_perception_state = None  # {label, all_observations, top_guess} — set by JS after perception loop


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
        "questions": cmd_questions,
        "categorize": cmd_categorize,
        "categories": cmd_categories,
    }

    if cmd == "tree":
        if not args.strip():
            return "  Usage: tree <neuron>"
        return render_paths_from(brain, args.strip())

    if cmd == "perceive":
        return "  Use the Vision panel to upload an image, or type: help perceive"

    if cmd == "no":
        return cmd_no_web(args.strip().lower())

    if cmd == "see":
        return cmd_see_web(args.strip().lower())

    if cmd == "ask":
        return _ask_sara(args)

    if cmd == "help":
        return (
            "  Commands:\n"
            "    teach <statement>           — Teach a fact (e.g., teach apples are red)\n"
            "    recognize <a>, <b>, ...     — Recognize from inputs\n"
            "    trace <neuron>              — Outgoing paths from a neuron\n"
            "    why <concept>               — Incoming paths to a concept\n"
            "    similar <neuron>            — Find similar neurons\n"
            "    analyze                     — Run full similarity analysis\n"
            "    ask <question>              — Ask in plain English (no LLM needed)\n"
            "    define <assoc> <qword>      — Define association (e.g., define taste how)\n"
            "    describe <a> as <props>     — Register properties under an association\n"
            "    associations                — List all associations and properties\n"
            "    questions                   — List all question word patterns\n"
            "    <qword> <concept> <assoc>   — Query (e.g., how apple taste)\n"
            "    categorize <concept> <cat>  — Tag a concept (e.g., categorize apple item)\n"
            "    categories                  — List all categories\n"
            "    tree <neuron>               — ASCII path tree\n"
            "    neurons                     — List all neurons\n"
            "    paths                       — List all paths\n"
            "    stats                       — Brain statistics\n"
            "    perceive                    — Upload an image (via Vision panel)\n"
            "    no <correct_label>          — Correct last perception: no ball\n"
            "    see <property>              — Point out a missed property: see seams\n"
            "    reset                       — Reset brain to empty\n"
            "    seed                        — Load fruit demo data\n"
            "    seed wiki                   — Load Wikipedia demo (Newton + Solar System)\n"
            "    help                        — Show this help"
        )

    if cmd == "reset":
        init_brain()
        return "  Brain reset to empty state."

    if cmd == "seed":
        if args.strip().lower() == "wiki":
            return _seed_wiki()
        return _seed_brain()

    handler = dispatch.get(cmd)
    if handler is None:
        # Check if cmd is a registered question word (e.g., "how", "what")
        associations = brain.resolve_question_word(cmd)
        if associations:
            return cmd_query(brain, cmd, args)
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


def _seed_wiki():
    """Load the Wikipedia demo: Newton's Laws + Solar System."""
    global brain

    try:
        # Fetch the pre-baked JSON from the server
        from pyodide.http import open_url
        raw = open_url("python/wiki_demo_brain.json").read()
    except Exception:
        return "  Error: could not load wiki_demo_brain.json"

    result = import_db(raw)
    return (
        "  Loaded Wikipedia demo: Newton's Laws of Motion + Solar System\n"
        "  Sources:\n"
        "    https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion\n"
        "    https://en.wikipedia.org/wiki/Solar_System\n"
        "  Content licensed under CC BY-SA 4.0.\n"
        f"{result}\n"
        "  Try: ask what is gravity  |  ask how does orbit work  |  ask link between force and orbit"
    )


def _ask_sara(question):
    """Parse a natural language question into Sara brain commands. No LLM needed.

    Patterns:
      what is <concept>           -> why <concept>
      explain <concept>           -> why <concept>
      how does <concept> work     -> trace <concept>
      where does <concept> lead   -> trace <concept>
      link between <a> and <b>    -> recognize <a>, <b>
      show link between <a> and <b> -> recognize <a>, <b>
      compare <a> and <b>         -> similar <a> + similar <b>
      what connects <a> and <b>   -> recognize <a>, <b>
      what equation <concept>     -> why <concept> (filtered to equations)
      equation for <concept>      -> why <concept> (filtered to equations)
    """
    global brain
    if brain is None:
        return "  Brain is empty. Try: seed wiki"

    q = question.strip().lower()
    if not q:
        return (
            "  Ask Sara a question:\n"
            "    what is <concept>         — What Sara knows about it\n"
            "    explain <concept>         — All paths leading to it\n"
            "    how does <concept> work   — Where it leads\n"
            "    link between <a> and <b>  — Find cross-domain connections\n"
            "    compare <a> and <b>       — Find shared paths\n"
            "    equation for <concept>    — Show equations\n"
            "  Examples:\n"
            "    ask what is gravity\n"
            "    ask how does orbit work\n"
            "    ask link between force and orbit\n"
            "    ask equation for force"
        )

    # Strip trailing ? and common filler
    q = q.rstrip("?!.")
    for filler in ["can you ", "please ", "could you ", "tell me ", "show me "]:
        if q.startswith(filler):
            q = q[len(filler):]

    # --- equation for <concept> ---
    for prefix in ["equation for ", "what equation ", "what is the equation for ",
                   "what is the equation of ", "equations for ", "formula for "]:
        if q.startswith(prefix):
            concept = q[len(prefix):].strip()
            return _ask_equation(concept)

    # --- link between <a> and <b> ---
    for prefix in ["link between ", "show link between ", "show a link between ",
                   "connection between ", "what connects ", "what links ",
                   "how are ", "how is "]:
        if q.startswith(prefix):
            rest = q[len(prefix):]
            # Handle "<a> and <b>" or "<a> to <b>" or "<a> related to <b>"
            for sep in [" and ", " to ", " related to ", " connected to ", " with "]:
                if sep in rest:
                    parts = rest.split(sep, 1)
                    a = parts[0].strip().rstrip(" related connected")
                    b = parts[1].strip()
                    return _ask_link(a, b)

    # --- compare <a> and <b> ---
    if q.startswith("compare "):
        rest = q[8:]
        if " and " in rest:
            a, b = rest.split(" and ", 1)
            return _ask_compare(a.strip(), b.strip())

    # --- how does <concept> work ---
    for prefix in ["how does ", "how do ", "where does ", "where do "]:
        if q.startswith(prefix):
            rest = q[len(prefix):]
            for suffix in [" work", " lead", " connect", " flow"]:
                if rest.endswith(suffix):
                    rest = rest[:-len(suffix)]
            return _ask_trace(rest.strip())

    # --- what is / explain / describe / tell me about ---
    for prefix in ["what is ", "what are ", "explain ", "describe ",
                   "about ", "tell me about ", "what do you know about "]:
        if q.startswith(prefix):
            concept = q[len(prefix):].strip()
            return _ask_why(concept)

    # Fallback: treat the whole thing as a concept lookup
    return _ask_why(q)


def _ask_why(concept):
    """What is <concept> — show all paths leading to it."""
    from sara_brain.repl.formatters import format_why
    traces = brain.why(concept)
    if not traces:
        # Try trace instead (maybe it's a property, not a concept)
        traces = brain.trace(concept)
        if not traces:
            return f'  I don\'t know about "{concept}" yet. Try: teach {concept} is ...'
        lines = [f'  What I know starting from "{concept}":']
        for t in traces[:15]:
            lines.append(f"    {t}")
        if len(traces) > 15:
            lines.append(f"    ... and {len(traces) - 15} more paths")
        return "\n".join(lines)

    lines = [f'  What I know about "{concept}":']
    for t in traces:
        src = ""
        if t.source_text:
            src = f"  (from: {t.source_text})"
        lines.append(f"    {t}{src}")
    return "\n".join(lines)


def _ask_trace(concept):
    """How does <concept> work — trace outgoing paths."""
    traces = brain.trace(concept)
    if not traces:
        return f'  I don\'t know where "{concept}" leads. Try: teach {concept} is ...'
    lines = [f'  How "{concept}" connects to other concepts:']
    for t in traces[:20]:
        lines.append(f"    {t}")
    if len(traces) > 20:
        lines.append(f"    ... and {len(traces) - 20} more paths")
    return "\n".join(lines)


def _ask_link(a, b):
    """Link between <a> and <b> — recognize to find convergence."""
    from sara_brain.repl.formatters import format_recognition
    labels = [a.strip(), b.strip()]
    results = brain.recognizer.recognize(labels)
    brain.conn.commit()

    # Cache for animation
    global _last_recognition
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

    if not results:
        return f'  No connection found between "{a}" and "{b}".'

    # Filter to concept neurons with 2+ converging paths
    strong = [r for r in results if r.confidence >= 2 and r.neuron.neuron_type.value == "concept"]
    lines = [f'  Connections between "{a}" and "{b}":']
    shown = strong if strong else results[:10]
    for r in shown[:10]:
        lines.append(f"    {r.neuron.label} ({r.confidence} converging paths)")
        for t in r.converging_paths:
            lines.append(f"      {t}")
    if len(results) > 10:
        lines.append(f"    ... and {len(results) - 10} more")
    return "\n".join(lines)


def _ask_compare(a, b):
    """Compare <a> and <b> — find shared downstream paths."""
    links_a = brain.get_similar(a)
    links_b = brain.get_similar(b)
    all_links = links_a + links_b
    if not all_links:
        # Run analyze first
        brain.analyze_similarity()
        links_a = brain.get_similar(a)
        links_b = brain.get_similar(b)
        all_links = links_a + links_b

    if not all_links:
        return f'  No similarity data for "{a}" or "{b}". Try: analyze'

    lines = [f'  Comparing "{a}" and "{b}":']
    seen = set()
    for link in all_links:
        key = (link.neuron_a_label, link.neuron_b_label)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"    {link.neuron_a_label} <-> {link.neuron_b_label} "
                     f"(shared: {link.shared_paths}, overlap: {link.overlap_ratio:.0%})")
    return "\n".join(lines)


def _ask_equation(concept):
    """Show equations related to a concept."""
    traces = brain.why(concept)
    # Also check trace (outgoing)
    traces += brain.trace(concept)

    equations = []
    seen = set()
    for t in traces:
        src = t.source_text or ""
        if "equation" in src.lower() or "equals" in src.lower() or "formula" in src.lower():
            if src not in seen:
                seen.add(src)
                equations.append(src)
        # Also check neuron labels for equation content
        for n in t.neurons:
            if "equals" in n.label or "equation" in n.label:
                if n.label not in seen:
                    seen.add(n.label)
                    equations.append(n.label)

    if not equations:
        return f'  No equations found for "{concept}". Try: ask what is {concept}'

    lines = [f'  Equations related to "{concept}":']
    for eq in equations:
        lines.append(f"    {eq}")
    return "\n".join(lines)


def set_perception_state(state_json):
    """Called from JS after perception loop. Stores state for no/see commands."""
    global _perception_state
    _perception_state = json.loads(state_json)


def get_question_words():
    """Return JSON of brain.list_question_words()."""
    global brain
    if brain is None:
        return json.dumps({})
    qwords = brain.list_question_words()
    return json.dumps(qwords)


def get_candidate_properties(label):
    """Return JSON list of properties Sara knows about a concept (via brain.why)."""
    global brain
    if brain is None:
        return json.dumps([])
    props = []
    traces = brain.why(label)
    for trace in traces:
        if trace.neurons:
            props.append(trace.neurons[0].label)
    return json.dumps(props)


def cmd_no_web(correct_label):
    """Correction: teach correct identity + transfer all observed properties."""
    global brain, _perception_state
    if not correct_label:
        return "  Usage: no <correct_label>"
    if _perception_state is None:
        return "  No perception to correct. Use the Vision panel first."

    image_label = _perception_state["label"]
    all_obs = _perception_state.get("all_observations", [])
    old_guess = _perception_state.get("top_guess")

    # Teach identity
    brain.teach(f"{image_label} is {correct_label}")

    # Transfer all observed properties to the correct concept
    taught = []
    for prop in all_obs:
        r = brain.teach(f"{correct_label} is {prop}")
        if r is not None:
            taught.append(prop)

    brain.conn.commit()

    lines = []
    if old_guess:
        lines.append(f"  Corrected: was {old_guess}, now {correct_label}")
    else:
        lines.append(f"  Corrected: {image_label} is {correct_label}")
    lines.append(f"  Taught {len(taught)} properties to {correct_label}")
    return "\n".join(lines)


def cmd_see_web(property_label):
    """Parent points out: teach last perceived image a new property."""
    global brain, _perception_state
    if not property_label:
        return "  Usage: see <property>"
    if _perception_state is None:
        return "  No perception active. Use the Vision panel first."

    image_label = _perception_state["label"]
    r = brain.teach(f"{image_label} is {property_label}")
    brain.conn.commit()

    if r is not None:
        _perception_state.setdefault("all_observations", []).append(property_label)
        return f"  Taught: {image_label} is {property_label}"
    return f"  Already knew: {image_label} is {property_label}"


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
