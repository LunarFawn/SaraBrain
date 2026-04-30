"""Tool definitions for sara_reader.

These are the retrieval tools the LLM-as-orchestrator can call. Each tool
has a provider-agnostic schema (JSON Schema for inputs) and an executor
function that runs against a Brain instance.

Write tools (teach, refute, ingest) are intentionally NOT exposed here.
sara_reader is read-only by default; consumer apps that need to teach
should use the lower-level Brain.teach_triple API directly with explicit
authorization from the user.
"""
from __future__ import annotations

from typing import Any, Callable

from sara_brain.core.brain import Brain


# Tool name -> (description, JSON-schema parameters, executor)
def _build_tool_registry() -> dict[str, dict[str, Any]]:
    return {
        "brain_explore": {
            "description": (
                "Walk Sara's graph outward from `label` to `depth` semantic "
                "hops in both directions. Returns the neighborhood of "
                "concepts and edges around the seed.\n\n"
                "STEPPED PROTOCOL — ALWAYS START AT depth=1:\n"
                "  1. First call: depth=1 on the most specific term in "
                "the question. Default is 1 for a reason.\n"
                "  2. If depth=1 returns enough to answer, ANSWER. Do not "
                "escalate.\n"
                "  3. If depth=1 found nothing but suggests alternate "
                "labels, retry depth=1 on a different label first.\n"
                "  4. Only escalate to depth=2 when you have a clear "
                "reason: the answer is one hop further.\n"
                "  5. Only escalate to depth=3 for conceptual questions.\n"
                "  6. depth=4 is for broad orientation only. If the "
                "result is TRUNCATED or shows hundreds of edges, you have "
                "over-queried — back off and re-seed with a more specific "
                "label at lower depth.\n\n"
                "Right pattern for a value question (e.g. 'what is the "
                "KDOFF for super-performing mode?'): seed='super-"
                "performing mode' depth=1.\n"
                "Wrong pattern: seed='KDOFF' depth=4 — floods you with "
                "every kdoff fact in the graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Concept to center the walk on (case-insensitive). Prefer the most specific label that names the concept you actually want.",
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Hop distance in semantic hops. ALWAYS start at 1. Escalate to 2 only if 1 was insufficient. 3 for conceptual questions only. 4 for broad orientation only.",
                        "default": 1,
                    },
                },
                "required": ["label"],
            },
            "executor": _exec_brain_explore,
        },
        "brain_why": {
            "description": (
                "Return paths terminating at `label` (incoming, single-hop). "
                "Use only when you need precise incoming-direction facts; "
                "for general retrieval use brain_explore instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                },
                "required": ["label"],
            },
            "executor": _exec_brain_why,
        },
        "brain_trace": {
            "description": (
                "Return paths originating from `label` (outgoing, single-hop)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                },
                "required": ["label"],
            },
            "executor": _exec_brain_trace,
        },
        "brain_recognize": {
            "description": (
                "Wavefront convergence from comma-separated seed labels. "
                "Use when identifying a concept from a set of properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "inputs": {
                        "type": "string",
                        "description": "Comma-separated seed labels",
                    },
                },
                "required": ["inputs"],
            },
            "executor": _exec_brain_recognize,
        },
        "brain_value": {
            "description": (
                "Return value / range / quantity edges for `concept`. "
                "Two modes — pick one per call:\n\n"
                "  TARGETED — pass `type` to ask for one specific kind "
                "of number. `type` is a relation-name fragment like "
                "'kdoff', 'kdon', 'ratio', 'value', 'fold_change'. "
                "Returns only edges whose relation contains that "
                "fragment. Use this when the question names exactly "
                "one quantity. Forces you to commit — never asks for "
                "two quantities at once.\n"
                "  ALL-PROPERTIES — omit `type`. Returns all direct "
                "value/property edges for the concept. Use only when "
                "the question is 'tell me the properties of X' with "
                "no specific quantity named.\n\n"
                "Either way, filters out part_of decomposition and "
                "describes bridges. Cite returned edges verbatim — do "
                "NOT generalize across labels or merge two relations "
                "into one threshold.\n\n"
                "COMPOUND CONCEPT RULE: when the question names a "
                "broad concept AND a specific quantity, prefer the "
                "compound concept label. Examples:\n"
                "  Q: 'highest KDOFF for SSNG1?' → "
                "brain_value(concept='ssng1 highest kdoff', type='value')\n"
                "  Q: 'KDOFF range for super-performing mode?' → "
                "brain_value(concept='super-performing mode', type='kdoff')\n"
                "  Q: 'KDON for super-performing mode?' → "
                "brain_value(concept='super-performing mode', type='kdon')\n\n"
                "If no edges match, returns an explicit 'not found' "
                "message — do not fabricate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "Concept to look up values for (case-insensitive)",
                    },
                    "type": {
                        "type": "string",
                        "description": "Optional. Relation-name fragment to filter by (e.g. 'kdoff', 'kdon', 'ratio', 'value'). Omit for all-properties mode.",
                    },
                },
                "required": ["concept"],
            },
            "executor": _exec_brain_value,
        },
        "brain_define": {
            "description": (
                "Return ONLY the definitional / identity edges for "
                "`concept` — what the concept IS, what measures it, "
                "what it stands for, what it is a subsystem of, what "
                "it's a synonym for. Filters relations to the "
                "definitional set: measures, measured_by, is_a, "
                "is_an_instance_of, is_subsystem_of, stands_for, "
                "defined_as, means, synonym_of.\n\n"
                "MANDATORY USAGE: Before mentioning any concept, "
                "acronym, or domain term in your final answer, call "
                "brain_define on it. If brain_define returns 'no "
                "definition found', either omit the term or quote "
                "only what the substrate provides — DO NOT invent "
                "an expansion or definition from training.\n\n"
                "This tool prevents acronym-expansion confabulation "
                "(e.g. inventing 'kill-dead-on-demand' for KDON when "
                "the substrate has 'aptamer affinity to on state "
                "measures kdon')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept": {
                        "type": "string",
                        "description": "Concept, acronym, or term to define from the substrate (case-insensitive)",
                    },
                },
                "required": ["concept"],
            },
            "executor": _exec_brain_define,
        },
        "brain_did_you_mean": {
            "description": (
                "Fuzzy-match a possibly-misspelled term against Sara's "
                "neuron labels. Returns nearest matches with descriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                },
                "required": ["term"],
            },
            "executor": _exec_brain_did_you_mean,
        },
    }


def _exec_brain_explore(brain: Brain, args: dict) -> str:
    label = args["label"]
    depth = args.get("depth", 1)
    try:
        depth = int(depth)
    except (TypeError, ValueError):
        depth = 1
    if depth < 1:
        depth = 1
    if depth > 4:
        depth = 4
    result = brain.neighborhood(label, depth=depth, max_edges=1000)
    return _format_neighborhood(result)


_NOISE_RELATIONS = {"part_of", "describes", "is_a"}

_DEFINITIONAL_RELATIONS = {
    "measures",
    "measured_by",
    "is_a",
    "is_an_instance_of",
    "is_subsystem_of",
    "stands_for",
    "defined_as",
    "means",
    "synonym_of",
}


def _exec_brain_define(brain: Brain, args: dict) -> str:
    concept = args["concept"].strip().lower()
    conn = brain.conn

    row = conn.execute(
        "SELECT id FROM neurons WHERE label = ?", (concept,)
    ).fetchone()
    if row is None:
        return (
            f"Sara has no neuron matching '{concept}'. "
            f"DO NOT invent a definition — either omit this term or "
            f"use brain_did_you_mean to find the closest substrate label."
        )
    neuron_id = row[0]

    attr_row = conn.execute(
        "SELECT id FROM neurons WHERE label = ?", (f"{concept}_attribute",)
    ).fetchone()
    attr_id = attr_row[0] if attr_row else None

    ids = [neuron_id] + ([attr_id] if attr_id is not None else [])
    placeholders = ",".join("?" * len(ids))

    edges = conn.execute(
        f"""
        SELECT n1.label, s.relation, n2.label
        FROM segments s
        JOIN neurons n1 ON s.source_id = n1.id
        JOIN neurons n2 ON s.target_id = n2.id
        WHERE (s.source_id IN ({placeholders}) OR s.target_id IN ({placeholders}))
        """,
        ids + ids,
    ).fetchall()

    definitional = [
        (src, rel, tgt)
        for src, rel, tgt in edges
        if rel in _DEFINITIONAL_RELATIONS
    ]

    if not definitional:
        return (
            f"No definitional edges found for '{concept}'. "
            f"The concept exists in the substrate but has no "
            f"definition/identity relation. DO NOT invent one — "
            f"either omit the term or quote only what other tools "
            f"return for this concept."
        )

    seen = set()
    lines = [f"Definition of {concept!r}:"]
    for src, rel, tgt in definitional:
        src_clean = src.replace("_attribute", "")
        tgt_clean = tgt.replace("_attribute", "")
        key = (src_clean, rel, tgt_clean)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  '{src_clean}' --[{rel}]--> '{tgt_clean}'")
    return "\n".join(lines)


def _exec_brain_value(brain: Brain, args: dict) -> str:
    concept = args["concept"].strip().lower()
    type_filter = args.get("type")
    if type_filter is not None:
        type_filter = type_filter.strip().lower() or None

    conn = brain.conn

    row = conn.execute(
        "SELECT id FROM neurons WHERE label = ?", (concept,)
    ).fetchone()
    if row is None:
        return f"Sara has no neuron matching '{concept}'."
    neuron_id = row[0]

    attr_row = conn.execute(
        "SELECT id FROM neurons WHERE label = ?", (f"{concept}_attribute",)
    ).fetchone()
    attr_id = attr_row[0] if attr_row else None

    ids = [neuron_id] + ([attr_id] if attr_id is not None else [])
    placeholders = ",".join("?" * len(ids))

    edges = conn.execute(
        f"""
        SELECT n1.label, s.relation, n2.label
        FROM segments s
        JOIN neurons n1 ON s.source_id = n1.id
        JOIN neurons n2 ON s.target_id = n2.id
        WHERE (s.source_id IN ({placeholders}) OR s.target_id IN ({placeholders}))
        """,
        ids + ids,
    ).fetchall()

    filtered = [
        (src, rel, tgt)
        for src, rel, tgt in edges
        if rel not in _NOISE_RELATIONS
    ]

    if type_filter:
        filtered = [
            (src, rel, tgt)
            for src, rel, tgt in filtered
            if type_filter in rel.lower()
        ]

    if not filtered:
        if type_filter:
            return (
                f"No '{type_filter}' edges found for '{concept}'. "
                f"Try omitting `type` to see all properties, or pick a "
                f"different concept."
            )
        return (
            f"No value-relations found for '{concept}'. "
            f"Sara has the concept but no direct quantity/property edges."
        )

    header = f"Value-relations for {concept!r}"
    if type_filter:
        header += f" filtered by type={type_filter!r}"
    seen = set()
    lines = [header + ":"]
    for src, rel, tgt in filtered:
        src_clean = src.replace("_attribute", "")
        tgt_clean = tgt.replace("_attribute", "")
        key = (src_clean, rel, tgt_clean)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  '{src_clean}' --[{rel}]--> '{tgt_clean}'")
    return "\n".join(lines)


def _exec_brain_why(brain: Brain, args: dict) -> str:
    label = args["label"].strip().lower()
    traces = brain.why(label)
    if not traces:
        return f"No paths lead to '{label}'."
    return "\n".join(f"  {t}" for t in traces)


def _exec_brain_trace(brain: Brain, args: dict) -> str:
    label = args["label"].strip().lower()
    traces = brain.trace(label)
    if not traces:
        return f"No outgoing paths from '{label}'."
    return "\n".join(f"  {t}" for t in traces[:50])


def _exec_brain_recognize(brain: Brain, args: dict) -> str:
    results = brain.recognize(args["inputs"])
    if not results:
        return "No recognition."
    lines = []
    for r in results:
        lines.append(f"  {r.neuron.label} ({len(r.converging_paths)} converging paths)")
    return "\n".join(lines)


def _exec_brain_did_you_mean(brain: Brain, args: dict) -> str:
    candidates = brain.did_you_mean(args["term"])
    if not candidates:
        return f"No fuzzy matches for '{args['term']}'."
    lines = [f"Did you mean (for '{args['term']}')?"]
    for c in candidates[:8]:
        desc = f" — {c['description']}" if c.get("description") else ""
        lines.append(f"  - {c['label']} ({c['type']}){desc}")
    return "\n".join(lines)


def _format_neighborhood(result: dict) -> str:
    if not result.get("seed_found"):
        return f"Sara has no neuron matching '{result.get('seed')}'."
    lines = [
        f"Neighborhood of {result['seed']!r} "
        f"(depth={result['depth_queried']}, "
        f"{result['total_neurons']} neurons, {result['total_edges']} edges"
        f"{', TRUNCATED' if result.get('truncated') else ''})"
    ]
    seen_edges = set()
    for d in sorted(result.get("neurons_by_depth", {}).keys()):
        concepts = [
            lbl for lbl in result["neurons_by_depth"][d]
            if not lbl.endswith("_attribute")
        ]
        if not concepts:
            continue
        lines.append(f"  depth {d}: {', '.join(sorted(concepts)[:25])}")
    lines.append("Edges:")
    for e in result.get("edges", []):
        src, rel, tgt = e["source"], e["relation"], e["target"]
        # Skip the structural describes-edges that connect attribute-> concept
        if src.endswith("_attribute") and rel == "describes":
            continue
        key = (src, rel, tgt)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        marker = ""
        if e.get("strength", 0) < 0:
            marker = " [REFUTED]"
        lines.append(f"  {src!r} --[{rel}]--> {tgt!r}{marker}")
    return "\n".join(lines)


# Public registry — built lazily so executors are bound after definition.
TOOLS = _build_tool_registry()


def execute_tool(brain: Brain, tool_name: str, arguments: dict) -> str:
    """Run a tool call against the brain and return the formatted output."""
    if tool_name not in TOOLS:
        return f"<<unknown tool: {tool_name}>>"
    try:
        return TOOLS[tool_name]["executor"](brain, arguments)
    except Exception as e:
        return f"<<tool error in {tool_name}: {e}>>"


def get_tool_schemas() -> list[dict]:
    """Return all tool schemas in a provider-agnostic shape."""
    return [
        {
            "name": name,
            "description": meta["description"],
            "parameters": meta["parameters"],
        }
        for name, meta in TOOLS.items()
    ]
