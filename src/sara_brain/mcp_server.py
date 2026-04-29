"""Sara Brain MCP Server — persistent brain accessible by any MCP client.

The brain stays alive. LLMs connect and disconnect. Knowledge persists.

Usage:
    # Run directly
    python -m sara_brain.mcp_server

    # Add to Claude Code
    claude mcp add sara-brain -- python -m sara_brain.mcp_server

    # Custom database
    SARA_DB=/path/to/brain.db python -m sara_brain.mcp_server
"""

from __future__ import annotations

import os
import sys

from mcp.server.fastmcp import FastMCP

from .config import default_db_path
from .core.brain import Brain

# ── Initialize ──

DB_PATH = os.environ.get("SARA_DB", default_db_path())

mcp = FastMCP(
    "sara-brain",
    instructions=(
        "Sara Brain — path-of-thought knowledge system. Persistent memory "
        "that never forgets. The LLM is the senses, Sara is the brain.\n\n"
        "Retrieval strategy — prefer broader tools first:\n"
        "  1. brain_explore(label, depth=3) — RECOMMENDED for 'what is X?' "
        "     questions. Walks outward in both directions, surfaces the "
        "     author's full framing plus associative context across adjacent "
        "     neurons in one call. Sara is designed so the associative 'noise' "
        "     IS the signal the LLM consumer reasons with — more context is "
        "     better. Use this FIRST for any substantive question about a "
        "     taught topic.\n"
        "  2. brain_query(label) — 1-hop, both directions. Fallback when "
        "     brain_explore is too broad.\n"
        "  3. brain_why(label) / brain_trace(label) — 1-hop, single direction. "
        "     Use only when you need precise incoming/outgoing distinction.\n"
        "  4. brain_recognize(a, b, c) — wavefront convergence from multiple "
        "     seeds. Use when identifying a concept from properties.\n"
        "  5. brain_did_you_mean(label) — fuzzy match when spelling is uncertain.\n\n"
        "Teaching: brain_teach_triple(subject, relation, obj) is canonical; "
        "labels are lowercased unless prefixed with 'CAPITAL:'."
    ),
)

# Lazy brain initialization (connected on first tool call)
_brain: Brain | None = None


def _get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain(DB_PATH)
    return _brain


# ── Query Tools ──


@mcp.tool()
def brain_query(topic: str) -> str:
    """Query Sara Brain for everything she knows about a topic.

    Returns paths leading to and from the concept. Uses fuzzy matching
    so misspellings and variants are handled automatically.
    """
    brain = _get_brain()
    label = topic.strip().lower()

    traces_to = brain.why(label)
    traces_from = brain.trace(label)

    if not traces_to and not traces_from:
        # Try did_you_mean
        candidates = brain.did_you_mean(topic)
        if candidates:
            lines = [f"Sara doesn't know '{topic}'. Did you mean:"]
            for c in candidates:
                desc = f" — {c['description']}" if c['description'] else ""
                lines.append(f"  - {c['label']} ({c['type']}){desc}")
            return "\n".join(lines)
        return f"Sara doesn't know about '{topic}'."

    lines = []
    if traces_to:
        lines.append(f"Paths leading to '{label}':")
        for t in traces_to:
            src = f' (from: "{t.source_text}")' if t.source_text else ""
            lines.append(f"  {t}{src}")
    if traces_from:
        lines.append(f"Paths from '{label}':")
        for t in traces_from:
            src = f' (from: "{t.source_text}")' if t.source_text else ""
            lines.append(f"  {t}{src}")
    return "\n".join(lines)


@mcp.tool()
def brain_recognize(inputs: str) -> str:
    """Give Sara properties and see what she recognizes.

    Uses parallel wavefront propagation — launches wavefronts from each
    input property and finds where they converge. That's recognition.

    Args:
        inputs: Comma-separated properties (e.g., "red, round, sweet")
    """
    brain = _get_brain()
    results = brain.recognize(inputs)
    if not results:
        return "Sara doesn't recognize anything from those inputs."
    lines = []
    for r in results:
        lines.append(f"  {r.neuron.label} ({r.confidence} converging paths)")
        for trace in r.converging_paths:
            lines.append(f"    path: {trace}")
    return "\n".join(lines)


@mcp.tool()
def brain_why(label: str) -> str:
    """Show all paths that lead TO a neuron (reverse lookup).

    Answers: "Why does Sara know about X? What evidence leads to it?"
    """
    brain = _get_brain()
    traces = brain.why(label)
    if not traces:
        return f"No paths lead to '{label}'."
    lines = [f"Paths to '{label}':"]
    for t in traces:
        src = f' (from: "{t.source_text}")' if t.source_text else ""
        lines.append(f"  {t}{src}")
    return "\n".join(lines)


@mcp.tool()
def brain_trace(label: str) -> str:
    """Show all paths going OUT from a neuron.

    Answers: "What does X connect to? Where do paths from X lead?"
    """
    brain = _get_brain()
    traces = brain.trace(label)
    if not traces:
        return f"No outgoing paths from '{label}'."
    lines = [f"Paths from '{label}':"]
    for t in traces[:20]:  # Cap at 20 for readability
        lines.append(f"  {t}")
    if len(traces) > 20:
        lines.append(f"  ... and {len(traces) - 20} more")
    return "\n".join(lines)


@mcp.tool()
def brain_teach(statement: str) -> str:
    """Teach Sara a fact. She parses it into a neuron chain and remembers it forever.

    Examples:
        "apples are red"
        "RNA is a mechanical system"
        "metformin requires kidney monitoring"

    The fact is stored as a path: property → relation → concept.
    Sara never forgets what she's taught.
    """
    brain = _get_brain()
    result = brain.teach(statement)
    if result is None:
        return f"Could not parse: '{statement}'. Try 'X is/are Y' or 'X contains/requires/includes Y'."
    return f"Learned: {result.path_label} (path #{result.path_id})"


@mcp.tool()
def brain_explore(label: str, depth: int = 3) -> str:
    """Walk Sara's graph outward from ``label`` to ``depth`` hops in
    both directions. Surfaces the author's specific framing across a
    connected region of the substrate in a single call.

    **Prefer this for exploratory "what is X?" questions about any
    taught topic.** It returns more context than brain_why (1-hop,
    incoming only) or brain_trace (1-hop, outgoing only) and doesn't
    suffer from the direction-asymmetry issue those tools have.

    **Default is depth=3** to return a rich associative neighborhood.
    Sara's design philosophy is that the noise IS the signal for an
    LLM consumer — more context gives the reader more to reason with.
    Reduce to depth=2 or 1 only if the response is obviously too broad.

    When to use which tool:
    - brain_explore(X, depth=3)  — rich neighborhood for a concept (default)
    - brain_why(X)               — only paths terminating at X
    - brain_trace(X)             — only paths originating from X
    - brain_recognize(a, b, c)   — wavefront convergence from seeds
    - brain_did_you_mean(X)      — fuzzy-match X when spelling uncertain

    Args:
        label: concept to center the walk on (case-insensitive).
        depth: hop distance; 1, 2, 3, or 4. Default 3.

    Returns:
        Structured summary: neurons found grouped by distance from
        seed, plus each edge (source --[relation]--> target) with its
        strength and the depth at which it was discovered. Internal
        ``*_attribute`` chain nodes are shown but marked; the semantic
        triples can be read off by ignoring them.
    """
    brain = _get_brain()
    if depth < 1:
        depth = 1
    if depth > 4:
        depth = 4
    # Default max_edges raised to 1000 so a rich depth-3 walk on a
    # substrate hub doesn't truncate too aggressively.
    result = brain.neighborhood(label, depth=depth, max_edges=1000)

    if not result["seed_found"]:
        candidates = brain.did_you_mean(label)
        lines = [f"Sara doesn't have '{label}' as a neuron."]
        if candidates:
            lines.append("Did you mean:")
            for c in candidates[:6]:
                lines.append(f"  - {c['label']} ({c['type']})")
        return "\n".join(lines)

    # Build the formatted output.
    lines = [
        f"Neighborhood of {result['seed']!r}  "
        f"(depth={result['depth_queried']}, "
        f"{result['total_neurons']} neurons, {result['total_edges']} edges"
        f"{', TRUNCATED at max_edges=500' if result['truncated'] else ''})"
    ]
    lines.append("")

    # Group edges by depth; within a depth, sort to make output readable.
    edges_by_depth: dict[int, list[dict]] = {}
    for e in result["edges"]:
        edges_by_depth.setdefault(e["depth_seen"], []).append(e)

    # Emit neurons by depth for a quick orientation.
    lines.append("Neurons reachable (by hop distance from seed):")
    for d in sorted(result["neurons_by_depth"].keys()):
        concepts = [
            lbl for lbl in result["neurons_by_depth"][d]
            if not lbl.endswith("_attribute")
        ]
        attrs = [
            lbl for lbl in result["neurons_by_depth"][d]
            if lbl.endswith("_attribute")
        ]
        lines.append(f"  depth {d}: {len(concepts)} concepts"
                     f"{f' + {len(attrs)} internal attribute nodes' if attrs else ''}")
        for lbl in sorted(concepts):
            lines.append(f"    - {lbl}")
    lines.append("")

    # Emit edges by depth; skip pure attribute-to-attribute edges and
    # the redundant "describes" edges that connect attribute → concept.
    # Readers get the semantic substance, not the plumbing.
    lines.append("Edges (source --[relation]--> target):")
    for d in sorted(edges_by_depth.keys()):
        lines.append(f"")
        lines.append(f"  Discovered at depth {d}:")
        seen_keys = set()
        for e in edges_by_depth[d]:
            src, rel, tgt = e["source"], e["relation"], e["target"]
            # Skip the structural describe-edges (attr → concept) —
            # these are just chain plumbing and would double every
            # semantic edge.
            if src.endswith("_attribute") and rel == "describes":
                continue
            key = (src, rel, tgt)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            strength_marker = ""
            if e["strength"] < 0:
                strength_marker = "  [REFUTED]"
            elif e["strength"] >= 2.0:
                strength_marker = "  [strong]"
            lines.append(f"    {src!r} --[{rel}]--> {tgt!r}{strength_marker}")

    return "\n".join(lines)


@mcp.tool()
def brain_teach_triple(
    subject: str,
    relation: str,
    obj: str,
    source: str | None = None,
) -> str:
    """Teach Sara a fact as an explicit (subject, relation, object) triple.

    This is the canonical teach path for LLM callers. No parser — the
    LLM has already done the parsing by deciding the triple. Compound
    multi-word terms like "molecular snare" or "marker theory" land in
    the graph verbatim, not reduced to head nouns.

    **CASE HANDLING:** labels are normalized to lowercase by default so
    that "RNA" and "rna" resolve to the same neuron (case fragmentation
    was a common issue before this was added). If you genuinely need
    case preserved (e.g., to distinguish a proper-noun acronym from a
    common-word collision), prefix the label with "CAPITAL:" —
    everything after the colon is kept verbatim. Examples:
        subject="RNA"          → stored as "rna"
        subject="CAPITAL:RNA"  → stored as "RNA"
        subject="marker theory" → stored as "marker theory"

    Default to lowercase; only use CAPITAL: when case genuinely matters.

    Prefer this over brain_teach for technical prose, novel terminology,
    or any source where the LLM wants to preserve distinctive multi-word
    labels.

    Args:
        subject: the subject neuron label (lowercased unless CAPITAL:)
        relation: the verb/relation connecting subject to object
        obj: the object neuron label (lowercased unless CAPITAL:)
        source: optional source tag (e.g., "aptamer_paper") for
                provenance and two-witness confirmation

    Returns:
        A string describing what was stored, or the error if anything
        blocked the write.
    """
    brain = _get_brain()
    try:
        result = brain.teach_triple(
            subject, relation, obj, source_label=source,
        )
    except PermissionError as e:
        return f"Ethics gate blocked the teach: {e}"
    if result is None:
        return (
            f"Could not store: ({subject!r}, {relation!r}, {obj!r}). "
            "This is unusual for teach_triple — check that all fields "
            "are non-empty."
        )
    return f"Learned: {result.path_label} (path #{result.path_id})"


@mcp.tool()
def brain_refute(statement: str) -> str:
    """Refute a fact in Sara Brain. Sara never deletes — she marks the
    claim as known-to-be-false. The path stays as evidence of what was
    once claimed; its strength goes negative.

    Use this when correcting a hallucination, marking a debunked claim,
    or recording that something is no longer true.

    Examples:
        "the earth is flat"
        "vitamin C cures the common cold"
        "metformin is contraindicated with insulin"

    Repeated refutation drives strength further negative. Recognition
    will then weight this concept as actively-known-false.
    """
    brain = _get_brain()
    result = brain.refute(statement)
    if result is None:
        return f"Could not parse: '{statement}'. Try 'X is/are Y' format."
    return f"Refuted: {result.path_label} (path #{result.path_id}, marked as known-to-be-false)"


@mcp.tool()
def brain_refute_triple(
    subject: str,
    relation: str,
    obj: str,
) -> str:
    """Refute a fact stored as an explicit (subject, relation, object) triple.

    No parser — labels are matched exactly as stored (after case normalization),
    so compound terms like "molecular snare" are found and marked correctly.
    Use this instead of brain_refute when the fact was originally taught via
    brain_teach_triple.

    Sara never deletes — the path stays as evidence; its strength goes negative.
    Repeated refutation drives strength further negative.

    Args:
        subject: the subject neuron label (same value used in brain_teach_triple)
        relation: the relation verb (same value used in brain_teach_triple)
        obj: the object neuron label (same value used in brain_teach_triple)

    Returns:
        Confirmation with path label and id, or error if blocked.
    """
    brain = _get_brain()
    try:
        result = brain.refute_triple(subject, relation, obj)
    except PermissionError as e:
        return f"Ethics gate blocked the refute: {e}"
    if result is None:
        return (
            f"Could not refute: ({subject!r}, {relation!r}, {obj!r}). "
            "Check that all fields are non-empty."
        )
    return f"Refuted: {result.path_label} (path #{result.path_id}, marked as known-to-be-false)"


@mcp.tool()
def brain_did_you_mean(term: str) -> str:
    """Check for close matches to a term in Sara Brain.

    Use this when a query returns no results — it may be a misspelling.
    Returns candidate matches with descriptions for disambiguation.
    Critical for medical terms where a wrong match could be dangerous.
    """
    brain = _get_brain()
    candidates = brain.did_you_mean(term)
    if not candidates:
        n = brain.neuron_repo.resolve(term.strip().lower())
        if n:
            return f"'{term}' resolves to '{n.label}' (exact match)."
        return f"No matches found for '{term}'."

    lines = [f"Did you mean one of these? (searching for '{term}')"]
    for c in candidates:
        desc = f" — {c['description']}" if c['description'] else ""
        lines.append(f"  - {c['label']} ({c['type']}){desc}")
    return "\n".join(lines)


@mcp.tool()
def brain_ingest(source: str) -> str:
    """Ingest a document into Sara Brain from a file path or URL.

    Sara reads the document through the LLM cortex, extracts facts,
    learns them as neuron-chain paths, and reports what she understood.

    Works with local files (.txt, .md, .html) or URLs (http/https).

    Requires LLM to be configured (llm_provider in settings).
    """
    from .agent.bridge import AgentBridge

    brain = _get_brain()
    bridge = AgentBridge(brain)
    return bridge.ingest(source)


@mcp.tool()
def brain_scan_pollution() -> str:
    """Read-only scan for pollution caused by past parser bugs.

    Lists article-typo neurons, pronoun-subject neurons, and suspected
    content-word typos. Does NOT modify anything.
    """
    from .agent.bridge import AgentBridge
    return AgentBridge(_get_brain()).scan_pollution()


@mcp.tool()
def brain_list_article_candidates() -> str:
    """READ-ONLY list of paths attached to article-typo candidate neurons.

    The LLM MUST present these to the user and ask for per-item
    approval. What looks like a typo in English may be a valid word
    in Haitian Creole, Jamaican Patois, AAVE, or other dialects.
    Sara has no authority to silently erase a user's language.
    """
    from .agent.bridge import AgentBridge
    return AgentBridge(_get_brain()).list_article_candidates()


@mcp.tool()
def brain_list_pronoun_candidates() -> str:
    """READ-ONLY list of paths attached to pronoun-subject candidate neurons.

    The LLM MUST present these to the user and ask for per-item approval.
    """
    from .agent.bridge import AgentBridge
    return AgentBridge(_get_brain()).list_pronoun_candidates()


@mcp.tool()
def brain_list_suspected_typos() -> str:
    """List suspected content-word typos for USER REVIEW.

    The LLM MUST NOT decide to clean these — only the user can.
    Drug names that look alike are different drugs (metformin vs
    metoprolol). Always present the list and let the user choose.
    """
    from .agent.bridge import AgentBridge
    return AgentBridge(_get_brain()).list_suspected_typos()


@mcp.tool()
def brain_stats() -> str:
    """Get Sara Brain statistics — neuron count, segment count, path count."""
    brain = _get_brain()
    s = brain.stats()
    strongest = s.get("strongest_segment", "none")
    return (
        f"Neurons: {s['neurons']}\n"
        f"Segments: {s['segments']}\n"
        f"Paths: {s['paths']}\n"
        f"Strongest segment: {strongest}"
    )


@mcp.tool()
def brain_similar(label: str) -> str:
    """Find neurons that share downstream paths with the given neuron.

    Answers: "What is similar to X in Sara's knowledge?"
    """
    brain = _get_brain()
    links = brain.get_similar(label)
    if not links:
        return f"No similar neurons found for '{label}'."
    lines = [f"Similar to '{label}':"]
    for s in links:
        lines.append(
            f"  {s.neuron_a_label} <-> {s.neuron_b_label} "
            f"(overlap: {s.overlap_ratio:.0%})"
        )
    return "\n".join(lines)


# ── Resources ──


@mcp.resource("sara://brain/stats")
def get_brain_stats() -> str:
    """Current brain statistics."""
    return brain_stats()


@mcp.resource("sara://brain/neurons")
def get_neurons() -> str:
    """List all neurons in Sara Brain."""
    brain = _get_brain()
    neurons = brain.neuron_repo.list_all()
    lines = [f"Total neurons: {len(neurons)}\n"]
    for n in neurons[:100]:
        lines.append(f"  {n.label} ({n.neuron_type.value})")
    if len(neurons) > 100:
        lines.append(f"  ... and {len(neurons) - 100} more")
    return "\n".join(lines)


# ── Entry point ──


def main():
    """Run the Sara Brain MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
