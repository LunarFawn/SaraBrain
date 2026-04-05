"""MCP tool definitions and handlers for Sara Brain.

Exposes brain capabilities as MCP tools. No LLM-specific code —
any MCP client (Claude, Amazon Q, etc.) can use these.
"""

from __future__ import annotations

import re

from ..core.brain import Brain


# ── Tool Definitions (MCP format) ──

TOOLS = [
    {
        "name": "brain_teach",
        "description": (
            "Teach Sara Brain a fact. Sara learns through path-of-thought: "
            "each fact creates neurons and segments forming a knowledge path. "
            "Format: '<subject> is <property>' or '<subject> contains <object>'. "
            "Example: 'apples are red', 'project uses pytest'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "A teaching statement (e.g., 'apples are red')",
                }
            },
            "required": ["statement"],
        },
    },
    {
        "name": "brain_query",
        "description": (
            "Query Sara Brain for everything she knows about a topic. "
            "Returns all paths leading to and from the concept."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to query (e.g., 'python', 'flask')",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "brain_recognize",
        "description": (
            "Give Sara properties and see what she recognizes. Uses parallel "
            "wavefront propagation to find concepts where paths converge. "
            "Input is comma-separated properties."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "inputs": {
                    "type": "string",
                    "description": "Comma-separated properties (e.g., 'red, round, sweet')",
                }
            },
            "required": ["inputs"],
        },
    },
    {
        "name": "brain_context",
        "description": (
            "Search Sara Brain for knowledge relevant to keywords. "
            "Use this BEFORE taking action to check if Sara has guidance. "
            "Sara's knowledge comes from user teachings and is trusted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Space-separated keywords to search for",
                }
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "brain_validate",
        "description": (
            "Validate a proposed action against Sara Brain's knowledge. "
            "Call this before executing significant actions. Sara checks "
            "the proposal against her known paths and returns approval "
            "or a correction with reasoning. If Sara corrects you, adjust."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal": {
                    "type": "string",
                    "description": "Description of what you plan to do",
                }
            },
            "required": ["proposal"],
        },
    },
    {
        "name": "brain_observe",
        "description": (
            "Report an outcome to Sara Brain so she can learn from it. "
            "After you complete an action, tell Sara what happened. "
            "She records observations as knowledge paths. "
            "Format as a simple fact: '<subject> is <property>'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "observation": {
                    "type": "string",
                    "description": "What happened (e.g., 'test_auth.py passed all tests')",
                }
            },
            "required": ["observation"],
        },
    },
    {
        "name": "brain_summarize",
        "description": (
            "Get a complete summary of everything Sara knows about a topic, "
            "including similar concepts and their overlap."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to summarize",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "brain_stats",
        "description": "Get Sara Brain statistics: neuron count, segment count, path count.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Handlers ──


class ToolHandler:
    """Handles MCP tool calls by delegating to the Brain."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def handle(self, name: str, arguments: dict) -> str:
        """Dispatch a tool call and return the result string."""
        handler = getattr(self, f"_handle_{name}", None)
        if handler is None:
            return f"Unknown tool: {name}"
        return handler(arguments)

    def _handle_brain_teach(self, args: dict) -> str:
        statement = args.get("statement", "").strip()
        if not statement:
            return "No statement provided."
        result = self.brain.teach(statement)
        if result is None:
            return f"Could not parse: {statement}"
        self.brain.conn.commit()
        return f"Learned: {result.path_label}"

    def _handle_brain_query(self, args: dict) -> str:
        label = args.get("topic", "").strip().lower()
        if not label:
            return "No topic provided."

        traces_to = self.brain.why(label)
        traces_from = self.brain.trace(label)

        if not traces_to and not traces_from:
            return f"Sara doesn't know about '{label}'."

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

    def _handle_brain_recognize(self, args: dict) -> str:
        inputs = args.get("inputs", "").strip()
        if not inputs:
            return "No inputs provided."
        results = self.brain.recognize(inputs)
        if not results:
            return "Sara doesn't recognize anything from those inputs."
        lines = []
        for r in results:
            lines.append(f"  {r.neuron.label} ({r.confidence} converging paths)")
            for trace in r.converging_paths:
                lines.append(f"    path: {trace}")
        return "\n".join(lines)

    def _handle_brain_context(self, args: dict) -> str:
        keywords = args.get("keywords", "").strip()
        if not keywords:
            return "No keywords provided."

        words = [w.strip().lower() for w in keywords.split() if len(w.strip()) > 2]
        all_traces: list[tuple[str, object]] = []
        checked: set[int] = set()

        for kw in words:
            neuron = self.brain.neuron_repo.get_by_label(kw)
            if neuron is None or neuron.id in checked:
                continue
            checked.add(neuron.id)
            for t in self.brain.why(kw):
                all_traces.append((kw, t))
            for t in self.brain.trace(kw):
                all_traces.append((kw, t))

        if not all_traces:
            return f"Sara has no knowledge about: {keywords}"

        lines = [f"Sara knows {len(all_traces)} relevant fact(s):"]
        for label, t in all_traces:
            src = f' (from: "{t.source_text}")' if t.source_text else ""
            lines.append(f"  [{label}] {t}{src}")
        return "\n".join(lines)

    def _handle_brain_validate(self, args: dict) -> str:
        proposal = args.get("proposal", "").strip()
        if not proposal:
            return "No proposal provided."

        # Extract concepts from proposal
        concepts = _extract_concepts(proposal)
        if not concepts:
            return "APPROVED: No relevant concepts found to check."

        # Query Sara for relevant knowledge
        knowledge_lines = []
        checked: set[int] = set()
        for concept in concepts:
            neuron = self.brain.neuron_repo.get_by_label(concept)
            if neuron is None or neuron.id in checked:
                continue
            checked.add(neuron.id)
            for t in self.brain.why(concept):
                src = f' (from: "{t.source_text}")' if t.source_text else ""
                knowledge_lines.append(f"  [{concept}] {t}{src}")

        if not knowledge_lines:
            return "APPROVED: Sara has no relevant knowledge to check against."

        # Check for conflicts
        knowledge = "\n".join(knowledge_lines)
        conflict = _detect_conflict(proposal.lower(), knowledge_lines)

        if conflict:
            return (
                f"CORRECTION: {conflict}\n\n"
                f"Sara's relevant knowledge:\n{knowledge}\n\n"
                f"Please adjust your approach to align with Sara's knowledge."
            )

        return (
            f"APPROVED with context.\n"
            f"Sara's relevant knowledge:\n{knowledge}"
        )

    def _handle_brain_observe(self, args: dict) -> str:
        observation = args.get("observation", "").strip()
        if not observation:
            return "No observation provided."
        result = self.brain.teach(observation)
        if result is not None:
            self.brain.conn.commit()
            return f"Sara observed: {result.path_label}"
        return f"Sara noted but could not parse as a path: {observation}"

    def _handle_brain_summarize(self, args: dict) -> str:
        label = args.get("topic", "").strip().lower()
        if not label:
            return "No topic provided."

        traces = self.brain.why(label)
        similar = self.brain.get_similar(label)

        lines = []
        if traces:
            lines.append(f"Paths to '{label}':")
            for t in traces:
                lines.append(f"  {t}")
        if similar:
            lines.append(f"Similar to '{label}':")
            for s in similar:
                lines.append(
                    f"  {s.neuron_a_label} <-> {s.neuron_b_label}"
                    f" (overlap: {s.overlap_ratio:.0%})"
                )
        if not lines:
            return f"Sara knows nothing about '{label}'."
        return "\n".join(lines)

    def _handle_brain_stats(self, _args: dict) -> str:
        s = self.brain.stats()
        return (
            f"Neurons: {s['neurons']}, Segments: {s['segments']}, "
            f"Paths: {s['paths']}"
        )


# ── Helpers ──

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "but", "or", "not", "so", "if", "when", "while", "where", "how",
    "what", "which", "who", "that", "this", "these", "those", "then",
    "there", "it", "its", "they", "them", "we", "us", "you", "your",
    "he", "she", "him", "her", "i", "me", "my", "up", "out", "about",
    "file", "code", "run", "use", "using", "write", "read", "create",
    "make", "want", "need", "please", "let", "get", "set", "put", "try",
}


def _extract_concepts(text: str) -> list[str]:
    """Pull keywords from text for brain lookup."""
    words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
    seen: set[str] = set()
    concepts = []
    for w in words:
        if w not in _STOP_WORDS and len(w) > 2 and w not in seen:
            seen.add(w)
            concepts.append(w)
    return concepts


def _detect_conflict(proposal: str, knowledge_lines: list[str]) -> str | None:
    """Check if a proposal contradicts Sara's knowledge."""
    for line in knowledge_lines:
        line_lower = line.strip().lower()
        # Extract the fact from path format
        parts = [p.strip() for p in line_lower.split("→")]
        if len(parts) < 2:
            continue
        prop = parts[0].strip().lstrip("[").split("]")[-1].strip()
        concept = parts[-1].strip()

        # Check for explicit contradictions
        if concept in proposal:
            if f"not {prop}" in proposal:
                return f"Sara knows '{concept}' is '{prop}', but proposal says 'not {prop}'"
            if f"don't use {prop}" in proposal or f"dont use {prop}" in proposal:
                return f"Sara knows '{concept}' is '{prop}', but proposal says 'don't use {prop}'"
            if f"instead of {prop}" in proposal:
                return f"Sara knows '{concept}' is '{prop}', but proposal says 'instead of {prop}'"

    return None
