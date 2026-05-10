"""v045 — multi-hop reasoning over substrate.

Heuristic planner + bounded-BFS orchestrator. Chains substrate
retrievals when the question shape implies multi-hop is needed.

The orchestrator is purely structural: bounded BFS over substrate
edges with a topic anchor, cycle detection via a visited set, edge
cap, depth cap. No ML, no learned policies. Every claim in the
final output still traces to a substrate edge — the chain logic is
deterministic graph walking, not invented reasoning.

See docs/v045_multihop_reasoning_plan.md for the architecture.
"""
from __future__ import annotations

import re
from typing import Any

from sara_brain.core.brain import Brain
from sara_reader.tools import execute_tool


# ── Planner ───────────────────────────────────────────────────────

# Question-shape patterns that suggest the answer requires chaining
# multiple substrate retrievals. v0 heuristic; could become a learned
# classifier later. Mis-fires both ways are expected — we keep the
# patterns conservative-but-useful.
_MULTIHOP_PATTERNS: tuple[str, ...] = (
    r"\bwhy\b",
    r"\bhow\s+(does|do|can|could|might|is|are)\b",
    r"\bbecause\b",
    r"\bdue\s+to\b",
    r"\bcaused?\s+by\b",
    r"\bcauses?\b",
    r"\bresults?\s+in\b",
    r"\bleads?\s+to\b",
    r"\bconnects?\s+to\b",
    r"\bwhat\s+does\s+\w+\s+do\b",
    r"\brelat(e|es|ed)\s+to\b",
    r"\bdepends?\s+on\b",
)

_MULTIHOP_RE = re.compile("|".join(_MULTIHOP_PATTERNS), re.IGNORECASE)


def should_multihop(question: str) -> bool:
    """Heuristic: does the question shape suggest multi-hop?

    Triggers on `why`, `how does X`, `because`, `caused by`, etc.
    Plain `what is X` / `tell me about Y` stays single-hop.

    .. deprecated::
        Superseded by :func:`sara_brain.cortex.question_intent.classify`.
        chat.py now dispatches on
        ``intent == QuestionIntent.MECHANISM``. Function retained for
        external callers and ``__all__`` compatibility; new code should
        use the classifier directly.
    """
    if not question:
        return False
    return bool(_MULTIHOP_RE.search(question))


# ── Orchestrator ──────────────────────────────────────────────────

# Stop words / generic particles we never follow as candidates
# (substrate ingestion sometimes emits is_part_of edges from
# constituent tokens of multi-word labels — `in`, `of`, `the` etc.
# would otherwise spawn useless hops).
_NEVER_FOLLOW: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "with",
    "to", "from", "as", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "has", "have", "had", "do", "does",
    "did", "this", "that", "these", "those", "it", "its", "their",
})

# Edge-line regex: 'X' --[rel]--> 'Y'  (apostrophes match) or
# "X" --[rel]--> "Y" (substrate uses double quotes when the label
# contains an apostrophe).
_EDGE_RE = re.compile(
    r"""(?P<sq>['"])(?P<src>.+?)(?P=sq)"""
    r"""\s*--\[(?P<rel>[^\]]+)\]-->\s*"""
    r"""(?P<tq>['"])(?P<tgt>.+?)(?P=tq)"""
)


def _candidates_from_edges(edges_text: str, anchor: str) -> list[str]:
    """Parse a `brain_explore` result string and return concepts on
    the OTHER side of each edge from the anchor — those are the
    candidates worth recursing into. Stop-word concepts are skipped."""
    candidates: list[str] = []
    seen: set[str] = set()
    anchor_l = anchor.lower().strip()
    for m in _EDGE_RE.finditer(edges_text):
        src = m["src"]
        tgt_raw = m["tgt"]
        tgt = tgt_raw.replace("_attribute", "")
        if src.lower().strip() == anchor_l:
            other = tgt
        elif tgt.lower().strip() == anchor_l:
            other = src
        else:
            # Neither end matches the anchor — both are weakly
            # related; pick the target side (substrate convention).
            other = tgt
        other_n = other.lower().strip()
        if not other_n or other_n in seen or other_n in _NEVER_FOLLOW:
            continue
        seen.add(other_n)
        candidates.append(other.strip())
    return candidates


def _count_edges_in_text(s: str) -> int:
    return len(_EDGE_RE.findall(s))


def _count_edges_in_gathered(gathered: list[dict]) -> int:
    return sum(_count_edges_in_text(g.get("result", "")) for g in gathered)


def _anchor_from_args(args: dict) -> str | None:
    """Extract the topic concept from a router decision's args."""
    for field in ("concept", "label", "term"):
        if field in args and args[field]:
            return str(args[field])
    return None


def plan_chain(
    brain: Brain,
    initial_decision_dict: dict[str, Any],
    max_depth: int = 1,
    max_extra_edges: int = 15,
) -> list[dict]:
    """Bounded BFS over substrate edges starting from the initial
    router decision.

    `initial_decision_dict` is the same shape v044 single-hop
    consumed: `{"tool": ..., **args}`. The args's anchor field
    (`concept` / `label` / `term`) is the seed concept; the
    orchestrator walks outward from there.

    Returns `gathered` in the existing shape — list of
    `{call: {tool, args}, result: <edges_text>}` — so the downstream
    `_synthesize()` consumes it identically.

    Caps (tightened in v045 follow-up after first chat-REPL test):
      - `max_depth`: how many BFS levels past the seed (default 1).
        Originally 2; lowered because depth=2 produced wall-of-text
        output for typical questions. depth=1 still covers the
        common "X → Y → Z" two-fact chains because the seed already
        contains some adjacent edges.
      - `max_extra_edges`: edges accumulated from EXPANSION hops only
        (the seed hop is always returned whole). Default 15.
        Originally 50; lowered for the same reason — chat output was
        unreadable at 50.
    Cycle detection via a visited set of lowercase-stripped concept
    labels.
    """
    tool = initial_decision_dict["tool"]
    args = {k: v for k, v in initial_decision_dict.items() if k != "tool"}

    anchor = _anchor_from_args(args)

    # Step 1: execute the seed query.
    result = execute_tool(brain, tool, args)
    gathered: list[dict] = [{"call": {"tool": tool, "args": args}, "result": result}]

    if anchor is None:
        # No concept to chain from — degrades to single-hop output.
        return gathered

    visited: set[str] = {anchor.lower().strip()}
    frontier = _candidates_from_edges(result, anchor)
    extra_edges_so_far = 0  # only count edges from expansion hops

    for _depth in range(1, max_depth + 1):
        if not frontier or extra_edges_so_far >= max_extra_edges:
            break
        next_frontier: list[str] = []
        for concept in frontier:
            c_norm = concept.lower().strip()
            if c_norm in visited:
                continue
            visited.add(c_norm)
            if extra_edges_so_far >= max_extra_edges:
                break
            sub_args = {"label": concept, "depth": 1}
            try:
                sub_result = execute_tool(brain, "brain_explore", sub_args)
            except Exception:
                continue
            gathered.append({
                "call": {"tool": "brain_explore", "args": sub_args},
                "result": sub_result,
            })
            extra_edges_so_far += _count_edges_in_text(sub_result)
            next_frontier.extend(_candidates_from_edges(sub_result, concept))
        frontier = next_frontier

    return gathered


__all__ = ["should_multihop", "plan_chain"]
