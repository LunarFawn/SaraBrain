"""Agent bridge — Brain interface for the agent loop.

Two roles:
1. Query tools: LLM reads Sara's knowledge (read-only)
2. Observational learning: agent loop feeds outcomes to Sara

Follows the QBridge pattern (nlp/q_bridge.py) — every method returns str.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.brain import Brain


class AgentBridge:
    """Brain interface for the agent. Read + observe."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    # ── Query tools (LLM reads Sara's knowledge) ──

    def query(self, topic: str) -> str:
        """What does Sara know about a topic? Uses why + trace."""
        label = topic.strip().lower()
        traces = self.brain.why(label)
        forward = self.brain.trace(label)

        if not traces and not forward:
            return f"Sara doesn't know about '{topic}'."

        lines = []
        if traces:
            lines.append(f"Paths leading to '{label}':")
            for t in traces:
                src = f' (from: "{t.source_text}")' if t.source_text else ""
                lines.append(f"  {t}{src}")
        if forward:
            lines.append(f"Paths from '{label}':")
            for t in forward:
                src = f' (from: "{t.source_text}")' if t.source_text else ""
                lines.append(f"  {t}{src}")
        return "\n".join(lines)

    def recognize(self, inputs: str) -> str:
        """Given properties, what does Sara recognize?"""
        results = self.brain.recognize(inputs)
        if not results:
            return "Sara doesn't recognize anything from those inputs."
        lines = []
        for r in results:
            lines.append(
                f"  {r.neuron.label} ({r.confidence} converging paths)"
            )
            for trace in r.converging_paths:
                lines.append(f"    path: {trace}")
        return "\n".join(lines)

    def context(self, keywords: str) -> str:
        """Search brain for knowledge relevant to keywords."""
        words = [
            w.strip().lower()
            for w in keywords.split()
            if len(w.strip()) > 2
        ]
        all_traces = []
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

    def summarize(self, topic: str) -> str:
        """Aggregate everything Sara knows about a topic."""
        label = topic.strip().lower()
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
            return f"Sara knows nothing about '{topic}'."
        return "\n".join(lines)

    def stats(self) -> str:
        """Brain statistics."""
        s = self.brain.stats()
        return (
            f"Neurons: {s['neurons']}, Segments: {s['segments']}, "
            f"Paths: {s['paths']}"
        )

    def brain_summary(self, max_items: int = 20) -> str:
        """Compact summary of Sara's knowledge for system prompt injection."""
        neurons = self.brain.neuron_repo.list_all()
        if not neurons:
            return "Sara's brain is empty — no knowledge yet."

        concepts = [n for n in neurons if n.neuron_type.value == "concept"]
        properties = [n for n in neurons if n.neuron_type.value == "property"]

        lines = [f"Sara knows {len(neurons)} neurons ({len(concepts)} concepts, {len(properties)} properties)."]
        if concepts:
            labels = [n.label for n in concepts[:max_items]]
            lines.append(f"Concepts: {', '.join(labels)}")
            if len(concepts) > max_items:
                lines.append(f"  ... and {len(concepts) - max_items} more")
        return "\n".join(lines)

    # ── Observational learning (agent loop feeds outcomes) ──

    def observe(self, fact: str) -> str | None:
        """Sara observes what happened and learns.

        Called by the agent loop after actions execute — not by the LLM directly.
        Returns the path label if learned, None if unparseable.
        """
        result = self.brain.teach(fact)
        if result is not None:
            self.brain.conn.commit()
            return result.path_label
        return None

    def observe_many(self, facts: list[str]) -> int:
        """Observe multiple facts. Returns count of successfully learned."""
        count = 0
        for fact in facts:
            result = self.brain.teach(fact)
            if result is not None:
                count += 1
        if count > 0:
            self.brain.conn.commit()
        return count

    # ── Import/Export ──

    def import_brain(self, file_path: str) -> str:
        """Import a JSON brain export into Sara's database."""
        p = Path(file_path)
        if not p.is_file():
            return f"File not found: {file_path}"

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return f"Error reading {file_path}: {e}"

        conn = self.brain.conn
        conn.execute("PRAGMA foreign_keys=OFF")

        counts = {"neurons": 0, "segments": 0, "paths": 0}
        for n in data.get("neurons", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO neurons (id,label,neuron_type,created_at,metadata) VALUES (?,?,?,?,?)",
                    (n["id"], n["label"], n["neuron_type"], n.get("created_at", 0), None),
                )
                counts["neurons"] += 1
            except Exception:
                pass
        for s in data.get("segments", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO segments (id,source_id,target_id,relation,strength,traversals,created_at,last_used) VALUES (?,?,?,?,?,?,?,?)",
                    (s["id"], s["source_id"], s["target_id"], s["relation"],
                     s["strength"], s["traversals"], s.get("created_at", 0), s.get("last_used", 0)),
                )
                counts["segments"] += 1
            except Exception:
                pass
        for p_rec in data.get("paths", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO paths (id,origin_id,terminus_id,source_text,created_at) VALUES (?,?,?,?,?)",
                    (p_rec["id"], p_rec["origin_id"], p_rec["terminus_id"],
                     p_rec.get("source_text"), p_rec.get("created_at", 0)),
                )
                counts["paths"] += 1
            except Exception:
                pass
        for ps in data.get("path_steps", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO path_steps (id,path_id,step_order,segment_id) VALUES (?,?,?,?)",
                    (ps["id"], ps["path_id"], ps["step_order"], ps["segment_id"]),
                )
            except Exception:
                pass

        conn.execute("PRAGMA foreign_keys=ON")
        conn.commit()

        stats = self.brain.stats()
        return (
            f"Imported from {p.name}: "
            f"{counts['neurons']} neurons, {counts['segments']} segments, {counts['paths']} paths. "
            f"Brain now has: {stats['neurons']} neurons, {stats['segments']} segments, {stats['paths']} paths."
        )
