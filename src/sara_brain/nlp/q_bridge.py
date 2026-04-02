"""Q Bridge — Amazon Q's direct interface to Sara Brain.

Q acts as cortex + parent: reads documents, extracts facts, teaches Sara,
queries her brain before coding, and corrects when needed.
No HTTP, no proxy — Q calls Sara's brain directly through Python.
"""

from __future__ import annotations

from pathlib import Path

from ..core.brain import Brain


class QBridge:
    """Q's interface to Sara Brain. Direct Python calls, no HTTP."""

    def __init__(self, db_path: str = "sara.db") -> None:
        self.brain = Brain(db_path)

    def teach(self, statement: str) -> str:
        """Teach Sara a fact. Returns description of what was learned."""
        result = self.brain.teach(statement)
        if result is None:
            return f"Could not parse: {statement}"
        return f"Learned: {result.path_label}"

    def teach_many(self, statements: list[str]) -> str:
        """Teach Sara multiple facts at once."""
        taught = 0
        for stmt in statements:
            r = self.brain.teach(stmt)
            if r is not None:
                taught += 1
        self.brain.conn.commit()
        return f"Taught {taught}/{len(statements)} facts."

    def query(self, label: str) -> str:
        """Ask Sara what she knows about something."""
        traces = self.brain.why(label.strip().lower())
        if not traces:
            return f"Sara doesn't know about '{label}'."
        lines = [f"Sara knows {len(traces)} path(s) to '{label}':"]
        for t in traces:
            src = f" (from: \"{t.source_text}\")" if t.source_text else ""
            lines.append(f"  {t}{src}")
        return "\n".join(lines)

    def recognize(self, inputs: str) -> str:
        """Give Sara properties, see what she recognizes."""
        results = self.brain.recognize(inputs)
        if not results:
            return "Sara doesn't recognize anything from those inputs."
        lines = []
        for r in results:
            lines.append(f"  {r.neuron.label} ({r.confidence} converging paths)")
        return "\n".join(lines)

    def check_rules(self, context: str) -> str:
        """Query Sara's brain for rules relevant to a context.

        Searches for neurons matching context keywords and returns
        all paths leading to them — Sara's knowledge about the topic.
        """
        keywords = [w.strip().lower() for w in context.split() if len(w.strip()) > 2]
        all_traces = []
        checked = set()

        for kw in keywords:
            neuron = self.brain.neuron_repo.get_by_label(kw)
            if neuron is None or neuron.id in checked:
                continue
            checked.add(neuron.id)
            traces = self.brain.why(kw)
            for t in traces:
                all_traces.append((kw, t))

        if not all_traces:
            return f"Sara has no rules about: {context}"

        lines = [f"Sara knows {len(all_traces)} relevant fact(s):"]
        for label, t in all_traces:
            src = f" (from: \"{t.source_text}\")" if t.source_text else ""
            lines.append(f"  [{label}] {t}{src}")
        return "\n".join(lines)

    def ingest_file(self, filepath: str) -> str:
        """Read a file and teach Sara everything in it via the digester."""
        p = Path(filepath)
        if not p.is_file():
            return f"File not found: {filepath}"

        text = p.read_text(encoding="utf-8")
        return self.ingest_text(text, source=filepath)

    def ingest_text(self, text: str, source: str = "text") -> str:
        """Ingest text through the digester — Q acts as cortex."""
        try:
            result = self.brain.ingest(text, source=source)
        except ValueError as e:
            return f"Error: {e}"

        lines = [
            f"Ingested: {source}",
            f"  Statements extracted: {len(result.all_statements)}",
            f"  Facts taught: {result.total_taught}",
        ]
        if result.unknown_concepts:
            lines.append(f"  Unknown concepts explored: {', '.join(result.unknown_concepts)}")
        if result.summary:
            lines.append(f"  Summary: {result.summary}")
        return "\n".join(lines)

    def summarize(self, topic: str) -> str:
        """Summarize everything Sara knows about a topic."""
        traces = self.brain.why(topic.strip().lower())
        similar = self.brain.get_similar(topic.strip().lower())

        lines = []
        if traces:
            lines.append(f"Paths to '{topic}':")
            for t in traces:
                lines.append(f"  {t}")
        if similar:
            lines.append(f"Similar to '{topic}':")
            for s in similar:
                lines.append(f"  {s.neuron_a_label} ↔ {s.neuron_b_label} (overlap: {s.overlap_ratio:.0%})")
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

    def close(self) -> None:
        self.brain.close()

    def __enter__(self) -> QBridge:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
