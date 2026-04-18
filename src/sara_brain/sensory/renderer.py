"""Renderer — converts Sara's converging paths back to natural language.

Every sentence in the output points to a specific path ID and source
text. If no paths converge, the renderer says "I don't know."
No invention. No hallucination. Only paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models.result import RecognitionResult, PathTrace


@dataclass
class SourcedLine:
    """One line of output with its provenance."""
    text: str
    path_id: int | None = None
    source_text: str | None = None
    weight: float = 0.0

    def __str__(self) -> str:
        provenance = ""
        if self.path_id is not None:
            provenance = f"  [path #{self.path_id}"
            if self.source_text:
                provenance += f", source: {self.source_text[:60]}"
            provenance += "]"
        return f"{self.text}{provenance}"


class Renderer:
    """Converts convergence results and path traces to readable output.

    The renderer has no knowledge. It reads paths and assembles them
    into sentences using the source_text that was stored when the
    fact was taught. Every output line is traceable.
    """

    def __init__(self, brain) -> None:
        self.brain = brain

    def render_recognition(self, results: list[RecognitionResult]) -> list[SourcedLine]:
        """Render recognition results as sourced lines.

        Each recognized concept becomes a header, followed by the
        paths that converged on it — showing the full trace through
        intermediate nodes so every step is visible.
        """
        if not results:
            return [SourcedLine(text="I don't know. No paths converged.")]

        lines: list[SourcedLine] = []
        for result in results:
            if result.is_refuted:
                continue

            lines.append(SourcedLine(
                text=f"{result.neuron.label} (confidence {result.confidence})",
            ))

            for trace in result.converging_paths:
                if trace.is_refuted:
                    continue
                line = self._render_trace(trace)
                if line:
                    lines.append(line)
                # Show the full path trace — the intermediate nodes
                # are HOW the wavefront got here. Without them there's
                # no traceability.
                if trace.neurons and len(trace.neurons) > 1:
                    path_str = " -> ".join(n.label for n in trace.neurons)
                    lines.append(SourcedLine(text=f"    trace: {path_str}"))

        if not lines:
            return [SourcedLine(text="I don't know. No paths converged.")]
        return lines

    def render_query(self, topic: str, traces: list[PathTrace]) -> list[SourcedLine]:
        """Render a 'what do you know about X' query.

        Returns all non-refuted paths leading to/from the topic,
        each with provenance.
        """
        if not traces:
            return [SourcedLine(
                text=f"I don't know anything about {topic!r}.",
            )]

        lines: list[SourcedLine] = []
        for trace in traces:
            if trace.is_refuted:
                continue
            line = self._render_trace(trace)
            if line:
                lines.append(line)

        if not lines:
            return [SourcedLine(
                text=f"Everything I knew about {topic!r} has been refuted.",
            )]
        return lines

    def render_gaps(self, tokens: list[str], known: set[str]) -> list[SourcedLine]:
        """Render what the shell doesn't know — curiosity output."""
        unknown = [t for t in tokens if t not in known]
        if not unknown:
            return []
        return [SourcedLine(
            text=f"I don't know: {', '.join(unknown)}",
        )]

    @staticmethod
    def format_output(lines: list[SourcedLine], show_provenance: bool = True) -> str:
        """Format sourced lines into a display string."""
        if not lines:
            return "I don't know."
        if show_provenance:
            return "\n".join(str(line) for line in lines)
        return "\n".join(line.text for line in lines)

    def _render_trace(self, trace: PathTrace) -> SourcedLine | None:
        """Convert a single PathTrace to a SourcedLine."""
        if not trace.neurons:
            return None

        # Use the original source text if available — it's already English
        if trace.source_text and not trace.source_text.startswith("[cleanup]"):
            text = trace.source_text.rstrip(".") + "."
        else:
            # Fall back to neuron labels
            text = " -> ".join(n.label for n in trace.neurons)

        # Find the path ID from the brain
        path_id = self._find_path_id(trace)

        return SourcedLine(
            text=text,
            path_id=path_id,
            source_text=trace.source_text,
            weight=trace.weight,
        )

    def _find_path_id(self, trace: PathTrace) -> int | None:
        """Look up the path ID for a trace by matching its terminus."""
        if not trace.neurons:
            return None
        terminus = trace.neurons[-1]
        paths = self.brain.path_repo.get_paths_to(terminus.id)
        for p in paths:
            if p.source_text == trace.source_text:
                return p.id
        # Fall back to first path to this terminus
        return paths[0].id if paths else None
