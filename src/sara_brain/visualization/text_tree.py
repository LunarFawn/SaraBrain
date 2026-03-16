"""ASCII path visualization."""

from __future__ import annotations

from ..core.brain import Brain


def render_paths_from(brain: Brain, label: str) -> str:
    """Render an ASCII tree of all paths from a neuron."""
    neuron = brain.neuron_repo.get_by_label(label.strip().lower())
    if neuron is None:
        return f"Neuron '{label}' not found."

    lines: list[str] = [neuron.label]
    _render_children(brain, neuron.id, lines, prefix="", visited=set())
    return "\n".join(lines)


def _render_children(
    brain: Brain, neuron_id: int, lines: list[str], prefix: str, visited: set[int]
) -> None:
    visited.add(neuron_id)
    segments = brain.segment_repo.get_outgoing(neuron_id)

    for i, seg in enumerate(segments):
        if seg.target_id in visited:
            continue
        target = brain.neuron_repo.get_by_id(seg.target_id)
        if target is None:
            continue

        is_last = i == len(segments) - 1
        connector = "└── " if is_last else "├── "
        child_prefix = "    " if is_last else "│   "

        strength_info = f" [{seg.relation}, s={seg.strength:.1f}]"
        lines.append(f"{prefix}{connector}{target.label}{strength_info}")
        _render_children(brain, target.id, lines, prefix + child_prefix, visited)


def render_graph_dot(brain: Brain) -> str:
    """Export the brain graph as Graphviz DOT format."""
    lines = ["digraph sara_brain {", "  rankdir=LR;", "  node [shape=box];"]

    for neuron in brain.neuron_repo.list_all():
        style = {
            "concept": 'style=filled, fillcolor="#a8d8ea"',
            "property": 'style=filled, fillcolor="#ffcfdf"',
            "relation": 'style=filled, fillcolor="#fefdca"',
        }.get(neuron.neuron_type.value, "")
        lines.append(f'  n{neuron.id} [label="{neuron.label}", {style}];')

    for seg in brain.segment_repo.list_all():
        width = max(0.5, min(3.0, seg.strength))
        lines.append(
            f'  n{seg.source_id} -> n{seg.target_id} '
            f'[label="{seg.relation}", penwidth={width:.1f}];'
        )

    lines.append("}")
    return "\n".join(lines)
