"""Pretty-print paths, traces, and results."""

from __future__ import annotations

from ..models.result import RecognitionResult, PathTrace
from ..core.similarity import SimilarityLink


def format_learn_result(path_label: str, segments_created: int, neurons_created: int) -> str:
    parts = [f"  Created path: {path_label}"]
    details = []
    if segments_created:
        details.append(f"{segments_created} new segment{'s' if segments_created > 1 else ''}")
    if neurons_created:
        details.append(f"{neurons_created} new neuron{'s' if neurons_created > 1 else ''}")
    if details:
        parts[0] += f" ({', '.join(details)})"
    return "\n".join(parts)


def format_recognition(results: list[RecognitionResult]) -> str:
    if not results:
        return "  No recognition results."

    lines: list[str] = []
    for i, result in enumerate(results, 1):
        lines.append(f"  #{i} {result.neuron.label} ({result.confidence} converging path{'s' if result.confidence != 1 else ''})")
        for trace in result.converging_paths:
            lines.append(f"    {trace}")
    return "\n".join(lines)


def format_why(label: str, traces: list[PathTrace]) -> str:
    if not traces:
        return f"  No paths lead to \"{label}\"."

    lines = [f"  {len(traces)} path{'s' if len(traces) != 1 else ''} of thought lead to \"{label}\":"]
    for i, trace in enumerate(traces, 1):
        src = f" (from: \"{trace.source_text}\")" if trace.source_text else ""
        lines.append(f"    {i}. {trace}{src}")
    return "\n".join(lines)


def format_trace(label: str, traces: list[PathTrace]) -> str:
    if not traces:
        return f"  No paths from \"{label}\"."

    lines = [f"  Paths from \"{label}\":"]
    for trace in traces:
        lines.append(f"    {trace}")
    return "\n".join(lines)


def format_stats(stats: dict) -> str:
    lines = [
        f"  Neurons:  {stats['neurons']}",
        f"  Segments: {stats['segments']}",
        f"  Paths:    {stats['paths']}",
    ]
    if stats.get("strongest_segment"):
        lines.append(f"  Strongest: {stats['strongest_segment']}")
    return "\n".join(lines)


def format_neurons(neurons: list) -> str:
    if not neurons:
        return "  No neurons."
    lines = []
    for n in neurons:
        lines.append(f"  [{n.neuron_type.value:>8}] {n.label}")
    return "\n".join(lines)


def format_paths(paths: list) -> str:
    if not paths:
        return "  No paths."
    lines = []
    for p in paths:
        src = f" \"{p.source_text}\"" if p.source_text else ""
        lines.append(f"  Path #{p.id}: origin={p.origin_id} → terminus={p.terminus_id}{src}")
    return "\n".join(lines)


def format_similarities(links: list[SimilarityLink]) -> str:
    if not links:
        return "  No similarities found."
    lines = []
    for link in links:
        lines.append(
            f"  {link.neuron_a_label} ↔ {link.neuron_b_label} "
            f"(shared: {link.shared_paths}, overlap: {link.overlap_ratio:.1%})"
        )
    return "\n".join(lines)
