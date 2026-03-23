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


def format_define_result(name: str, question_word: str | None = None) -> str:
    if question_word:
        return f'  Created association: {name} (question word: "{question_word}")'
    return f"  Created association: {name}"


def format_describe_result(name: str, registered: list[str]) -> str:
    lines = [f"  Registered {len(registered)} propert{'y' if len(registered) == 1 else 'ies'} under \"{name}\":"]
    for prop in registered:
        lines.append(f"    {prop} → {name}")
    return "\n".join(lines)


def format_associations(assocs: dict[str, list[str]]) -> str:
    if not assocs:
        return "  No associations defined."
    lines = []
    for name, props in sorted(assocs.items()):
        lines.append(f"  {name}: {', '.join(props)}")
    return "\n".join(lines)


def format_query(question_word: str, subject: str, association: str, properties: list[str]) -> str:
    if not properties:
        return f'  No "{association}" known for "{subject}".'
    return f"  {subject} {association}: {', '.join(sorted(properties))}"


def format_questions(question_words: dict[str, list[str]]) -> str:
    if not question_words:
        return "  No question words defined."
    lines = ["  Available questions:"]
    for qword, assocs in sorted(question_words.items()):
        for assoc in sorted(assocs):
            lines.append(f"    {qword} <concept> {assoc}")
    return "\n".join(lines)


def format_categorize(label: str, category: str) -> str:
    return f'  Categorized "{label}" as "{category}".'


def format_categories(categories: dict[str, list[str]]) -> str:
    if not categories:
        return "  No categories defined."
    lines = []
    for cat, labels in sorted(categories.items()):
        lines.append(f"  {cat}: {', '.join(sorted(labels))}")
    return "\n".join(lines)


def format_perception_step(step) -> str:
    """Format one phase of the perception loop."""
    lines = [f"  [{step.phase}]"]
    if step.observations:
        lines.append(f"    Observed: {', '.join(step.observations)}")
    else:
        lines.append("    No new observations.")
    if step.taught_count:
        lines.append(f"    Taught {step.taught_count} fact{'s' if step.taught_count != 1 else ''}.")
    if step.recognition:
        top = step.recognition[0]
        lines.append(f"    Recognition: {top.neuron.label} ({top.confidence} converging path{'s' if top.confidence != 1 else ''})")
        for trace in top.converging_paths:
            lines.append(f"      {trace}")
    else:
        lines.append("    No recognition yet.")
    return "\n".join(lines)


def format_perception_result(result) -> str:
    """Format the final perception summary."""
    lines = [
        f"  Perception of {result.label}:",
        f"    Image: {result.image_path}",
        f"    Total observations: {len(result.all_observations)}",
        f"    Total facts taught: {result.total_taught}",
    ]
    if result.final_recognition:
        top = result.final_recognition[0]
        lines.append(f"    Final recognition: {top.neuron.label} ({top.confidence} converging path{'s' if top.confidence != 1 else ''})")
    else:
        lines.append("    Final recognition: none")
    return "\n".join(lines)


def format_correction(wrong_guess: str | None, correct_label: str, properties_taught: list[str]) -> str:
    """Format correction output."""
    lines = []
    if wrong_guess:
        lines.append(f"  Corrected: not {wrong_guess}, this is {correct_label}.")
    else:
        lines.append(f"  Corrected: this is {correct_label}.")
    if properties_taught:
        lines.append(f"  Taught {correct_label}: {', '.join(properties_taught)}")
    lines.append("  (Original observations retained — Sara never erases.)")
    return "\n".join(lines)


def format_see(image_label: str, property_label: str, taught: bool) -> str:
    """Format parent-points-out output."""
    if taught:
        return f"  Taught {image_label} is {property_label}."
    return f"  {image_label} already knows about {property_label}."
