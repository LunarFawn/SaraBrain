"""Wavefront renderer — translates convergence maps into readable facts.

The wavefront does the thinking (which neurons are relevant).
This module renders that thinking into human/LLM-readable form.

Three levels of rendering:
  - Level 1: Just the converged neuron labels + scores (current format)
  - Level 2: For each converged neuron, pull its paths (source triples)
  - Level 3: Full path chains with source text provenance

The cortex receives Level 2 or 3 — actual facts, not raw scores.
"""
from __future__ import annotations

from sara_brain.core.brain import Brain


def render_wavefront_facts(brain: Brain, seeds: list[str],
                           depth: int = 2,
                           max_facts: int = 30,
                           include_provenance: bool = False) -> str:
    """Run wavefront and render the result as readable facts.

    Returns a string of facts derived from the converged neurons,
    not raw convergence scores. Each fact is a readable triple path.
    """
    # Run wavefront
    brain.recognizer.max_depth = depth
    with brain.short_term(event_type="render") as st:
        brain.propagate_into(seeds, st, exact_only=True)
        convergence_map = dict(st.convergence_map)
        intersections = st.intersections(min_sources=2)

    # Collect converged neurons, sorted by score
    scored_neurons = []
    for item in intersections:
        nid, weight = item[0], item[1]
        n = brain.neuron_repo.get_by_id(nid)
        if n and not _is_noise(n.label):
            scored_neurons.append((n, weight))

    # If few intersections, also use top convergence map entries
    if len(scored_neurons) < 5:
        for nid, weight in sorted(convergence_map.items(), key=lambda x: -x[1]):
            if len(scored_neurons) >= max_facts:
                break
            n = brain.neuron_repo.get_by_id(nid)
            if n and not _is_noise(n.label) and n not in [s[0] for s in scored_neurons]:
                scored_neurons.append((n, weight))

    # For each converged neuron, pull source_text from its paths
    facts = []
    seen = set()
    for neuron, score in scored_neurons[:max_facts]:
        # Get source sentences from paths involving this neuron
        cur = brain.conn.execute(
            "SELECT source_text FROM paths WHERE origin_id = ? OR terminus_id = ?",
            (neuron.id, neuron.id),
        )
        for (src_text,) in cur:
            if src_text and src_text not in seen:
                seen.add(src_text)
                facts.append(src_text)
                if len(facts) >= max_facts:
                    break
        if len(facts) >= max_facts:
            break

    # Build output
    lines = [f"Wavefront from seeds {seeds}: {len(intersections)} intersections, {len(convergence_map)} neurons reached."]
    lines.append(f"Facts ({len(facts)}):")
    lines.append("")
    for f in facts:
        lines.append(f"  - {f}")

    return "\n".join(lines)


_NOISE_WORDS = frozenset({
    "it", "they", "this", "that", "the", "which", "these", "those",
    "we", "he", "she", "its", "their", "our", "his", "her",
    "therefore", "however", "also", "can", "may", "will",
    "some", "each", "many", "most", "all", "both", "other",
})


def _is_noise(label: str) -> bool:
    """Filter noise neurons (pronouns, stopwords, _attribute suffixes of them)."""
    base = label.removesuffix("_attribute")
    return base in _NOISE_WORDS
