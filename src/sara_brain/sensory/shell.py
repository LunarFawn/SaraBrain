"""The sensory shell — an empty processing engine over Sara Brain.

No knowledge. No weights. No training. The shell takes text input,
tokenizes it into wavefront seeds, runs parallel propagation through
Sara's graph, and renders the converging paths as output with full
provenance.

Sara Brain IS the weight store. Teaching a fact is adding a weight.
Every answer traces to specific taught facts. No black box.

Usage:
    from sara_brain.core.brain import Brain
    from sara_brain.sensory import SensoryShell

    brain = Brain("sara.db")
    shell = SensoryShell(brain)
    response = shell.process("what has one carbon atom")
    print(response.text)
    for src in response.sources:
        print(f"  path #{src.path_id}: {src.source_text}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.short_term import ShortTerm
from ..models.neuron import NeuronType
from ..models.result import RecognitionResult, PathTrace
from .tokenizer import Tokenizer, Token
from .renderer import Renderer, SourcedLine


@dataclass
class ShellResponse:
    """Result of processing one input through the shell.

    Every field is traceable: text came from paths, sources list
    exactly which paths contributed, gaps show what Sara doesn't know.
    """
    text: str
    sources: list[SourcedLine] = field(default_factory=list)
    confidence: int = 0
    gaps: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    recognition: list[RecognitionResult] = field(default_factory=list)


class SensoryShell:
    """Empty processing shell. All knowledge comes from Sara Brain.

    The shell has no weights, no training, no baked-in knowledge.
    It is a thin layer that:
    1. Tokenizes input into words/phrases
    2. Feeds tokens as wavefront seeds into Sara's graph
    3. Runs parallel wavefront propagation (echo, multi-threshold)
    4. Renders converging paths as output with provenance
    5. Reports gaps — concepts Sara doesn't know (curiosity triggers)
    """

    def __init__(self, brain) -> None:
        self.brain = brain
        self.tokenizer = Tokenizer(brain)
        self.renderer = Renderer(brain)

    def process(self, text: str) -> ShellResponse:
        """Process one input through Sara's graph.

        Returns a ShellResponse with traceable output, source
        provenance, confidence from convergence count, and gaps
        where Sara has no knowledge.
        """
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return ShellResponse(
                text="I don't know. I couldn't find any words to look up.",
                tokens=[],
            )

        seed_labels = [t.label for t in tokens]
        token_labels = list(seed_labels)

        # Track which tokens Sara knows about
        known_tokens: set[str] = set()
        for t in tokens:
            if t.neuron_id is not None:
                known_tokens.add(t.label)

        # Open a short-term scratchpad — read-only, no graph mutation
        with self.brain.short_term(event_type="shell-query") as st:
            # Run echo propagation at multiple thresholds
            # This is the wavefront engine — the core of the "empty transformer"
            for min_strength in [0.5, 0.3, 0.1]:
                self.brain.propagate_echo(
                    seed_labels,
                    st,
                    max_rounds=3,
                    min_strength=min_strength,
                )

            # Find convergence points — neurons reached by 2+ wavefronts
            intersections = st.intersections(min_sources=2)

            if intersections:
                # Build recognition results from convergence
                results = self._build_recognition(intersections, st)
                sourced = self.renderer.render_recognition(results)
                confidence = max(r.confidence for r in results) if results else 0
            else:
                # No convergence — try single-wavefront traces
                results = []
                sourced = self._try_direct_lookup(tokens)
                confidence = 0

        # Identify gaps
        gaps = [t.label for t in tokens if t.label not in known_tokens]
        gap_lines = self.renderer.render_gaps(token_labels, known_tokens)

        # Combine output
        all_lines = sourced + gap_lines
        text_output = self.renderer.format_output(all_lines, show_provenance=True)

        return ShellResponse(
            text=text_output,
            sources=sourced,
            confidence=confidence,
            gaps=gaps,
            tokens=token_labels,
            recognition=results,
        )

    def query(self, topic: str) -> ShellResponse:
        """Direct query: what does Sara know about this topic?

        Returns all paths leading to/from the topic, with provenance.
        """
        # Get paths TO and FROM this topic
        traces_to = self.brain.why(topic)
        traces_from = self.brain.trace(topic)
        all_traces = list(traces_to) + list(traces_from)

        sourced = self.renderer.render_query(topic, all_traces)
        text_output = self.renderer.format_output(sourced, show_provenance=True)

        return ShellResponse(
            text=text_output,
            sources=sourced,
            confidence=len(all_traces),
            tokens=[topic],
        )

    def _build_recognition(self, intersections: list[tuple[int, float, int]],
                           st: ShortTerm) -> list[RecognitionResult]:
        """Build RecognitionResults from short-term convergence data."""
        results: list[RecognitionResult] = []
        for neuron_id, weight, source_count in intersections:
            neuron = self.brain.neuron_repo.get_by_id(neuron_id)
            if neuron is None:
                continue

            # Only show CONCEPT neurons — skip intermediate RELATION,
            # PROPERTY, and ASSOCIATION nodes that wavefronts pass through
            if neuron.neuron_type != NeuronType.CONCEPT:
                continue

            # Get the paths that led to this neuron
            paths = self.brain.path_repo.get_paths_to(neuron_id)
            traces: list[PathTrace] = []
            for p in paths:
                steps = self.brain.path_repo.get_steps(p.id)
                neurons = []
                total_weight = 0.0
                for step in steps:
                    seg = self.brain.segment_repo.get_by_id(step.segment_id)
                    if seg is None:
                        continue
                    total_weight += seg.strength
                    src_neuron = self.brain.neuron_repo.get_by_id(seg.source_id)
                    if src_neuron and not neurons:
                        neurons.append(src_neuron)
                    tgt_neuron = self.brain.neuron_repo.get_by_id(seg.target_id)
                    if tgt_neuron:
                        neurons.append(tgt_neuron)

                traces.append(PathTrace(
                    neurons=neurons,
                    source_text=p.source_text,
                    weight=total_weight,
                ))

            results.append(RecognitionResult(
                neuron=neuron,
                converging_paths=traces,
            ))

        # Sort by confidence (number of converging paths)
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _try_direct_lookup(self, tokens: list[Token]) -> list[SourcedLine]:
        """When no convergence, try direct path lookup for each token."""
        all_sourced: list[SourcedLine] = []
        for token in tokens:
            if token.neuron_id is None:
                continue
            traces = self.brain.why(token.label)
            outgoing = self.brain.trace(token.label)
            for trace in list(traces) + list(outgoing):
                if trace.is_refuted:
                    continue
                line = self.renderer._render_trace(trace)
                if line:
                    all_sourced.append(line)
        return all_sourced
