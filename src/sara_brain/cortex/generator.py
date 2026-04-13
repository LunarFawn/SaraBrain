"""Sara Cortex output generator (motor cortex).

Takes Sara's grounded paths and produces fluent English. Phase 1 is
template-based — pattern-match each path's relation type to a sentence
template. Phase 2 will replace this with a tiny grammar-only learned
model trained on the templates as data.

The generator has NO knowledge of its own. It only renders structures
into sentences. If Sara has no paths to render, the generator says so
honestly. There is no fallback to "let me think" — there is no thinking
in this layer, only rendering.
"""

from __future__ import annotations

from ..models.result import PathTrace
from . import grammar


class TemplateGenerator:
    """Render Sara's paths into English sentences using templates."""

    def render_path(self, trace: PathTrace) -> str | None:
        """Render a single PathTrace as one English sentence.

        Uses the path's source_text if available (it's already English),
        otherwise falls back to the relation template.

        Returns None if the path can't be rendered.
        """
        # If we have provenance text from the original teaching, that's
        # the most authentic English form — use it directly.
        if trace.source_text and not trace.source_text.startswith("[cleanup]"):
            return trace.source_text.rstrip(".") + "."

        # Otherwise reconstruct from neuron labels
        if len(trace.neurons) < 3:
            return None
        subject = trace.neurons[-1].label   # terminus = the concept
        obj = trace.neurons[0].label        # origin = the property
        # Walk the middle for relation hint
        relation = "is_a"  # safe default
        if len(trace.neurons) >= 3:
            mid = trace.neurons[1].label
            if "_" in mid:
                relation = mid.split("_")[-1]
                relation = f"has_{relation}" if relation != "attribute" else "is_a"

        template = grammar.TEMPLATES_BY_RELATION.get(
            relation, grammar.DEFAULT_TEMPLATE
        )
        return template.format(subject=subject, relation=relation.replace("_", " "), object=obj)

    def render_query(self, topic: str, traces: list[PathTrace]) -> str:
        """Render the answer to a query: what does Sara know about <topic>?

        Groups paths by source text. Uses the PathTrace.is_refuted property
        (which checks signed weight from segment strengths) to determine
        whether a path is believed or refuted — NOT source_text prefixes.
        """
        if not traces:
            return grammar.NO_KNOWLEDGE_WITH_HINT.format(topic=topic)

        # Group paths by source text. For each group, categorize by
        # signed weight (refuted = negative weight, believed = positive).
        groups: dict[str, dict] = {}
        for t in traces:
            canonical = self._canonical_text(t)
            if not canonical:
                continue
            g = groups.setdefault(canonical, {"taught": 0, "refuted": 0, "trace": t})
            if t.is_refuted:
                g["refuted"] += 1
            else:
                g["taught"] += 1

        if not groups:
            return grammar.NO_KNOWLEDGE.format(topic=topic)

        sentences = []
        for canonical, g in groups.items():
            sentence = canonical.rstrip(".") + "."
            if g["refuted"] > 0 and g["taught"] == 0:
                sentences.append(grammar.LIST_REFUTED_PREFIX.format(sentence=sentence))
            elif g["refuted"] > 0 and g["taught"] > 0:
                sentences.append(
                    f"  • [contested: taught {g['taught']}x, refuted {g['refuted']}x] {sentence}"
                )
            else:
                sentences.append(grammar.LIST_ITEM.format(sentence=sentence))

        intro = grammar.LIST_INTRO.format(topic=topic)
        return intro + "\n" + "\n".join(sentences)

    @staticmethod
    def _canonical_text(trace: PathTrace) -> str | None:
        """Return the canonical English form of a path's source text.

        No prefix stripping needed — source_text is never prefixed.
        Refutation state is in the graph, not the string.
        """
        if trace.source_text and not trace.source_text.startswith("[cleanup]"):
            return trace.source_text
        if len(trace.neurons) >= 2:
            return f"{trace.neurons[0].label} → {trace.neurons[-1].label}"
        return None

    def confirm_taught(self, fact_text: str) -> str:
        return grammar.CONFIRM_TAUGHT.format(fact=fact_text)

    def confirm_refuted(self, fact_text: str) -> str:
        return grammar.CONFIRM_REFUTED.format(fact=fact_text)

    def confirm_taught_multi(self, count: int) -> str:
        if count == 1:
            return "Learned 1 fact."
        return grammar.CONFIRM_TAUGHT_MULTI.format(count=count)

    def parse_failure(self, text: str) -> str:
        return grammar.CONFIRM_PARSE_FAIL.format(text=text)

    def no_knowledge(self, topic: str, with_hint: bool = True) -> str:
        if with_hint:
            return grammar.NO_KNOWLEDGE_WITH_HINT.format(topic=topic)
        return grammar.NO_KNOWLEDGE.format(topic=topic)
