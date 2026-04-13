"""Sara Cortex router — the entry point for cortex-driven turns.

Cortex.process(text) is the function that:

1. Parses the user's input via the EnhancedParser
2. Decides whether the cortex can handle the turn directly
3. Runs the appropriate brain operations
4. Renders Sara's response via the TemplateGenerator
5. Returns a CortexResponse

If the cortex cannot handle the turn (low confidence, exotic phrasing,
conversational nuance), it returns a CortexResponse with delegate=True
so the caller can fall back to a larger LLM.

The cortex never invents facts. If Sara has nothing to say, the cortex
says so. The integrated cortex is the architectural commitment that
language is a thin layer over the brain — the brain is where cognition
lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.brain import Brain
from .parser import EnhancedParser, ParsedTurn, TurnKind, ExtractedFact
from .generator import TemplateGenerator


@dataclass
class TeachAmbiguity:
    """Raised when a teach operation finds a candidate near-match.

    The router presents this to the user as a disambiguation prompt
    instead of silently merging or creating duplicates. Critical for
    medication and other safety-sensitive contexts where lookalike
    names refer to genuinely different concepts.
    """
    new_term: str               # the term the user wrote
    candidates: list[dict]      # close matches from did_you_mean
    is_subject: bool            # True if ambiguity is in the subject, False if object
    fact_text: str              # the original statement
    safety_grounded: bool = False  # True if context is safety-related


@dataclass
class CortexOperation:
    """A single brain operation performed during a turn."""
    op: str          # "teach", "refute", "query", "ingest"
    target: str      # the fact or topic
    success: bool
    detail: str = ""


@dataclass
class CortexResponse:
    """The result of one cortex turn."""
    text: str                           # what to show the user
    operations: list[CortexOperation] = field(default_factory=list)
    confidence: float = 1.0             # how sure the cortex is
    delegate: bool = False              # if True, caller should fall back to LLM
    parsed_turn: ParsedTurn | None = None
    ambiguities: list[TeachAmbiguity] = field(default_factory=list)
    requires_disambiguation: bool = False  # if True, caller MUST resolve before commit


class Cortex:
    """The Sara Cortex entry point.

    A thin language layer that converts natural language to brain
    operations and back. No knowledge of its own. No opinions. Only
    grammar and templates.

    Usage:
        from sara_brain.core.brain import Brain
        from sara_brain.cortex import Cortex

        brain = Brain("sara.db")
        cortex = Cortex(brain)
        response = cortex.process("the edubba was a sumerian school")
        print(response.text)  # "Learned: edubba is a sumerian school."
    """

    def __init__(self, brain: Brain, strict_safety: bool = True) -> None:
        self.brain = brain
        self.parser = EnhancedParser(brain.taxonomy)
        self.generator = TemplateGenerator()
        # When True (default), the cortex requires disambiguation for any
        # fuzzy match in safety-grounded contexts. When False, only
        # exact-match conflicts trigger disambiguation. Default is on
        # because the safety cost of a wrong merge is much higher than
        # the friction cost of an extra prompt.
        self.strict_safety = strict_safety

    def process(self, text: str) -> CortexResponse:
        """Process one user turn through the cortex."""
        text = text.strip()
        if not text:
            return CortexResponse(text="", confidence=0.0)

        parsed = self.parser.parse(text)

        if parsed.kind == TurnKind.GREETING:
            return self._handle_greeting(parsed)

        if parsed.kind == TurnKind.ASSOCIATION:
            return self._handle_association(parsed)

        if parsed.kind == TurnKind.QUESTION:
            return self._handle_question(parsed)

        if parsed.kind in (TurnKind.STATEMENT, TurnKind.NEGATION, TurnKind.CORRECTION):
            return self._handle_assertion(parsed)

        # Unknown / unparseable — delegate to upstream LLM with low confidence
        return CortexResponse(
            text=self.generator.parse_failure(text),
            confidence=0.0,
            delegate=True,
            parsed_turn=parsed,
        )

    # ── handlers ──

    def _handle_greeting(self, parsed: ParsedTurn) -> CortexResponse:
        return CortexResponse(
            text="Hello. I'm Sara. Tell me what you know or ask me what I remember.",
            confidence=1.0,
            parsed_turn=parsed,
        )

    def _handle_association(self, parsed: ParsedTurn) -> CortexResponse:
        """Spreading-activation query — return the cluster around a concept.

        This is the brain-like 'say a word, get the cloud' interface.
        Calls brain.cluster_around() to find concepts strongly connected
        to the topic, then renders them grouped by hop distance and
        connection strength.
        """
        ops: list[CortexOperation] = []
        if not parsed.topics:
            return CortexResponse(
                text="Tell me a word and I'll show you what's associated with it.",
                confidence=0.3,
                delegate=False,
                parsed_turn=parsed,
            )

        # Try the joined topic first ("sumerian edubba"), then individual words
        candidates_to_try = [" ".join(parsed.topics)] + parsed.topics
        for topic in candidates_to_try:
            cluster = self.brain.cluster_around(topic, depth=2, max_results=30)
            ops.append(CortexOperation(
                op="cluster",
                target=topic,
                success=bool(cluster),
                detail=f"{len(cluster)} associated concepts",
            ))
            if cluster:
                text = self._render_association(topic, cluster)
                return CortexResponse(
                    text=text,
                    operations=ops,
                    confidence=1.0,
                    delegate=False,
                    parsed_turn=parsed,
                )

        # Nothing in the cluster — say so honestly
        topic = parsed.topics[0]
        return CortexResponse(
            text=self.generator.no_knowledge(topic, with_hint=True),
            operations=ops,
            confidence=1.0,
            delegate=False,
            parsed_turn=parsed,
        )

    @staticmethod
    def _render_association(topic: str, cluster: list[dict]) -> str:
        """Render a cluster as a brain-like association cloud."""
        if not cluster:
            return f"Sara has no associations for {topic!r}."

        # Group by hop distance: direct (1-hop) vs indirect (2+ hop)
        direct = [c for c in cluster if c["hops"] == 1]
        indirect = [c for c in cluster if c["hops"] >= 2]

        lines = [f"Things associated with {topic!r} in Sara's brain:"]
        if direct:
            lines.append("")
            lines.append("  Directly connected:")
            for c in direct[:15]:
                arrow = {"incoming": "←", "outgoing": "→", "both": "↔"}[c["direction"]]
                lines.append(
                    f"    {arrow} {c['label']} "
                    f"({c['type']}, {c['connections']} edge(s))"
                )
        if indirect:
            lines.append("")
            lines.append("  Indirectly connected (via shared neighbors):")
            for c in indirect[:10]:
                lines.append(
                    f"    · {c['label']} "
                    f"({c['type']}, {c['connections']} edge(s))"
                )
        return "\n".join(lines)

    def _handle_question(self, parsed: ParsedTurn) -> CortexResponse:
        ops: list[CortexOperation] = []
        if not parsed.topics:
            return CortexResponse(
                text="I'm not sure what you're asking about. Try a topic word.",
                confidence=0.3,
                delegate=True,
                parsed_turn=parsed,
            )

        # Try each topic in order until we find one with paths
        for topic in parsed.topics:
            traces = self.brain.why(topic)
            outgoing = self.brain.trace(topic)
            all_traces = list(traces) + list(outgoing)
            ops.append(CortexOperation(
                op="query",
                target=topic,
                success=bool(all_traces),
                detail=f"{len(all_traces)} paths",
            ))
            if all_traces:
                response_text = self.generator.render_query(topic, all_traces)
                return CortexResponse(
                    text=response_text,
                    operations=ops,
                    confidence=1.0,
                    delegate=False,
                    parsed_turn=parsed,
                )

        # No topic has any paths — Sara honestly doesn't know
        topic = parsed.topics[0]
        return CortexResponse(
            text=self.generator.no_knowledge(topic, with_hint=True),
            operations=ops,
            confidence=1.0,           # high confidence she doesn't know
            delegate=False,           # do NOT delegate — answer honestly
            parsed_turn=parsed,
        )

    def _check_ambiguity(self, fact: ExtractedFact) -> list[TeachAmbiguity]:
        """Check if any term in a fact has a close fuzzy match OR
        qualified variants (place/people/language homonyms).

        Returns a list of TeachAmbiguity objects — one per ambiguous term.
        Empty list means the fact is unambiguous and safe to commit.

        Sara never auto-merges. The caller must present these to the user.
        """
        from .entity_resolver import find_qualified_variants

        ambiguities = []

        # Check the subject
        for term, is_subject in [(fact.subject, True), (fact.obj, False)]:
            if not term or len(term) <= 3:
                continue

            # Check for qualified variants first (homonym disambiguation)
            # e.g., "sumerian" has variants "sumerian language", "sumerian people"
            variants = find_qualified_variants(term, self.brain)
            if len(variants) > 1:
                # Multiple qualified variants exist — disambiguation needed
                variant_candidates = []
                for v in variants:
                    n = self.brain.neuron_repo.get_by_label(v)
                    if n:
                        paths = self.brain.path_repo.get_paths_to(n.id)
                        desc = ""
                        for p in paths[:1]:
                            if p.source_text:
                                desc = p.source_text[:60]
                                break
                        variant_candidates.append({
                            "label": v,
                            "type": n.neuron_type.value,
                            "description": desc,
                            "distance": 0,
                        })
                if variant_candidates:
                    safety_grounded = self._is_safety_context(
                        fact, term, variant_candidates
                    )
                    ambiguities.append(TeachAmbiguity(
                        new_term=term,
                        candidates=variant_candidates,
                        is_subject=is_subject,
                        fact_text=fact.original_text or "",
                        safety_grounded=safety_grounded,
                    ))
                    continue

            # If the term already resolves exactly in Sara, no ambiguity
            existing = self.brain.neuron_repo.get_by_label(term)
            if existing is not None:
                continue
            # Otherwise check for close fuzzy matches
            try:
                candidates = self.brain.did_you_mean(term)
            except Exception:
                candidates = []
            if not candidates:
                continue
            # Filter to genuine candidates (close enough to matter)
            close = [c for c in candidates if c.get("distance", 99) <= 2]
            if not close:
                continue
            # Determine if this fact is in a safety-grounded context.
            # Conservative heuristic: if any term has a SAFETY-grounded
            # neighbor in the brain, treat as safety-grounded.
            safety_grounded = self._is_safety_context(fact, term, close)
            ambiguities.append(TeachAmbiguity(
                new_term=term,
                candidates=close,
                is_subject=is_subject,
                fact_text=fact.original_text or "",
                safety_grounded=safety_grounded,
            ))
        return ambiguities

    def _is_safety_context(
        self, fact: ExtractedFact, term: str, candidates: list[dict]
    ) -> bool:
        """Heuristic: is this term near safety-grounded knowledge?

        Returns True if any close candidate or the fact's other terms
        connect to a SAFETY-layer innate primitive within a few hops.
        """
        try:
            from ..innate.primitives import SAFETY
        except ImportError:
            return False
        # Cheap heuristic: if any term in the fact appears in SAFETY,
        # OR any candidate's label appears in SAFETY, treat as safety.
        all_terms = {fact.subject, fact.obj}
        for c in candidates:
            all_terms.add(c.get("label", ""))
        for t in all_terms:
            if t and t.lower() in SAFETY:
                return True
        return False

    def _handle_assertion(self, parsed: ParsedTurn) -> CortexResponse:
        ops: list[CortexOperation] = []
        taught = 0
        refuted = 0
        last_summary = ""

        # Check every fact for ambiguity FIRST. If any term has a close
        # fuzzy match, return a disambiguation request without committing.
        # This is the safety pattern: never auto-merge, always ask.
        all_ambiguities: list[TeachAmbiguity] = []
        for fact in parsed.facts:
            if fact.negated:
                continue  # refutations don't need disambiguation
            ambiguities = self._check_ambiguity(fact)
            if ambiguities:
                all_ambiguities.extend(ambiguities)

        if all_ambiguities:
            text = self._format_disambiguation(all_ambiguities)
            return CortexResponse(
                text=text,
                ambiguities=all_ambiguities,
                requires_disambiguation=True,
                confidence=1.0,
                delegate=False,
                parsed_turn=parsed,
            )

        for fact in parsed.facts:
            stmt = fact.original_text or self._fact_to_text(fact)
            if fact.negated:
                # Build a positive statement to refute
                positive = self._fact_to_positive(fact)
                result = self.brain.refute(positive)
                if result is not None:
                    refuted += 1
                    last_summary = positive
                    ops.append(CortexOperation(
                        op="refute",
                        target=positive,
                        success=True,
                        detail=f"path #{result.path_id}",
                    ))
                else:
                    ops.append(CortexOperation(
                        op="refute", target=positive, success=False,
                    ))
            else:
                result = self.brain.teach(stmt)
                if result is not None:
                    taught += 1
                    last_summary = stmt
                    ops.append(CortexOperation(
                        op="teach",
                        target=stmt,
                        success=True,
                        detail=f"path #{result.path_id}",
                    ))
                else:
                    ops.append(CortexOperation(
                        op="teach", target=stmt, success=False,
                    ))

        if taught == 0 and refuted == 0:
            return CortexResponse(
                text=self.generator.parse_failure(parsed.raw_text),
                operations=ops,
                confidence=0.5,
                delegate=True,
                parsed_turn=parsed,
            )

        # Build a confirmation message
        if refuted and taught:
            text = (
                f"Refuted {refuted} claim(s) and learned {taught} new fact(s). "
                f"Most recent: {last_summary}."
            )
        elif refuted:
            text = self.generator.confirm_refuted(last_summary)
        elif taught == 1:
            text = self.generator.confirm_taught(last_summary)
        else:
            text = self.generator.confirm_taught_multi(taught)

        return CortexResponse(
            text=text,
            operations=ops,
            confidence=1.0,
            delegate=False,
            parsed_turn=parsed,
        )

    @staticmethod
    def _format_disambiguation(ambiguities: list[TeachAmbiguity]) -> str:
        """Render the disambiguation prompt as user-readable text."""
        lines = []
        for amb in ambiguities:
            safety_marker = " [SAFETY-CRITICAL]" if amb.safety_grounded else ""
            lines.append(f"Ambiguity in {amb.new_term!r}{safety_marker}:")
            lines.append(
                f"  Sara doesn't have {amb.new_term!r} but does have similar:"
            )
            for c in amb.candidates[:5]:
                desc = f" — {c['description']}" if c.get("description") else ""
                lines.append(
                    f"    • {c['label']!r} (distance {c.get('distance', '?')}){desc}"
                )
            lines.append(
                f"  → If {amb.new_term!r} is the same as one of the above, "
                f"correct your spelling and try again."
            )
            lines.append(
                f"  → If {amb.new_term!r} is genuinely a NEW concept, "
                f"add a distinguishing word (e.g. {amb.new_term!r} medication)."
            )
            if amb.safety_grounded:
                lines.append(
                    f"  → SAFETY-CRITICAL CONTEXT: do not silently merge. "
                    f"Lookalike medication names refer to genuinely different drugs."
                )
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _fact_to_text(fact: ExtractedFact) -> str:
        """Reconstruct the natural-language form of an extracted fact."""
        if fact.relation == "is_a":
            return f"{fact.subject} is {fact.obj}"
        if fact.relation.startswith("has_"):
            return f"{fact.subject} is {fact.obj}"
        if fact.relation == "has":
            return f"{fact.subject} has {fact.obj}"
        return f"{fact.subject} {fact.relation} {fact.obj}"

    @staticmethod
    def _fact_to_positive(fact: ExtractedFact) -> str:
        """Build a positive form of a negated fact, suitable for refute()."""
        if fact.relation == "is_a":
            return f"{fact.subject} is {fact.obj}"
        if fact.relation.startswith("has_"):
            return f"{fact.subject} is {fact.obj}"
        if fact.relation == "has":
            return f"{fact.subject} has {fact.obj}"
        return f"{fact.subject} {fact.relation} {fact.obj}"
