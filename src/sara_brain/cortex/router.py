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

    def __init__(self, brain: Brain) -> None:
        self.brain = brain
        self.parser = EnhancedParser(brain.taxonomy)
        self.generator = TemplateGenerator()

    def process(self, text: str) -> CortexResponse:
        """Process one user turn through the cortex."""
        text = text.strip()
        if not text:
            return CortexResponse(text="", confidence=0.0)

        parsed = self.parser.parse(text)

        if parsed.kind == TurnKind.GREETING:
            return self._handle_greeting(parsed)

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

    def _handle_assertion(self, parsed: ParsedTurn) -> CortexResponse:
        ops: list[CortexOperation] = []
        taught = 0
        refuted = 0
        last_summary = ""

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
