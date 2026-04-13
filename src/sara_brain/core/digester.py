"""Document digestion loop — learn from text the way perception learns from images.

Read → learn → report understanding → parent corrects → ask about unknowns →
LLM breaks them down → parent confirms → associations grow → next doc is richer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..innate.primitives import is_innate, get_all
from ..nlp.reader import DocumentReader


@dataclass
class DigestionStep:
    """One phase of the digestion loop."""
    phase: str  # "read", "directed", "report", "explain"
    statements: list[str] = field(default_factory=list)
    taught_count: int = 0
    unknown_concepts: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class DigestionResult:
    """Complete result of a document digestion cycle."""
    source: str  # filepath or "text"
    steps: list[DigestionStep] = field(default_factory=list)
    total_taught: int = 0
    all_statements: list[str] = field(default_factory=list)
    unknown_concepts: list[str] = field(default_factory=list)
    summary: str = ""


class Digester:
    """Orchestrates the document learning loop.

    Same pattern as Perceiver but for text:
    1. LLM reads doc, extracts facts
    2. Sara learns each fact as paths
    3. Sara checks associations, asks directed questions
    4. LLM voices what Sara understood
    5. User corrects, Sara asks about unknowns
    6. LLM breaks down unknowns using innate primitives
    7. Associations carry forward to next document
    """

    def __init__(self, brain, reader: DocumentReader) -> None:
        self.brain = brain
        self.reader = reader

    def ingest(
        self,
        text: str,
        source: str = "text",
        callback: Callable[[DigestionStep], None] | None = None,
    ) -> DigestionResult:
        """Run the full digestion loop on a document.

        Args:
            text: The document content.
            source: Label for the source (filepath or description).
            callback: Called after each step for interactive display.

        Returns:
            DigestionResult with all steps and what was learned.
        """
        result = DigestionResult(source=source)

        # --- Phase 1: Read ---
        step = DigestionStep(phase="read")
        statements = self.reader.read(text)
        step.statements = statements

        # Temporal linker — connects facts to dates found in their source text
        from .temporal import TemporalLinker
        temporal = TemporalLinker(
            self.brain.neuron_repo,
            self.brain.segment_repo,
            self.brain.path_repo,
        )

        taught = 0
        for stmt in statements:
            r = self.brain.teach(stmt)
            if r is not None:
                taught += 1
                # If the statement mentions a date/era, link the fact
                # to temporal neurons grounding in TEMPORAL primitives
                concept = self.brain.neuron_repo.get_by_id(
                    self.brain.path_repo.get_by_id(r.path_id).terminus_id
                )
                if concept:
                    temporal.link_fact_to_time(concept.id, stmt)
        step.taught_count = taught
        result.total_taught += taught
        result.all_statements.extend(statements)

        if callback:
            callback(step)
        result.steps.append(step)

        # --- Phase 2: Directed Inquiry ---
        associations = self.brain.list_associations()
        if associations:
            step = DigestionStep(phase="directed")
            new_statements = self.reader.inquire(text, associations)
            step.statements = new_statements

            taught = 0
            for stmt in new_statements:
                r = self.brain.teach(stmt)
                if r is not None:
                    taught += 1
            step.taught_count = taught
            result.total_taught += taught
            result.all_statements.extend(new_statements)

            if callback:
                callback(step)
            result.steps.append(step)

        # --- Phase 3: Find unknowns ---
        unknowns = self._find_unknown_concepts(result.all_statements)
        result.unknown_concepts = unknowns

        # --- Phase 4: Explain unknowns via cortex ---
        if unknowns:
            step = DigestionStep(phase="explain")
            step.unknown_concepts = unknowns
            all_explanations: list[str] = []

            for concept in unknowns:
                explanations = self.reader.explain(concept)
                for stmt in explanations:
                    r = self.brain.teach(stmt)
                    if r is not None:
                        step.taught_count += 1
                all_explanations.extend(explanations)

            step.statements = all_explanations
            result.total_taught += step.taught_count
            result.all_statements.extend(all_explanations)

            if callback:
                callback(step)
            result.steps.append(step)

        # --- Phase 5: Report ---
        step = DigestionStep(phase="report")
        summary = self.reader.summarize(result.all_statements)
        step.summary = summary
        result.summary = summary

        if callback:
            callback(step)
        result.steps.append(step)

        self.brain.conn.commit()
        return result

    def _find_unknown_concepts(self, statements: list[str]) -> list[str]:
        """Find concepts in statements that Sara doesn't know and aren't innate."""
        unknowns: list[str] = []
        seen: set[str] = set()

        for stmt in statements:
            # Extract words that might be concepts
            words = self._extract_concepts(stmt)
            for word in words:
                if word in seen:
                    continue
                seen.add(word)
                # Skip if innate
                if is_innate(word):
                    continue
                # Skip if Sara already knows it
                neuron = self.brain.neuron_repo.get_by_label(word)
                if neuron is not None:
                    continue
                unknowns.append(word)

        return unknowns

    @staticmethod
    def _extract_concepts(statement: str) -> list[str]:
        """Pull potential concept words from a statement."""
        # Strip common relational words to find the nouns/concepts
        skip = get_all() | {"a", "an", "the", "and", "or", "of", "to", "in",
                            "for", "with", "at", "by", "from", "on", "are",
                            "be", "been", "being", "was", "were", "will",
                            "should", "must", "can", "could", "would", "may",
                            "might", "shall", "do", "does", "did", "not", "no"}
        words: list[str] = []
        for word in statement.lower().split():
            cleaned = word.strip(".,;:!?\"'()-")
            if cleaned and cleaned not in skip and len(cleaned) > 2:
                words.append(cleaned)
        return words
