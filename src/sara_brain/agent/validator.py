"""Validator — Sara Brain checks LLM proposals before execution.

The cerebellum checking motor plans: the cortex (LLM) proposes,
Sara validates against known paths, and corrects if conflicts exist.
The loop continues until no conflicts remain or max rounds reached.

Conflict detection is keyword/path-based (not LLM-based) to avoid recursion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .bridge import AgentBridge


@dataclass
class ValidationResult:
    approved: bool
    correction: str | None = None
    relevant_knowledge: str | None = None


class Validator:
    """Validate LLM proposals against Sara Brain's knowledge."""

    def __init__(self, bridge: AgentBridge) -> None:
        self.bridge = bridge

    def check_proposal(self, proposal: str) -> ValidationResult:
        """Check a proposed action against Sara's knowledge.

        1. Extract key concepts from the proposal
        2. Query Sara for relevant knowledge
        3. Detect conflicts between proposal and Sara's knowledge
        4. Return approved or correction
        """
        concepts = self.extract_concepts(proposal)
        if not concepts:
            return ValidationResult(approved=True)

        # Query Sara for relevant knowledge
        knowledge = self.bridge.context(" ".join(concepts))

        # "Sara has no knowledge" means no conflict possible
        if knowledge.startswith("Sara has no knowledge"):
            return ValidationResult(approved=True)

        # Check for conflicts
        conflict = self.detect_conflicts(proposal, knowledge)
        if conflict:
            correction = self.format_correction(proposal, conflict, knowledge)
            return ValidationResult(
                approved=False,
                correction=correction,
                relevant_knowledge=knowledge,
            )

        return ValidationResult(
            approved=True,
            relevant_knowledge=knowledge,
        )

    def extract_concepts(self, text: str) -> list[str]:
        """Pull meaningful keywords from a proposal for brain lookup.

        Filters out common stop words and short words, returns unique terms.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "and", "but", "or", "nor", "not", "so",
            "yet", "both", "either", "neither", "each", "every", "all",
            "any", "few", "more", "most", "other", "some", "such", "no",
            "only", "own", "same", "than", "too", "very", "just", "because",
            "if", "when", "while", "where", "how", "what", "which", "who",
            "that", "this", "these", "those", "then", "there", "here",
            "up", "out", "about", "it", "its", "they", "them", "their",
            "we", "us", "our", "you", "your", "he", "she", "him", "her",
            "i", "me", "my", "file", "code", "run", "use", "using",
            "write", "read", "create", "make", "want", "need", "like",
            "please", "let", "get", "set", "put", "call", "try",
        }

        # Extract words, normalize
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower())
        seen: set[str] = set()
        concepts = []
        for w in words:
            if w not in stop_words and len(w) > 2 and w not in seen:
                seen.add(w)
                concepts.append(w)
        return concepts

    def detect_conflicts(self, proposal: str, knowledge: str) -> str | None:
        """Compare proposal against Sara's knowledge.

        Returns a conflict description if found, None if no conflict.

        Conflict detection strategy:
        - Extract factual claims from knowledge (path traces)
        - Check if the proposal contradicts any known facts
        - Focus on direct contradictions (X is Y vs X is Z)
        """
        proposal_lower = proposal.lower()
        knowledge_lower = knowledge.lower()

        # Extract known facts from Sara's knowledge
        # Format: "[keyword] property → relation → concept (from: "source")"
        known_facts = self._parse_known_facts(knowledge)

        conflicts = []
        for fact_key, fact_value, source in known_facts:
            # Check if proposal mentions the same subject with a different claim
            if fact_key in proposal_lower:
                # Look for contradictions: proposal says X but Sara knows Y
                # This is deliberately conservative — only flags clear contradictions
                negation = self._check_negation(
                    proposal_lower, fact_key, fact_value
                )
                if negation:
                    conflicts.append(
                        f"Sara knows '{fact_key}' is '{fact_value}'"
                        + (f' (from: "{source}")' if source else "")
                        + f", but proposal suggests: {negation}"
                    )

        return "\n".join(conflicts) if conflicts else None

    def _parse_known_facts(
        self, knowledge: str
    ) -> list[tuple[str, str, str]]:
        """Extract (subject, property, source) tuples from Sara's knowledge."""
        facts = []
        for line in knowledge.split("\n"):
            line = line.strip()
            if not line or line.startswith("Sara knows"):
                continue

            # Parse path traces: "[keyword] prop → rel → concept"
            # or "prop → rel → concept (from: "text")"
            source = ""
            source_match = re.search(r'\(from: "([^"]+)"\)', line)
            if source_match:
                source = source_match.group(1)
                line = line[: source_match.start()].strip()

            # Remove leading [keyword] bracket
            line = re.sub(r"^\[.*?\]\s*", "", line)

            # Split on arrows
            parts = [p.strip() for p in line.split("→")]
            if len(parts) >= 2:
                prop = parts[0].strip()
                concept = parts[-1].strip()
                facts.append((concept, prop, source))

        return facts

    def _check_negation(
        self, proposal: str, subject: str, known_property: str
    ) -> str | None:
        """Check if proposal contradicts a known property.

        Returns the contradicting claim if found, None otherwise.
        Very conservative: only flags when proposal explicitly uses
        a different value for the same subject+category.
        """
        # Look for "not <known_property>" patterns
        if f"not {known_property}" in proposal:
            return f"says 'not {known_property}'"

        # Look for "don't use <known_property>" patterns
        if f"don't use {known_property}" in proposal or f"dont use {known_property}" in proposal:
            return f"says 'don't use {known_property}'"

        # Look for "instead of <known_property>" patterns
        if f"instead of {known_property}" in proposal:
            return f"says 'instead of {known_property}'"

        return None

    def format_correction(
        self, proposal: str, conflict: str, knowledge: str
    ) -> str:
        """Format Sara's correction for the LLM to incorporate."""
        return (
            f"Sara Brain has detected a conflict with your proposal.\n\n"
            f"Conflict:\n{conflict}\n\n"
            f"Sara's relevant knowledge:\n{knowledge}\n\n"
            f"Please adjust your approach to align with Sara's knowledge. "
            f"Sara's knowledge comes from the user's teachings and is trusted."
        )
