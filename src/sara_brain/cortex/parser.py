"""Sara Cortex enhanced parser.

The cortex parser wraps the existing rule-based StatementParser and adds:

- Question vs statement detection
- Compound statement splitting (one input → many sub-statements)
- Source/authority extraction
- Quantifier / strength weighting
- Negation propagation
- Topic extraction for queries

It returns a single ParsedTurn describing the user's intent at a high
level — what kind of turn this is and what brain operations should fire.

The cortex parser does NOT call the brain. It only analyzes language.
The router consumes its output and decides what brain operations to run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ..parsing.statement_parser import ParsedStatement, StatementParser
from ..parsing.taxonomy import Taxonomy
from . import grammar


class TurnKind(Enum):
    """The kind of turn the user just took."""
    QUESTION = "question"        # asking Sara about something
    STATEMENT = "statement"      # asserting facts (teach)
    NEGATION = "negation"        # asserting facts are false (refute)
    CORRECTION = "correction"    # correcting a previous response
    GREETING = "greeting"        # social, no knowledge op needed
    UNKNOWN = "unknown"          # cortex can't classify confidently


@dataclass
class ExtractedFact:
    """A single subject-relation-object triple extracted from user input."""
    subject: str
    relation: str
    obj: str
    negated: bool = False
    source: str | None = None       # citation if user gave one
    confidence: float = 1.0          # 0..1, modulated by quantifiers
    original_text: str = ""          # the substring this came from


@dataclass
class ParsedTurn:
    """High-level analysis of one user turn."""
    kind: TurnKind
    raw_text: str
    facts: list[ExtractedFact] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)  # for questions: what to look up
    confidence: float = 1.0   # how sure the cortex is about its analysis

    @property
    def is_question(self) -> bool:
        return self.kind == TurnKind.QUESTION

    @property
    def is_assertion(self) -> bool:
        return self.kind in (TurnKind.STATEMENT, TurnKind.NEGATION, TurnKind.CORRECTION)


class EnhancedParser:
    """Cortex-level parser. Wraps StatementParser with more structure.

    The cortex defaults to strict_dialect=True. This means the parser
    makes ZERO assumptions about word roles. Articles are not stripped,
    spelling is not canonicalized, dialect words are preserved exactly.
    The user is the sole authority on what their words mean. See
    feedback_never_assume_dialect.md for the principle.
    """

    def __init__(
        self,
        taxonomy: Taxonomy | None = None,
        strict_dialect: bool = True,
    ) -> None:
        self.taxonomy = taxonomy or Taxonomy()
        self.base = StatementParser(self.taxonomy, strict_dialect=strict_dialect)
        self.strict_dialect = strict_dialect

    def parse(self, text: str) -> ParsedTurn:
        """Analyze a user turn and return a structured ParsedTurn.

        The returned object describes intent and contains zero or more
        ExtractedFact objects ready for brain operations.
        """
        text = text.strip()
        if not text:
            return ParsedTurn(kind=TurnKind.UNKNOWN, raw_text="")

        text_lower = text.lower()

        # ── Greeting detection (social, no knowledge op) ──
        if self._is_greeting(text_lower):
            return ParsedTurn(kind=TurnKind.GREETING, raw_text=text)

        # ── Question detection ──
        if self._is_question(text_lower):
            topics = self._extract_topics(text_lower)
            return ParsedTurn(
                kind=TurnKind.QUESTION,
                raw_text=text,
                topics=topics,
                confidence=1.0 if topics else 0.5,
            )

        # ── Source extraction ──
        source, body = self._extract_source(text)
        body_lower = body.lower()

        # ── Compound statement splitting ──
        sub_statements = self._split_compound(body)

        # ── Parse each sub-statement ──
        facts: list[ExtractedFact] = []
        any_negated = False
        for sub in sub_statements:
            if not sub.strip():
                continue
            quantifier_weight = self._extract_quantifier(sub)
            cleaned_sub = self._strip_quantifiers(sub)
            parsed: ParsedStatement | None = self.base.parse(cleaned_sub)
            if parsed is None:
                continue
            fact = ExtractedFact(
                subject=parsed.subject,
                relation=parsed.relation,
                obj=parsed.obj,
                negated=parsed.negated,
                source=source,
                confidence=quantifier_weight,
                original_text=sub.strip(),
            )
            if parsed.negated:
                any_negated = True
            facts.append(fact)

        # ── Determine kind ──
        if not facts:
            return ParsedTurn(
                kind=TurnKind.UNKNOWN,
                raw_text=text,
                confidence=0.0,
            )

        if any_negated and len(facts) == 1:
            kind = TurnKind.NEGATION
        elif any_negated:
            kind = TurnKind.CORRECTION  # mixed teach + refute
        else:
            kind = TurnKind.STATEMENT

        return ParsedTurn(
            kind=kind,
            raw_text=text,
            facts=facts,
            confidence=1.0,
        )

    # ── helpers ──

    @staticmethod
    def _is_greeting(text_lower: str) -> bool:
        greetings = (
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings", "howdy", "yo",
        )
        first_word = text_lower.split()[0] if text_lower.split() else ""
        return first_word in greetings or text_lower in greetings

    @staticmethod
    def _is_question(text_lower: str) -> bool:
        if text_lower.endswith("?"):
            return True
        first_word = text_lower.split()[0] if text_lower.split() else ""
        return first_word in grammar.QUESTION_PREFIXES

    def _extract_topics(self, text_lower: str) -> list[str]:
        """Pull candidate topic words from a question.

        Returns content nouns (words that aren't stopwords, question words,
        or typo'd articles). The article filter is critical: without it,
        a query like "what is teh edubba" would treat "teh" as a topic and
        return all the historical typo-pollution paths attached to it.
        """
        from ..parsing.statement_parser import _ARTICLE_FORMS
        # Strip the leading question word
        words = text_lower.replace("?", "").split()
        if not words:
            return []
        # Drop the first word if it's a question word
        if words[0] in grammar.QUESTION_PREFIXES:
            words = words[1:]
        # Drop stopwords, short words, AND typo'd article forms
        topics = []
        for w in words:
            w_clean = w.strip(".,;:!?\"'()-")
            if (
                w_clean
                and len(w_clean) > 2
                and w_clean not in grammar.STOPWORDS
                and w_clean not in grammar.QUESTION_PREFIXES
                and w_clean not in _ARTICLE_FORMS
            ):
                topics.append(w_clean)
        return topics

    def _extract_source(self, text: str) -> tuple[str | None, str]:
        """If the text starts with a source phrase, extract it.

        Returns (source, remaining_body).
        """
        text_lower = text.lower()
        for phrase in grammar.SOURCE_PHRASES:
            if text_lower.startswith(phrase):
                rest = text[len(phrase):].strip()
                # Try to find where the source ends and the fact begins.
                # Patterns we handle:
                #   "according to wikipedia, the edubba was..."  → comma split
                #   "i read in wikipedia that the edubba..."     → "that" split
                #   "wikipedia says the edubba..."               → "says" split
                m = re.search(r"\b(says|that|claims|tells us)[,\s]+", rest.lower())
                if m:
                    source = rest[:m.start()].strip().rstrip(",.:;")
                    body = rest[m.end():].strip()
                    return source, body
                # Try comma split — "according to X, Y"
                if "," in rest:
                    source, _, body = rest.partition(",")
                    return source.strip().rstrip(".:;"), body.strip()
                # Otherwise the whole rest is unstructured — fall through
                return None, text
        return None, text

    @staticmethod
    def _split_compound(text: str) -> list[str]:
        """Split a compound statement into sub-statements."""
        # Try sentence-level splits first
        parts = [text]
        for term in grammar.SENTENCE_TERMINATORS:
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(term))
            parts = new_parts
        # Then conjunction splits
        for conj in grammar.CONJUNCTIONS:
            new_parts = []
            for p in parts:
                new_parts.extend(p.split(conj))
            parts = new_parts
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _extract_quantifier(text: str) -> float:
        """Return a confidence multiplier based on hedge/strength words."""
        words = set(text.lower().split())
        if words & grammar.QUANTIFIERS_HIGH:
            return 1.0
        if words & grammar.QUANTIFIERS_LOW:
            return 0.5
        if words & grammar.QUANTIFIERS_NEGATING:
            return 0.2
        return 1.0

    @staticmethod
    def _strip_quantifiers(text: str) -> str:
        """Remove quantifier words so the underlying parser sees clean input."""
        all_q = (
            grammar.QUANTIFIERS_HIGH
            | grammar.QUANTIFIERS_LOW
            | grammar.QUANTIFIERS_NEGATING
        )
        words = [w for w in text.split() if w.lower() not in all_q]
        return " ".join(words)
