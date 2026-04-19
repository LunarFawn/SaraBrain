"""Simple rule-based parser for teaching statements.

Handles patterns like:
  "apples are red"            → (apple, has_color, red)
  "circles are round"         → (circle, has_shape, round)
  "a dog is an animal"        → (dog, is_a, animal)
  "qmse includes auditability"→ (qmse, includes, auditability)
  "stem loop contains terminus" → (stem loop, contains, terminus)
  "RNA requires equilibrium"  → (rna, requires, equilibrium)
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import Taxonomy
from ..innate.primitives import get_relational

# Verbs that use the taxonomy path (is_a / has_<type>)
# Includes past tense and possession verbs so historical and biographical
# facts can be auto-taught: "the edubba was a school", "she had wisdom".
_COPULAS = frozenset({"is", "are", "was", "were", "be", "been", "being"})
# Possession verbs that should also use the taxonomy path
_POSSESSION = frozenset({"has", "have", "had"})
# Auxiliary verbs that signal a negated main-verb construction:
#   "the edubba did not teach akkadian"
#   "apples do not contain protein"
# These are detected separately and the next word is the main verb.
_AUXILIARIES = frozenset({"do", "does", "did", "don't", "doesn't", "didn't"})
# All relational primitives minus "is" (covered by copulas above)
_RELATIONAL_VERBS = get_relational() - {"is", "has"}

# Pronouns that cannot be used as standalone subjects — they reference
# context the parser cannot resolve. A statement like "it was a school"
# is meaningless without knowing what "it" refers to.
_PRONOUN_SUBJECTS = frozenset({
    "it", "they", "them", "this", "that", "these", "those",
    "he", "she", "him", "her", "his", "hers", "its", "their",
    "we", "us", "i", "me", "you", "your", "my", "our",
})

# Standard English articles, only used when strict_dialect=False.
# In dialect-safe mode (the default for the cortex), Sara does NOT
# silently strip even these — what looks like an English article may
# be a content word in another language. Every assumption is the
# user's to approve. See feedback_never_assume_dialect.md.
_ARTICLE_FORMS = frozenset({
    "a", "an", "the",
})

# Known article typo variants. NEVER stripped automatically. Listed only
# so the cleanup utility can present them to the user for explicit
# per-instance review. A user typing "tteh" in their dialect may have
# meant exactly that.
_ARTICLE_TYPO_VARIANTS = frozenset({
    "tthe", "thte", "teh", "tteh", "het", "tha", "thr",
    "ann", "ane", "anne",
    "ah",
})


@dataclass
class ParsedStatement:
    subject: str
    relation: str
    obj: str
    original: str
    negated: bool = False  # True if "X is not Y" form — caller should refute, not teach
    # operation: optional arithmetic intent detected in the statement.
    # Populated by MathResolver during parse() when the text contains
    # phrases like "by half", "doubles", "reduces by 2", etc. The
    # learner attaches this to the relation segment at teach time.
    # Typed as Any to avoid circular imports; checked with hasattr.
    operation: object | None = None


class StatementParser:
    def __init__(self, taxonomy: Taxonomy, strict_dialect: bool = False) -> None:
        """Construct a parser.

        Args:
            taxonomy: The brain's property/category taxonomy.
            strict_dialect: When True, the parser makes ZERO assumptions
                about word roles. No articles stripped, no canonicalization.
                Use this for any context where the user may speak a creole,
                pidgin, or non-standard dialect — what looks like a typo
                may be the correct spelling in their language.

                Default is False for backward compatibility with existing
                tests and the bare REPL. The cortex defaults this to True
                because it enforces "never assume" as a foundational
                principle.
        """
        self.taxonomy = taxonomy
        self.strict_dialect = strict_dialect

    def parse(self, text: str) -> ParsedStatement | None:
        """Parse a natural-language teaching statement."""
        text = text.strip()
        if not text:
            return None

        original = text
        # Normalize: lowercase. In standard mode we also strip exact
        # English articles ("the"/"a"/"an"). In strict_dialect mode we
        # strip nothing — every word stays as the user typed it.
        words = text.lower().split()
        if not self.strict_dialect:
            words = [w for w in words if w not in _ARTICLE_FORMS]

        if len(words) < 3:
            return None

        # ── Multi-word temporal/causal pattern normalization ──
        # Before verb splitting, replace multi-word phrases with
        # canonical single verbs. The dictionary handles single-verb
        # synonyms (triggers → causes). These patterns handle
        # multi-word phrases the dictionary can't resolve.
        text_joined = " ".join(words)
        _MULTI_PATTERNS = [
            # "comes before" / "occurs before" / "happens before" → precedes
            (r"\b(\w+)\s+comes\s+before\s+", r"\1 precedes "),
            (r"\b(\w+)\s+occurs\s+before\s+", r"\1 precedes "),
            (r"\b(\w+)\s+happens\s+before\s+", r"\1 precedes "),
            # "comes after" / "occurs after" / "happens after" → follows
            (r"\b(\w+)\s+comes\s+after\s+", r"\1 follows "),
            (r"\b(\w+)\s+occurs\s+after\s+", r"\1 follows "),
            (r"\b(\w+)\s+happens\s+after\s+", r"\1 follows "),
            # "results in" / "leads to" → causes
            (r"\bresults\s+in\b", "causes"),
            (r"\bleads\s+to\b", "causes"),
            # "is followed by X" → X follows (swap handled by verb logic)
            (r"\bis\s+followed\s+by\b", "precedes"),
            # "is caused by" / "is triggered by" → swap + causes
            (r"\bis\s+caused\s+by\b", "caused_by"),
            (r"\bis\s+triggered\s+by\b", "caused_by"),
            # "first X then Y" → X precedes Y
            (r"\bfirst\s+(.+?)\s+then\s+", r"\1 precedes "),
            # "X and then Y" → X precedes Y
            (r"\band\s+then\b", "precedes"),
        ]
        import re as _re
        for pattern, replacement in _MULTI_PATTERNS:
            text_joined = _re.sub(pattern, replacement, text_joined)
        words = text_joined.split()

        if len(words) < 3:
            return None

        # Pattern: "<subject> <verb> <object>"
        # Recognised verbs: copulas (is/are/was/were) + possession (has/have/had)
        # + all innate RELATIONAL primitives
        # Also auxiliary patterns: "did not teach", "does not contain"
        # which collapse to refute(subject + main_verb + object).
        _ALL_VERBS = _COPULAS | _POSSESSION | _RELATIONAL_VERBS

        # First pass: look for an auxiliary verb followed by "not" + main verb.
        # In this construction we accept ANY main verb (not just those in
        # our primitive set) because the negated form is meaningful even
        # for verbs we don't normally track. The main verb becomes the
        # relation label.
        aux_negated_verb: str | None = None
        verb_idx = None
        verb_word = None
        for i, w in enumerate(words):
            if w in _AUXILIARIES and i + 2 < len(words) and words[i + 1] == "not":
                # "X did not teach Y" → verb_idx = i, main_verb = words[i+2],
                # object starts at i+3, negated=True
                main_verb = words[i + 2]
                # Accept any non-stopword as a main verb in this position
                if main_verb and len(main_verb) > 1 and main_verb not in {"a", "an", "the"}:
                    verb_idx = i
                    verb_word = main_verb
                    aux_negated_verb = main_verb
                    break

        # If no auxiliary pattern, fall back to the regular verb scan
        if verb_idx is None:
            for i, w in enumerate(words):
                if w in _ALL_VERBS:
                    verb_idx = i
                    verb_word = w
                    break

        # ── Fallback 1: temporal/locational preposition split ──
        # "chromosomes condense DURING prophase" → split on "during"
        # The preposition marks where subject+action ends and context begins.
        if verb_idx is None:
            _TEMPORAL_PREPS = frozenset({
                "during", "within", "throughout", "through",
            })
            for i, w in enumerate(words):
                if w in _TEMPORAL_PREPS and i > 1 and i < len(words) - 1:
                    verb_idx = i
                    verb_word = w
                    break

        # ── Fallback 2: any-verb heuristic (verb suffixes) ──
        # "apoptosis eliminateS damaged cells" — verb ends in -s/-es/-ed/-ing
        # Only fires when both verb scan AND preposition scan failed.
        if verb_idx is None:
            _VERB_SUFFIXES = (
                "tes", "ses", "zes", "ges", "ces", "ves", "nes",
                "des", "kes", "les", "mes", "pes", "res",
                "ms", "ns", "rs", "ts", "ls", "ds", "gs", "ks", "ps",
                "ed", "ing",
            )
            _NOT_VERBS = frozenset({
                "cells", "molecules", "proteins", "organisms",
                "genes", "chromosomes", "species", "processes",
                "tissues", "structures", "phases", "stages",
                "things", "names", "ones", "types", "kinds",
            })
            for i, w in enumerate(words):
                if i > 0 and i < len(words) - 1 and len(w) > 3:
                    if w not in _NOT_VERBS and w.endswith(_VERB_SUFFIXES):
                        verb_idx = i
                        verb_word = w
                        break

        if verb_idx is None or verb_idx == 0 or verb_idx >= len(words) - 1:
            return None

        # Subject is everything before the verb. Reject pronoun-only
        # subjects BEFORE singularization, because singularization can
        # mangle pronouns ("this" → "thi" etc.) and we want to catch
        # them in their natural form.
        subject_words_raw = words[:verb_idx]
        if len(subject_words_raw) == 1 and subject_words_raw[0] in _PRONOUN_SUBJECTS:
            return None
        subject = self._singularize(" ".join(subject_words_raw))

        # Also reject if singularization happens to leave a pronoun
        if subject.strip() in _PRONOUN_SUBJECTS:
            return None

        # Reject subjects longer than 4 words — anything longer is almost
        # certainly a parse failure where a sentence fragment got captured
        # as the subject. Real concept names are at most 3-4 words
        # ("green tea", "the edubba", "sickle cell anemia").
        if len(subject.split()) > 4:
            return None

        # Reject subjects that CONTAIN a pronoun (not just ARE one).
        # "for learning akkadian it" contains "it" as a buried pronoun
        # and shouldn't be a subject.
        subject_words_set = set(subject.split())
        if subject_words_set & _PRONOUN_SUBJECTS:
            return None

        # Object is everything after the verb
        # Special case: if we matched an auxiliary pattern, the verb_idx
        # points to the auxiliary, but we already extracted the main verb
        # as verb_word. Skip past "aux + not + main_verb" to get the object.
        if aux_negated_verb is not None:
            obj_words = words[verb_idx + 3:]  # skip aux + "not" + main_verb
            negated = True
        else:
            obj_words = words[verb_idx + 1:]

            # Negation detection: if the next word after the verb is "not",
            # "never", or "no", strip it and mark the statement as negated.
            negated = False
            if obj_words and obj_words[0] in ("not", "never", "no"):
                negated = True
                obj_words = obj_words[1:]

        if not obj_words:
            return None

        obj = " ".join(obj_words)

        # Determine relation: copulas use taxonomy (is_a / has_<type>);
        # possession verbs use the "has" path (always); relational primitives
        # use the verb itself.
        if verb_word in _COPULAS:
            prop_type = self.taxonomy.property_type(obj)
            if prop_type != "attribute":
                relation = f"has_{prop_type}"
            else:
                relation = "is_a"
        elif verb_word in _POSSESSION:
            relation = "has"
        else:
            relation = verb_word

        # Detect an arithmetic operation embedded in the statement
        # (e.g., "by half", "doubles"). MathResolver is imported here
        # lazily to avoid a circular dependency at module import time.
        op = None
        try:
            from ..core.math import MathResolver
            op = MathResolver().resolve(original)
        except Exception:
            op = None

        return ParsedStatement(
            subject=subject,
            relation=relation,
            obj=obj,
            original=original,
            negated=negated,
            operation=op,
        )

    @staticmethod
    def _singularize(word: str) -> str:
        """Very basic English singularization."""
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith(("ches", "shes", "xes", "zes", "ses")):
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        return word
