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

# Article forms — including common typos. We strip these aggressively
# during normalization so subjects like "tteh edubba" become "edubba".
# Adding a typo here is cheap and pays for itself the next time someone
# fat-fingers the word.
_ARTICLE_FORMS = frozenset({
    "a", "an", "the",
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


class StatementParser:
    def __init__(self, taxonomy: Taxonomy) -> None:
        self.taxonomy = taxonomy

    def parse(self, text: str) -> ParsedStatement | None:
        """Parse a natural-language teaching statement."""
        text = text.strip()
        if not text:
            return None

        original = text
        # Normalize: lowercase, strip articles (including common typos)
        words = text.lower().split()
        words = [w for w in words if w not in _ARTICLE_FORMS]

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

        return ParsedStatement(
            subject=subject,
            relation=relation,
            obj=obj,
            original=original,
            negated=negated,
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
