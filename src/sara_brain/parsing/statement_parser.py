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
# All relational primitives minus "is" (covered by copulas above)
_RELATIONAL_VERBS = get_relational() - {"is", "has"}


@dataclass
class ParsedStatement:
    subject: str
    relation: str
    obj: str
    original: str


class StatementParser:
    def __init__(self, taxonomy: Taxonomy) -> None:
        self.taxonomy = taxonomy

    def parse(self, text: str) -> ParsedStatement | None:
        """Parse a natural-language teaching statement."""
        text = text.strip()
        if not text:
            return None

        original = text
        # Normalize: lowercase, strip articles
        words = text.lower().split()
        words = [w for w in words if w not in ("a", "an", "the")]

        if len(words) < 3:
            return None

        # Pattern: "<subject> <verb> <object>"
        # Recognised verbs: copulas (is/are/was/were) + possession (has/have/had)
        # + all innate RELATIONAL primitives
        _ALL_VERBS = _COPULAS | _POSSESSION | _RELATIONAL_VERBS
        verb_idx = None
        verb_word = None
        for i, w in enumerate(words):
            if w in _ALL_VERBS:
                verb_idx = i
                verb_word = w
                break

        if verb_idx is None or verb_idx == 0 or verb_idx >= len(words) - 1:
            return None

        # Subject is everything before the verb (singularized)
        subject_words = words[:verb_idx]
        subject = self._singularize(" ".join(subject_words))

        # Object is everything after the verb
        obj_words = words[verb_idx + 1:]
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
