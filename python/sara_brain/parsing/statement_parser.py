"""Simple rule-based parser for teaching statements.

Handles patterns like:
  "apples are red"       → (apple, has_color, red)
  "circles are round"    → (circle, has_shape, round)
  "a dog is an animal"   → (dog, is_a, animal)
"""

from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import Taxonomy


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

        # Pattern: "<subject> is/are <object>"
        # Find the verb
        verb_idx = None
        for i, w in enumerate(words):
            if w in ("is", "are"):
                verb_idx = i
                break

        if verb_idx is None or verb_idx == 0 or verb_idx >= len(words) - 1:
            return None

        # Subject is everything before the verb (singularized)
        subject_words = words[:verb_idx]
        subject = self._singularize(" ".join(subject_words))

        # Object is everything after the verb
        obj_words = words[verb_idx + 1:]
        obj = " ".join(obj_words)

        # Determine relation type from taxonomy
        prop_type = self.taxonomy.property_type(obj)
        if prop_type != "attribute":
            relation = f"has_{prop_type}"
        else:
            relation = "is_a"

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
