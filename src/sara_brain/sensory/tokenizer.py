"""Tokenizer for the sensory shell.

No learned vocabulary. No BPE. No training. Just:
1. Split text into words
2. Look up multi-word phrases in Sara's graph (greedy longest match)
3. Expand synonyms via Sara's dictionary region
4. Return tokens that become wavefront seeds

The tokenizer uses Sara Brain's existing neuron labels as its
vocabulary. If a word isn't in Sara's graph, it stays as-is —
Sara might still find paths for it through synonym bridging.
"""

from __future__ import annotations

import re

# Articles and stopwords to strip before seeding wavefronts.
# These carry no semantic weight in Sara's graph.
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "has", "have", "had", "will",
    "shall", "would", "could", "should", "may", "might", "can",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "not", "no", "if", "then", "so", "as",
    "it", "its", "this", "that", "these", "those",
    "what", "who", "where", "when", "why", "how", "which",
    "tell", "me", "about", "describe", "explain", "define",
})

# Pattern for splitting text into word tokens.
_WORD_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def _singularize(word: str) -> str:
    """Basic English singularization — same rules as the statement parser.

    Used only for graph lookup: try the singular form, but if Sara
    doesn't know it, the original form is kept. This prevents
    "osmosis" → "osmosi" when Sara was taught "osmosis".
    """
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes", "ses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class Token:
    """A single token extracted from input text."""

    __slots__ = ("label", "is_phrase", "neuron_id")

    def __init__(self, label: str, is_phrase: bool = False,
                 neuron_id: int | None = None) -> None:
        self.label = label
        self.is_phrase = is_phrase
        self.neuron_id = neuron_id

    def __repr__(self) -> str:
        kind = "phrase" if self.is_phrase else "word"
        nid = f" #{self.neuron_id}" if self.neuron_id else ""
        return f"Token({self.label!r}, {kind}{nid})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Token):
            return NotImplemented
        return self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)


class Tokenizer:
    """Decomposes text into wavefront seeds using Sara's graph.

    The tokenizer has no knowledge of its own. It uses Sara's neuron
    labels to detect multi-word phrases (greedy longest match) and
    strips stopwords that carry no semantic weight.
    """

    def __init__(self, brain) -> None:
        self.brain = brain

    def tokenize(self, text: str) -> list[Token]:
        """Split text into tokens suitable as wavefront seeds.

        1. Lowercase and extract word tokens
        2. Greedy longest-match against Sara's neuron labels
           (catches multi-word phrases like "electron transport chain")
        3. Strip stopwords
        4. Resolve each token to a neuron ID if it exists in the graph
        """
        words = _WORD_RE.findall(text.lower())
        if not words:
            return []

        tokens: list[Token] = []
        i = 0
        while i < len(words):
            # Greedy longest match: try 5-word, 4-word, ... 2-word phrases
            matched = False
            for length in range(min(5, len(words) - i), 1, -1):
                phrase = " ".join(words[i:i + length])
                neuron = self.brain.neuron_repo.get_by_label(phrase)
                if neuron is not None:
                    tokens.append(Token(
                        label=phrase,
                        is_phrase=True,
                        neuron_id=neuron.id,
                    ))
                    i += length
                    matched = True
                    break
            if not matched:
                word = words[i]
                if word not in _STOP:
                    neuron = self.brain.neuron_repo.get_by_label(word)
                    if neuron is None:
                        # Try singularized form — but only keep it if
                        # Sara actually knows the singular version.
                        # Prevents "osmosis" → "osmosi" when Sara was
                        # taught "osmosis".
                        singular = _singularize(word)
                        if singular != word:
                            s_neuron = self.brain.neuron_repo.get_by_label(singular)
                            if s_neuron is not None:
                                word = singular
                                neuron = s_neuron
                    tokens.append(Token(
                        label=word,
                        is_phrase=False,
                        neuron_id=neuron.id if neuron else None,
                    ))
                i += 1

        return tokens

    def seed_labels(self, text: str) -> list[str]:
        """Convenience: tokenize and return just the labels for wavefront seeding."""
        return [t.label for t in self.tokenize(text)]
