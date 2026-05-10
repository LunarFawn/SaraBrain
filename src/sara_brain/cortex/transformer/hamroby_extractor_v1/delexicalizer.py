"""Delexicalize real prose while preserving syntactic structure.

The core insight (Jennifer Pearl, 2026-05-09): replace each open-class
content word with a deterministic nonsense substitute, keep closed-class
function words verbatim. The same surface word always maps to the same
nonsense token within a corpus, so anaphoric references and repeated
mentions stay coherent.

Result: text with the EXACT real-world grammatical distribution but
zero domain content the head can memorize. Subordinate clauses,
gerund subjects, parentheticals, anaphora — all show up at their
natural frequency. Used to train the grammar-feature transformer on
real-distribution syntax without breaking content orthogonality.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# spaCy POS tags. Closed-class function words stay verbatim; open-class
# content words get substituted. Punctuation stays as-is.
_CLOSED_CLASS_POS: frozenset[str] = frozenset({
    "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PRON", "PART",
})
_KEEP_AS_IS_POS: frozenset[str] = frozenset({"PUNCT", "SPACE"})


# Pronounceable nonsense generator — same shape as the upstream
# `_random_word` in papers/instrument_validation/generate_synthetic_substrate.py.
# Reproduced here so we don't need to import the heavy generator module.
_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"


def _random_word(rng: random.Random, min_len: int = 5, max_len: int = 8) -> str:
    """One pronounceable nonsense word."""
    length = rng.randint(min_len, max_len)
    word = []
    use_consonant = rng.random() < 0.5
    for _ in range(length):
        word.append(rng.choice(_CONSONANTS if use_consonant else _VOWELS))
        use_consonant = not use_consonant
    return "".join(word)


@dataclass
class DelexMapping:
    """Persistent surface-word → nonsense map. Stays consistent within
    a corpus so repeated mentions of the same word produce the same
    nonsense token."""
    word_to_nonsense: dict[str, str] = field(default_factory=dict)
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def substitute(self, word: str) -> str:
        """Get the nonsense token for `word` (case-insensitive). The
        nonsense token's length tracks the original word's length within
        a window, so the rendered prose has plausible word-length
        distribution."""
        key = word.lower()
        if key not in self.word_to_nonsense:
            # Length window roughly matches the original ±2 chars,
            # clamped to a sensible range.
            target = max(3, min(12, len(word)))
            min_len = max(3, target - 2)
            max_len = min(12, target + 2)
            if min_len > max_len:
                min_len = max_len
            while True:
                cand = _random_word(self.rng, min_len=min_len, max_len=max_len)
                if cand not in self.word_to_nonsense.values():
                    self.word_to_nonsense[key] = cand
                    break
        return self.word_to_nonsense[key]


def delexicalize_text(
    text: str,
    nlp,
    *,
    mapping: DelexMapping | None = None,
) -> tuple[str, DelexMapping]:
    """Delexicalize prose. Returns (delex_text, mapping).

    Open-class content words → nonsense substitutes (deterministic per
    word). Closed-class function words pass through verbatim.
    Punctuation passes through. Capitalization of the original
    SUBSTITUTED word is dropped (nonsense is always lowercase).

    The output is space-joined; punctuation is attached to its
    preceding token (matches normal spaCy text reconstruction).
    """
    if mapping is None:
        mapping = DelexMapping()
    doc = nlp(text)
    out_pieces: list[str] = []
    for tok in doc:
        if tok.is_punct or tok.pos_ in _KEEP_AS_IS_POS:
            out_pieces.append(tok.text)
            continue
        if tok.pos_ in _CLOSED_CLASS_POS:
            out_pieces.append(tok.text)
            continue
        # Open-class content word — substitute.
        out_pieces.append(mapping.substitute(tok.text))
    # Reattach with single spaces; downstream tokenizers re-split.
    return " ".join(out_pieces), mapping


def delexicalize_phrase(phrase: str, mapping: DelexMapping) -> str:
    """Apply the same word-level mapping to a multi-word phrase
    (e.g. an extracted triple's subject or object). Words not in the
    mapping fall through unchanged — this is what we want for closed-
    class words ("the", "of") and any open-class word that didn't
    appear in the source prose under the same form."""
    out_words: list[str] = []
    for word in phrase.split():
        key = word.lower().strip(",.;:!?\"'()[]")
        suffix = ""
        # Strip trailing punctuation so it doesn't become a fresh
        # vocabulary entry.
        while word and word[-1] in ",.;:!?\"'()[]":
            suffix = word[-1] + suffix
            word = word[:-1]
        key = word.lower()
        if key in mapping.word_to_nonsense:
            out_words.append(mapping.word_to_nonsense[key] + suffix)
        else:
            out_words.append(word + suffix)
    return " ".join(out_words)


__all__ = [
    "DelexMapping",
    "delexicalize_text",
    "delexicalize_phrase",
]
