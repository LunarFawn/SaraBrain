"""Post-extraction label normalization.

Cleans subject/object labels emitted by the sara extractor before
they become neurons. Rejects garbage, normalizes plurals, strips
punctuation framing.
"""
from __future__ import annotations

import re

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "can", "may", "might", "shall", "must",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "he", "she", "him", "her", "his", "we", "us", "our", "i", "me", "my",
    "you", "your", "who", "whom", "which", "what", "where", "when", "how",
    "and", "or", "but", "not", "no", "nor", "yet", "so", "also",
    "if", "then", "than", "both", "either", "neither",
    "from", "to", "in", "on", "at", "by", "for", "with", "as", "of",
    "into", "onto", "upon", "during", "between", "after", "before",
    "through", "over", "under", "about", "above", "below",
    "each", "every", "all", "some", "any", "many", "much", "more", "most",
    "other", "such", "only", "same", "very", "just", "even", "still",
    "however", "therefore", "thus", "hence", "moreover", "furthermore",
    "there", "here", "where", "now", "then", "always", "never",
})

# Common irregular plurals we don't want to blindly strip 's' from
_IRREGULAR_KEEP = frozenset({
    "nucleus", "stimulus", "apparatus", "process", "consensus",
    "analysis", "basis", "crisis", "thesis", "mitosis", "meiosis",
    "synthesis", "osmosis", "apoptosis", "cytokinesis", "karyokinesis",
})

_PUNCT_STRIP = re.compile(r'^[\s\-\"\'\(\)\[\]\{\}\.\,\;\:\!\?\#\*\/\\]+|[\s\-\"\'\(\)\[\]\{\}\.\,\;\:\!\?\#\*\/\\]+$')


def normalize_label(label: str) -> str | None:
    """Normalize a subject/object label. Returns None if the label is garbage."""
    if not label:
        return None

    # Strip framing punctuation
    label = _PUNCT_STRIP.sub("", label)
    label = label.strip()

    if not label:
        return None

    # Reject pure punctuation/symbols
    if not any(c.isalpha() for c in label):
        return None

    # Reject if entire label is a stopword
    if label.lower() in _STOPWORDS:
        return None

    # Reject single characters
    if len(label) <= 1:
        return None

    # Lowercase
    label = label.lower()

    # Strip leading stopwords ("the cell cycle" → "cell cycle")
    words = label.split()
    while words and words[0] in _STOPWORDS:
        words.pop(0)
    if not words:
        return None
    # Only strip trailing stopwords if they're clearly function words,
    # NOT Roman numerals (I, II, III) or phrasal verb particles (over, up, out)
    _ROMAN = {"i", "ii", "iii", "iv", "v", "vi"}
    _PARTICLES = {"over", "up", "out", "off", "down", "in", "on", "through"}
    while len(words) > 1 and words[-1] in _STOPWORDS \
          and words[-1] not in _ROMAN and words[-1] not in _PARTICLES:
        words.pop()
    if not words:
        return None
    label = " ".join(words)

    # Reject if what remains is a stopword
    if label in _STOPWORDS:
        return None

    # Simple plural normalization (only trailing 's' on regular nouns)
    if label not in _IRREGULAR_KEEP and not label.endswith("ss"):
        if label.endswith("ies") and len(label) > 4:
            # batteries → battery, but not "series"
            candidate = label[:-3] + "y"
            label = candidate
        elif label.endswith("es") and len(label) > 4:
            # processes handled by _IRREGULAR_KEEP
            # "phases" → "phase", "stages" → "stage"
            candidate = label[:-2]
            if candidate.endswith(("sh", "ch", "x", "z", "s")):
                label = candidate
            else:
                label = label[:-1]  # "phases" → "phase"
        elif label.endswith("s") and len(label) > 3:
            label = label[:-1]

    # Final length check
    if len(label) <= 1:
        return None

    return label
