"""Deep-dive helpers for HamRobyLLM chat.

Two complementary modes:

  Sibling expansion: given a topic, find every substrate label whose
  words intersect with the topic's content words (filtered to non-stop
  words). For "inertia in rna" we surface "higher inertia", "more
  inertia", "law of inertia", "inertia". This reveals structurally
  related concepts the router only picked one of.

  Depth widening: re-run brain_explore at a higher hop distance
  (default 1, /depth 2 walks neighbors-of-neighbors).

Both are user-driven; nothing fires automatically except when the
question itself asks for breadth ("tell me everything about X",
"give me the complete picture of X" — see is_comprehensive_intent).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Words to ignore when computing sibling overlap — generic English particles
# that would otherwise match every multi-word label.
_STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "at", "by", "for", "with", "to",
    "from", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "as", "than", "this", "that", "these", "those",
    "do", "does", "did", "have", "has", "had",
}


# Question-shape phrases that signal the user wants comprehensive output,
# not a focused single-concept answer.
_COMPREHENSIVE_PHRASES = (
    "everything",
    "all you know",
    "all i know",  # same intent, slip
    "complete picture",
    "complete overview",
    "full picture",
    "full overview",
    "everything about",
    "everything you know",
    "everything i know",
    "all about",
    "the whole",
    "comprehensive",
)


def is_comprehensive_intent(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _COMPREHENSIVE_PHRASES)


def content_words(label: str) -> set[str]:
    return {w for w in label.lower().split() if w and w not in _STOP_WORDS}


# A word is "substrate-stop" if it appears in more than this fraction of
# concept labels. In an aptamer brain "rna" hits ~half of all labels and
# would otherwise dominate sibling matches.
_SUBSTRATE_STOP_FRACTION = 0.05


def _substrate_word_freq(conn: sqlite3.Connection) -> dict[str, int]:
    """Count how many distinct labels each content word appears in."""
    freq: dict[str, int] = {}
    for (label,) in conn.execute(
        "SELECT label FROM neurons WHERE label NOT LIKE '%_attribute'"
    ):
        for w in content_words(label):
            freq[w] = freq.get(w, 0) + 1
    return freq


def _substrate_stop_words(conn: sqlite3.Connection) -> set[str]:
    """Words that appear in too many labels to carry topical signal in
    THIS substrate (e.g. 'rna' in an aptamer brain)."""
    freq = _substrate_word_freq(conn)
    n_labels = sum(1 for _ in conn.execute(
        "SELECT 1 FROM neurons WHERE label NOT LIKE '%_attribute'"
    ))
    threshold = max(3, int(n_labels * _SUBSTRATE_STOP_FRACTION))
    return {w for w, c in freq.items() if c >= threshold}


def find_siblings(
    db_path: Path,
    topic: str,
    exclude: set[str] | None = None,
    max_results: int = 12,
) -> list[str]:
    """Return substrate labels (non-attribute) whose content words overlap
    with `topic`. Generic substrate-wide words (e.g. "rna" in an aptamer
    brain) are filtered out so sibling matches anchor on the topical word.

    For "inertia in rna" we want {higher inertia, more inertia, law of
    inertia, inertia} — NOT {acceleration in rna, axis of rna strand, ...}.
    """
    exclude = exclude or set()
    exclude = {e.lower() for e in exclude} | {topic.lower()}

    conn = sqlite3.connect(str(db_path))
    try:
        substrate_stops = _substrate_stop_words(conn)
        topic_topical = content_words(topic) - substrate_stops
        # If every word of the topic is generic, fall back to the full
        # content set so we return SOMETHING rather than nothing.
        match_set = topic_topical or content_words(topic)

        candidates: list[tuple[int, str]] = []
        for (label,) in conn.execute(
            "SELECT label FROM neurons WHERE label NOT LIKE '%_attribute' "
            "AND length(label) BETWEEN 2 AND 80"
        ):
            if label.lower() in exclude:
                continue
            label_words = content_words(label)
            # Match must hit at least one of the topic's topical (rare)
            # words, not just any shared generic substrate word.
            overlap = len(label_words & match_set)
            if overlap == 0:
                continue
            candidates.append((overlap, label))
    finally:
        conn.close()

    candidates.sort(key=lambda t: (-t[0], t[1]))
    return [label for _, label in candidates[:max_results]]


__all__ = [
    "is_comprehensive_intent", "content_words", "find_siblings",
    "_COMPREHENSIVE_PHRASES",
]
