"""Deep-dive helpers for HamlinLLM chat.

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


def find_siblings(
    db_path: Path,
    topic: str,
    exclude: set[str] | None = None,
    max_results: int = 12,
) -> list[str]:
    """Return substrate labels (non-attribute) whose content words
    overlap with `topic`. Excludes the topic itself (and any labels in
    `exclude`). Sorted by overlap size, then alphabetically."""
    topic_words = content_words(topic)
    if not topic_words:
        return []
    exclude = exclude or set()
    exclude = {e.lower() for e in exclude} | {topic.lower()}

    conn = sqlite3.connect(str(db_path))
    candidates: list[tuple[int, str]] = []
    # Cheap but effective: pull all non-attribute concept labels; intersect
    # in Python. Substrates are small enough (~few thousand neurons).
    for (label,) in conn.execute(
        "SELECT label FROM neurons WHERE label NOT LIKE '%_attribute' "
        "AND length(label) BETWEEN 2 AND 80"
    ):
        if label.lower() in exclude:
            continue
        words = content_words(label)
        overlap = len(words & topic_words)
        if overlap == 0:
            continue
        candidates.append((overlap, label))
    conn.close()

    candidates.sort(key=lambda t: (-t[0], t[1]))
    return [label for _, label in candidates[:max_results]]


__all__ = [
    "is_comprehensive_intent", "content_words", "find_siblings",
    "_COMPREHENSIVE_PHRASES",
]
