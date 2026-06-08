"""Compound-aware query resolution.

Before launching wavefronts, a query's content tokens are resolved
against the graph. Multi-word noun compounds resolve to the compound
neuron if it exists (`nerve cell` → neuron `nerve cell`). When the
compound neuron is absent, the tokens fall back to individual seeds
(bare `nerve` and bare `cell`). This is the mechanism that makes
the wavefront model's "collapse at the compound node" concrete: the
collapse happens at resolution, not during propagation.

Design notes:
- spaCy's `compound` and `amod` dependencies identify noun compounds
  cheaply. The head token is the rightmost NOUN/PROPN in each chain.
- Bare-token fallback preserves coverage when the graph doesn't yet
  have a compound node — the old lemma-overlap behavior as a floor.
- Proper-noun edge cases ("New York", "San Francisco") get the same
  treatment; if a neuron exists by that label, it resolves; else both
  tokens become seeds. No special-casing.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..storage.neuron_repo import NeuronRepo


_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}
_MODIFIER_DEPS = {"compound", "amod", "nmod"}


@dataclass(frozen=True)
class ResolvedSeed:
    """A seed for the wavefront engine.

    `label` is the neuron label (either a compound phrase or a bare
    lemma). `power` is the number of original content tokens this
    seed covers — a compound neuron resolved from two tokens carries
    power 2, a bare lemma carries power 1. The scorer uses power to
    amplify confluence at the seed itself.
    """
    label: str
    power: int
    is_compound: bool


def resolve_query(text: str, nlp, neuron_repo: NeuronRepo
                  ) -> list[ResolvedSeed]:
    """Resolve raw query text to a seed list for `Recognizer.recognize`.

    Order of operations:
      1. Parse with spaCy.
      2. Group contiguous (compound | amod | nmod) modifiers with their
         head noun.
      3. For each group, attempt compound-neuron resolution against
         `neuron_repo`. If found, emit one seed with power == group size.
      4. If compound resolution fails, emit one seed per content token
         in the group.
      5. Non-grouped content tokens emit as single seeds.
    """
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space and not t.is_punct]
    if not tokens:
        return []

    # First pass: for each noun head, walk left to collect its
    # adjacent (compound | amod | nmod) modifier chain. Record which
    # token indices are "consumed" as modifiers of a later head so we
    # don't re-emit them as standalone seeds.
    n = len(tokens)
    consumed: set[int] = set()
    head_spans: dict[int, tuple[int, int]] = {}  # head_idx → (start, end)
    for i, t in enumerate(tokens):
        if t.pos_ not in {"NOUN", "PROPN"}:
            continue
        start = i
        j = i - 1
        while j >= 0:
            prev = tokens[j]
            if (prev.pos_ in {"NOUN", "PROPN", "ADJ"}
                    and prev.dep_ in _MODIFIER_DEPS
                    and prev.head.i == t.i):
                start = j
                consumed.add(j)
                j -= 1
            else:
                break
        head_spans[i] = (start, i)

    # Second pass: emit groups in left-to-right order, skipping tokens
    # that were consumed as modifiers of a later head.
    groups: list[list] = []
    for i, t in enumerate(tokens):
        if i in consumed:
            continue
        if i in head_spans:
            start, end = head_spans[i]
            groups.append(tokens[start:end + 1])
        elif t.pos_ in _CONTENT_POS and not t.is_stop:
            groups.append([t])

    seeds: list[ResolvedSeed] = []
    emitted_labels: set[str] = set()

    for group in groups:
        if len(group) > 1:
            # Try compound resolution (join lemmas by space)
            compound_label = " ".join(
                tok.lemma_.lower().strip() for tok in group
            )
            if not compound_label:
                continue
            if compound_label in emitted_labels:
                continue
            if neuron_repo.resolve(compound_label, exact_only=True) is not None:
                seeds.append(ResolvedSeed(
                    label=compound_label,
                    power=len(group),
                    is_compound=True,
                ))
                emitted_labels.add(compound_label)
                continue
            # Fallback: emit each content token as a bare seed
            for tok in group:
                if tok.pos_ not in _CONTENT_POS:
                    continue
                if tok.is_stop:
                    continue
                lemma = tok.lemma_.lower().strip()
                if not lemma or lemma in emitted_labels:
                    continue
                emitted_labels.add(lemma)
                seeds.append(ResolvedSeed(
                    label=lemma, power=1, is_compound=False,
                ))
        else:
            tok = group[0]
            lemma = tok.lemma_.lower().strip()
            if not lemma or lemma in emitted_labels:
                continue
            emitted_labels.add(lemma)
            seeds.append(ResolvedSeed(
                label=lemma, power=1, is_compound=False,
            ))

    return seeds


# ---- spaCy-free alternative ----

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "tell", "about", "describe", "explain", "define",
    "following", "best", "most", "many", "some", "any", "each", "every",
    "both", "and", "or", "but", "not", "no", "yes", "if", "then", "than",
    "from", "into", "through", "over", "under", "between", "across",
    "after", "before", "during", "while", "since", "until", "because",
})


def resolve_query_nospacy(text: str, neuron_repo: NeuronRepo) -> list[ResolvedSeed]:
    """Resolve query text to seeds WITHOUT spaCy.

    Uses regex tokenization + bigram compound detection + substrate-aware
    filtering. Same output shape as resolve_query() — drop-in replacement.
    """
    import re
    # Capture words 2+ chars, plus single uppercase letters (Roman numerals I, V, X)
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+|[A-Z](?=\s|$|\b)", text)
    lowered = [w.lower() for w in words]

    # Also split hyphenated words ("crossing-over" → "crossing over")
    expanded: list[str] = []
    for w in lowered:
        if "-" in w:
            expanded.extend(w.split("-"))
            # Also try the space-joined form as one candidate
        else:
            expanded.append(w)

    if not expanded:
        return []

    seeds: list[ResolvedSeed] = []
    emitted: set[str] = set()
    subsumed: set[str] = set()

    # First: probe substrate with raw bigrams from expanded tokens
    # (before stopword filtering — catches "prophase i", "crossing over")
    for i in range(len(expanded) - 1):
        bi = f"{expanded[i]} {expanded[i+1]}"
        if bi in emitted:
            continue
        if neuron_repo.resolve(bi, exact_only=True) is not None:
            seeds.append(ResolvedSeed(label=bi, power=2, is_compound=True))
            emitted.add(bi)
            subsumed.add(expanded[i])
            subsumed.add(expanded[i+1])

    # Also try hyphenated words as space-joined ("crossing-over" → "crossing over")
    for w in lowered:
        if "-" in w:
            spaced = w.replace("-", " ")
            if spaced in emitted:
                continue
            if neuron_repo.resolve(spaced, exact_only=True) is not None:
                seeds.append(ResolvedSeed(label=spaced, power=2, is_compound=True))
                emitted.add(spaced)
                for part in spaced.split():
                    subsumed.add(part)

    # Now filter to content words for unigram resolution
    content = [w for w in expanded if w not in _STOPWORDS and len(w) >= 3]

    # Trigrams from content
    for i in range(len(content) - 2):
        tri = f"{content[i]} {content[i+1]} {content[i+2]}"
        if tri in emitted:
            continue
        parts = tri.split()
        if all(p in subsumed for p in parts):
            continue
        if neuron_repo.resolve(tri, exact_only=True) is not None:
            seeds.append(ResolvedSeed(label=tri, power=3, is_compound=True))
            emitted.add(tri)
            for part in parts:
                subsumed.add(part)

    # Bigrams from content
    for i in range(len(content) - 1):
        bi = f"{content[i]} {content[i+1]}"
        if bi in emitted:
            continue
        parts = bi.split()
        if all(p in subsumed for p in parts):
            continue
        if neuron_repo.resolve(bi, exact_only=True) is not None:
            seeds.append(ResolvedSeed(label=bi, power=2, is_compound=True))
            emitted.add(bi)
            for part in parts:
                subsumed.add(part)

    # Unigrams (skip subsumed)
    for w in content:
        if w in subsumed or w in emitted:
            continue
        if neuron_repo.resolve(w, exact_only=True) is not None:
            seeds.append(ResolvedSeed(label=w, power=1, is_compound=False))
            emitted.add(w)

    # Fallback: if nothing resolved, use raw content words
    if not seeds:
        for w in content[:5]:
            if w not in emitted:
                seeds.append(ResolvedSeed(label=w, power=1, is_compound=False))
                emitted.add(w)

    return seeds
