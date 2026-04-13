"""Entity resolver — disambiguate homonyms at ingest and teach time.

When a term like "Sumerian" appears in a fact, it could mean:
- Sumer (the place / geographic region)
- Sumerian people (the ethnic group)
- Sumerian language (the language they spoke)

The entity resolver checks context clues in the surrounding text to
determine which meaning is intended. It then qualifies the neuron
label so each meaning gets its own concept neuron. No shared hub.
No bleed-over. The same architecture as everything else in Sara.

Two use modes:

1. **Ingest-time** — the digester calls `qualify_term()` on each
   extracted fact's subject and object before teaching. If context
   clues indicate a specific meaning, the term gets a qualifier:
   "sumerian" → "sumerian language" or "sumerian people" or "sumer".

2. **Teach-time** — the cortex router calls `find_qualified_variants()`
   when a new fact's subject or object matches existing qualified
   neurons. If variants exist, the disambiguation prompt fires.

The resolver does NOT auto-qualify when context is ambiguous. It
returns the unqualified term and lets the disambiguation prompt
handle it. Never assume. Always ask when unsure.
"""

from __future__ import annotations

from . import grammar


# Entity categories and their qualifier suffixes
ENTITY_CATEGORIES = {
    "place": {"clues": grammar.PLACE_CLUES, "suffix": ""},
    "people": {"clues": grammar.PEOPLE_CLUES, "suffix": " people"},
    "language": {"clues": grammar.LANGUAGE_CLUES, "suffix": " language"},
}


def qualify_term(term: str, context: str) -> str:
    """Qualify a term based on context clues in the surrounding text.

    If the context clearly indicates the term refers to a place, people,
    or language, returns a qualified label. Otherwise returns the original
    term unchanged (the teach-time disambiguation prompt handles the rest).

    Args:
        term: The term to qualify (e.g., "sumerian")
        context: The full sentence or surrounding text

    Returns:
        The qualified label (e.g., "sumerian language") or the original
        term if context is ambiguous.
    """
    term_lower = term.strip().lower()
    context_lower = context.lower()

    # Count context clues for each category
    scores: dict[str, int] = {}
    for category, info in ENTITY_CATEGORIES.items():
        score = 0
        for clue in info["clues"]:
            if " " in clue:
                # Multi-word clue — check as phrase
                if clue in context_lower:
                    score += 2  # phrase matches are stronger
            else:
                # Single-word clue — check as word boundary
                # Use simple word splitting to avoid regex overhead
                if clue in context_lower.split():
                    score += 1
        scores[category] = score

    # Only qualify if one category clearly wins (>= 2 points ahead)
    if not any(scores.values()):
        return term  # no clues found — return unqualified

    best = max(scores, key=lambda k: scores[k])
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

    if scores[best] >= 2 and scores[best] - second >= 2:
        # Clear winner — qualify the term
        suffix = ENTITY_CATEGORIES[best]["suffix"]
        if suffix:
            return f"{term_lower}{suffix}"
        else:
            # Place category has no suffix — use the base term
            # (strip "ian"/"ians" to get the place name if applicable)
            if term_lower.endswith("ians"):
                return term_lower[:-4]
            elif term_lower.endswith("ian"):
                return term_lower[:-3]
            return term_lower

    # Ambiguous — return unqualified, let teach-time prompt handle it
    return term


def find_qualified_variants(term: str, brain) -> list[str]:
    """Find existing neurons that are qualified variants of a term.

    Searches for neurons whose label starts with the term and includes
    a qualifier suffix (e.g., "sumerian" finds "sumerian language",
    "sumerian people").

    Args:
        term: The base term to search for
        brain: The Brain instance to search in

    Returns:
        List of qualified neuron labels found in the brain.
    """
    term_lower = term.strip().lower()
    variants = []

    # Check for exact match (unqualified)
    if brain.neuron_repo.get_by_label(term_lower) is not None:
        variants.append(term_lower)

    # Check for place form (strip -ian/-ians suffix)
    place_form = term_lower
    if term_lower.endswith("ians"):
        place_form = term_lower[:-4]
    elif term_lower.endswith("ian"):
        place_form = term_lower[:-3]
    if place_form != term_lower:
        if brain.neuron_repo.get_by_label(place_form) is not None:
            variants.append(place_form)

    # Check for qualified forms
    for category, info in ENTITY_CATEGORIES.items():
        suffix = info["suffix"]
        if suffix:
            qualified = f"{term_lower}{suffix}"
            if brain.neuron_repo.get_by_label(qualified) is not None:
                variants.append(qualified)

    return variants


def format_disambiguation_prompt(term: str, variants: list[str], brain) -> str:
    """Format a disambiguation prompt showing all qualified variants.

    Returns a user-facing string asking which meaning is intended.
    """
    lines = [f"Ambiguity in {term!r}:"]
    lines.append(f"  Sara has multiple concepts with this name:")
    for i, v in enumerate(variants, 1):
        # Get a brief description from the first path
        n = brain.neuron_repo.get_by_label(v)
        desc = ""
        if n:
            paths = brain.path_repo.get_paths_to(n.id)
            for p in paths[:1]:
                if p.source_text:
                    desc = f" — {p.source_text[:60]}"
                    break
        lines.append(f"    {i}. {v}{desc}")
    lines.append(f"  → Which do you mean? Type the number, or 'new' for a new concept.")
    return "\n".join(lines)
