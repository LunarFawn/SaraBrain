"""Pre-write pollution filters for Sara Brain.

Rejects statements that look like citations, references, author names,
DOIs, nav text, or statements with stopword/pronoun subjects. These are
extraction artifacts — the LLM reading a Wikipedia page misparses the
reference list and produces things like "Jerry (1985) is the author of X."
Those should never reach the graph.

Reuses stopword sets from cortex.cleanup and parser constants rather than
duplicating them.
"""

from __future__ import annotations

import re


# Regex patterns compiled once
_DOI_URL_PATTERN = re.compile(r"(?:doi:|https?://|www\.)", re.IGNORECASE)
_AUTHOR_YEAR_PATTERN = re.compile(r"\b[A-Z][a-z'-]+\s+\(\d{4}\)")
_REF_BRACKET_PATTERN = re.compile(r"\[\d+\]")
_ET_AL_PATTERN = re.compile(r"\bet\s+al\.?", re.IGNORECASE)


def is_polluting_statement(text: str) -> tuple[bool, str]:
    """Check if a statement should be rejected before reaching the graph.

    Returns (rejected, reason). When rejected=True, the caller should
    drop the statement and not attempt to parse or learn it. The reason
    is for logging and debugging.
    """
    s = text.strip()
    if not s:
        return True, "empty"

    # URL / DOI — almost always citation material
    if _DOI_URL_PATTERN.search(s):
        return True, "url_or_doi"

    # Author-year citation: "Smith (1985)", "Jerry (1985)"
    if _AUTHOR_YEAR_PATTERN.search(s):
        return True, "author_year_citation"

    # Reference bracket: "[12]" somewhere in the text
    if _REF_BRACKET_PATTERN.search(s):
        return True, "reference_bracket"

    # "et al" pattern — attribution, not content
    if _ET_AL_PATTERN.search(s):
        return True, "et_al_attribution"

    # Very long — probably a paragraph or section dump, not a fact
    if len(s) > 220:
        return True, "too_long"

    # Too short — single word or partial
    if len(s) < 6:
        return True, "too_short"

    # Subject check — reuse existing pollution sets
    # Parse roughly: subject is text before the first " is/are/has/have/was/were "
    try:
        from ..cortex.cleanup import STOPWORD_SUBJECTS
        from ..parsing.statement_parser import (
            _ARTICLE_FORMS,
            _PRONOUN_SUBJECTS,
        )
    except ImportError:
        STOPWORD_SUBJECTS = frozenset()
        _ARTICLE_FORMS = frozenset()
        _PRONOUN_SUBJECTS = frozenset()

    # Split on common copula verbs to isolate the subject phrase
    subject_phrase = re.split(
        r"\s+(?:is|are|was|were|has|have|had|requires|contains)\s+",
        s,
        maxsplit=1,
    )[0].strip()

    if not subject_phrase:
        return True, "no_subject"

    subject_lower = subject_phrase.lower()

    # Stopword subject: "when is X", "what has Y"
    if subject_lower in STOPWORD_SUBJECTS:
        return True, "stopword_subject"

    # Article-only subject: "the is blue"
    if subject_lower in _ARTICLE_FORMS:
        return True, "article_subject"

    # Pronoun subject: "it is round", "they are fast"
    if subject_lower in _PRONOUN_SUBJECTS:
        return True, "pronoun_subject"

    # Sentence-as-subject: more than 6 words before the copula
    # ("The man walking his dog at noon is happy" — not a fact structure)
    subject_words = subject_phrase.split()
    if len(subject_words) > 6:
        return True, "subject_too_long"

    # Passed all filters
    return False, ""
