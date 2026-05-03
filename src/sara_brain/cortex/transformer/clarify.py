"""Clarification helpers for HamRobyLLM chat.

Detects two kinds of probable typos and asks the user before assuming:

1. Wh-word typos at the start of a question ("waht is X" -> "what is X")
2. Substrate-concept typos ("ssn1" when only "ssng1" exists)

Both produce a Clarification with candidate options. The chat REPL
presents them numerically; the user picks or types a new question.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches

from sara_brain.core.brain import Brain


# Vocabulary the leading wh-token of a question is *probably* meant to be.
WH_VOCAB = [
    "what", "what's", "who", "who's", "whose", "whom",
    "where", "when", "why", "how", "which",
    "tell", "show", "give", "explain", "define", "describe", "list", "find",
    "do", "does", "did", "is", "are", "was", "were", "can", "could",
]


@dataclass
class WhFix:
    original: str               # what the user typed
    candidates: list[str]       # close matches from WH_VOCAB
    replace_at_index: int = 0   # token index in the question to replace


@dataclass
class ConceptFix:
    original: str               # what the router extracted
    candidates: list[dict]      # substrate did_you_mean output (label, type, distance)
    field: str                  # which arg to replace ("concept" / "label" / "term")


@dataclass
class Clarification:
    """A pending question the user needs to answer before HamRobyLLM can
    proceed. Holds the original question and either a wh-fix or a
    concept-fix to apply once the user picks an option."""
    original_question: str
    wh_fix: WhFix | None = None
    concept_fix: ConceptFix | None = None
    pending_router_decision: dict | None = field(default=None)
    # ^ when set, we've already routed but the chosen concept is unknown;
    #   applying the fix means re-issuing the same tool with a new arg.

    def render_prompt(self) -> str:
        if self.wh_fix:
            opts = "\n".join(f"  {i+1}. {c}"
                             for i, c in enumerate(self.wh_fix.candidates))
            return (f'I see "{self.wh_fix.original}" — did you mean:\n'
                    f"{opts}\n"
                    f"  (or type a new question, or 'no' to cancel)")
        if self.concept_fix:
            cf = self.concept_fix
            opts_lines = []
            for i, c in enumerate(cf.candidates):
                desc = f" — {c['description']}" if c.get("description") else ""
                opts_lines.append(f"  {i+1}. {c['label']}{desc}")
            opts = "\n".join(opts_lines)
            return (f'No exact match for "{cf.original}". Did you mean:\n'
                    f"{opts}\n"
                    f"  (or type a new question, or 'no' to cancel)")
        return "(no clarification pending)"


def detect_wh_typo(question: str) -> WhFix | None:
    """If the first content token of `question` looks like a misspelled
    wh-word, return a WhFix with candidates. None if the leading token
    is already a known wh/start word or is too dissimilar to suggest."""
    tokens = question.strip().split()
    if not tokens:
        return None
    first = tokens[0].lower().strip(",.;:!?\"'")
    if first in WH_VOCAB:
        return None
    cand = get_close_matches(first, WH_VOCAB, n=3, cutoff=0.7)
    if not cand:
        return None
    return WhFix(original=tokens[0], candidates=cand, replace_at_index=0)


def apply_wh_fix(question: str, fix: WhFix, choice: str) -> str:
    tokens = question.strip().split()
    tokens[fix.replace_at_index] = choice
    return " ".join(tokens)


def find_concept_candidates(brain: Brain, term: str, max_n: int = 5) -> list[dict]:
    """Return the substrate's did-you-mean candidates for a missing concept."""
    try:
        candidates = brain.did_you_mean(term)
    except Exception:
        return []
    # Prefer close edit-distance matches first.
    candidates = sorted(candidates, key=lambda c: c.get("distance", 99))
    return candidates[:max_n]


def parse_choice(line: str, n_options: int) -> int | None:
    """Parse '1', '2', '#1', 'option 1', etc. Returns 0-based index or
    None if the line isn't a numeric choice."""
    s = line.strip().lower()
    s = s.lstrip("#")
    s = s.removeprefix("option").strip()
    if s.isdigit():
        i = int(s) - 1
        if 0 <= i < n_options:
            return i
    return None


def is_cancel(line: str) -> bool:
    return line.strip().lower() in {"no", "n", "cancel", "nope", "nvm", "never mind"}


__all__ = [
    "WH_VOCAB", "WhFix", "ConceptFix", "Clarification",
    "detect_wh_typo", "apply_wh_fix",
    "find_concept_candidates", "parse_choice", "is_cancel",
]
