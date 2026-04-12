"""Brain cleanup utility — find and refute pollution caused by past parser bugs.

This tool inspects an existing brain database for neurons that look like
pollution rather than real concepts. Pollution comes in three forms:

1. **Article-typo neurons** — neurons whose label is in `_ARTICLE_FORMS`
   ("teh", "tteh", "thte"). These were created by old parser versions
   that didn't strip article typos. Always safe to refute.

2. **Pronoun neurons** — neurons whose label is a pronoun ("it", "they",
   "this"). These were created when pronouns were accepted as subjects.
   Always safe to refute.

3. **Suspected typo neurons** — neurons whose label is within edit
   distance 1-2 of a much-better-connected neuron. The well-connected
   neuron is the canonical form; the close-matching one is likely a
   typo of it. **Never auto-merge these — always ask the user.**

Sara never deletes. The cleanup tool refutes the pollution paths so they
get marked known-to-be-false but their history is preserved. A user can
later look at what was refuted and why.

Usage:
    sara-cleanup --db /path/to/brain.db                # interactive review
    sara-cleanup --db /path/to/brain.db --auto-articles  # auto-refute article typos only
    sara-cleanup --db /path/to/brain.db --dry-run        # show what would happen, do nothing
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ..config import default_db_path
from ..core.brain import Brain
from ..parsing.statement_parser import _ARTICLE_FORMS, _PRONOUN_SUBJECTS
from ..storage.neuron_repo import NeuronRepo


@dataclass
class PollutionCandidate:
    """A neuron that looks like it might be pollution."""
    neuron_id: int
    label: str
    kind: str            # "article_typo", "pronoun", "suspected_typo"
    path_count: int
    canonical: str | None = None  # for suspected_typo: the likely correct form
    canonical_path_count: int = 0
    edit_distance: int = 0


def find_article_typo_neurons(brain: Brain) -> list[PollutionCandidate]:
    """Find neurons whose label is a known article-typo form."""
    candidates = []
    for n in brain.neuron_repo.list_all():
        if n.label.strip() in _ARTICLE_FORMS:
            paths_to = brain.path_repo.get_paths_to(n.id)
            paths_from = brain.path_repo.get_paths_from(n.id)
            candidates.append(PollutionCandidate(
                neuron_id=n.id,
                label=n.label,
                kind="article_typo",
                path_count=len(paths_to) + len(paths_from),
            ))
    return sorted(candidates, key=lambda c: -c.path_count)


def find_pronoun_neurons(brain: Brain) -> list[PollutionCandidate]:
    """Find neurons whose label is a pronoun."""
    candidates = []
    for n in brain.neuron_repo.list_all():
        if n.label.strip() in _PRONOUN_SUBJECTS:
            paths_to = brain.path_repo.get_paths_to(n.id)
            paths_from = brain.path_repo.get_paths_from(n.id)
            candidates.append(PollutionCandidate(
                neuron_id=n.id,
                label=n.label,
                kind="pronoun",
                path_count=len(paths_to) + len(paths_from),
            ))
    return sorted(candidates, key=lambda c: -c.path_count)


# Common question-word typos. NEVER stripped automatically — listed
# only so the cleanup tool can present them to the user for review.
_QUESTION_WORD_TYPOS = frozenset({
    "waht", "hwat", "whta", "wht",
    "wher", "wehre", "whree",
    "wehn", "whne",
    "hwy", "yhw",
    "hwo", "owh",
    "wich",
    "thi", "ths", "tihs",     # singularized "this"
    "thse", "tehse",
    "thi_attribute",          # leftover from old singularize bugs
})

# Common stopwords that should never be standalone subject neurons.
# These got into the brain when old parsers treated them as content.
_STOPWORD_SUBJECTS = frozenset({
    "not", "and", "or", "but", "if", "then", "than",
    "for", "with", "from", "by", "as", "at", "in", "on", "of", "to",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "can", "could", "would", "should", "may", "might", "must",
    "very", "just", "only", "also", "even", "still",
    "more", "most", "less", "least", "much",
    "some", "any", "all", "each", "every", "no",
    "here", "there", "where", "when", "why", "how",
    "yes", "no", "ok",
    "type", "kind", "sort",
    "thing", "stuff",
    "like", "such",
})


def find_question_word_typos(brain: Brain) -> list[PollutionCandidate]:
    """Find neurons whose label is a known typo of a question word."""
    candidates = []
    for n in brain.neuron_repo.list_all():
        label = n.label.strip().lower()
        if label in _QUESTION_WORD_TYPOS:
            paths_to = brain.path_repo.get_paths_to(n.id)
            paths_from = brain.path_repo.get_paths_from(n.id)
            candidates.append(PollutionCandidate(
                neuron_id=n.id,
                label=n.label,
                kind="question_word_typo",
                path_count=len(paths_to) + len(paths_from),
            ))
    return sorted(candidates, key=lambda c: -c.path_count)


def find_stopword_subject_neurons(brain: Brain) -> list[PollutionCandidate]:
    """Find neurons whose label is a bare stopword.

    Words like 'not', 'it', 'building', 'type' should never be standalone
    subjects — they came from old parsers that treated them as content.
    """
    candidates = []
    for n in brain.neuron_repo.list_all():
        label = n.label.strip().lower()
        if label in _STOPWORD_SUBJECTS:
            paths_to = brain.path_repo.get_paths_to(n.id)
            paths_from = brain.path_repo.get_paths_from(n.id)
            candidates.append(PollutionCandidate(
                neuron_id=n.id,
                label=n.label,
                kind="stopword_subject",
                path_count=len(paths_to) + len(paths_from),
            ))
    return sorted(candidates, key=lambda c: -c.path_count)


def find_sentence_subject_neurons(
    brain: Brain,
    min_words: int = 7,
) -> list[PollutionCandidate]:
    """Find neurons whose label is an entire sentence — almost always
    a digester or auto-teach failure where the LLM swallowed a full
    statement and gave it back as a single subject.

    Heuristic: any neuron whose label has more than `min_words` words.
    """
    candidates = []
    for n in brain.neuron_repo.list_all():
        label = n.label.strip()
        word_count = len(label.split())
        if word_count >= min_words:
            paths_to = brain.path_repo.get_paths_to(n.id)
            paths_from = brain.path_repo.get_paths_from(n.id)
            candidates.append(PollutionCandidate(
                neuron_id=n.id,
                label=n.label,
                kind="sentence_subject",
                path_count=len(paths_to) + len(paths_from),
            ))
    return sorted(candidates, key=lambda c: -len(c.label))


def find_punctuation_artifact_neurons(brain: Brain) -> list[PollutionCandidate]:
    """Find neurons whose label ends in punctuation — sentence fragments
    that got captured as subjects (e.g., 'agriculture:', 'school.', 'too').
    """
    candidates = []
    for n in brain.neuron_repo.list_all():
        label = n.label.strip()
        if label and label[-1] in ".,:;!?":
            paths_to = brain.path_repo.get_paths_to(n.id)
            paths_from = brain.path_repo.get_paths_from(n.id)
            candidates.append(PollutionCandidate(
                neuron_id=n.id,
                label=n.label,
                kind="punctuation_artifact",
                path_count=len(paths_to) + len(paths_from),
            ))
    return sorted(candidates, key=lambda c: -c.path_count)


def find_suspected_typo_neurons(
    brain: Brain,
    min_canonical_paths: int = 5,
    max_typo_paths: int = 3,
) -> list[PollutionCandidate]:
    """Find neurons that are likely typos of better-established neurons.

    Heuristic:
    - The candidate must have FEW paths (max_typo_paths or less)
    - There must be another neuron within edit distance 2
    - That other neuron must have MANY paths (min_canonical_paths or more)
    - The candidate must be longer than 3 characters

    This is conservative on purpose. We want to catch "choldren" → "children"
    but NOT catch "metoprolol" → "metformin" because the latter would
    typically have many paths in any real medication knowledge base.
    """
    repo = brain.neuron_repo
    all_neurons = repo.list_all()

    # Build a map of neuron → path count
    path_counts: dict[int, int] = {}
    for n in all_neurons:
        if n.id is None:
            continue
        path_counts[n.id] = (
            len(brain.path_repo.get_paths_to(n.id))
            + len(brain.path_repo.get_paths_from(n.id))
        )

    candidates = []
    for n in all_neurons:
        if n.id is None or len(n.label) <= 3:
            continue
        if path_counts.get(n.id, 0) > max_typo_paths:
            continue  # too well-connected to be a typo

        # Find well-connected neurons within edit distance 2
        for other in all_neurons:
            if other.id is None or other.id == n.id:
                continue
            if path_counts.get(other.id, 0) < min_canonical_paths:
                continue
            d = NeuronRepo._edit_distance(n.label, other.label, 2)
            if 0 < d <= 2:
                candidates.append(PollutionCandidate(
                    neuron_id=n.id,
                    label=n.label,
                    kind="suspected_typo",
                    path_count=path_counts.get(n.id, 0),
                    canonical=other.label,
                    canonical_path_count=path_counts.get(other.id, 0),
                    edit_distance=d,
                ))
                break  # only report once per candidate

    return sorted(candidates, key=lambda c: (c.edit_distance, -c.canonical_path_count))


def refute_neuron_paths(brain: Brain, candidate: PollutionCandidate) -> int:
    """Refute all paths involving a pollution candidate.

    Sara never deletes — the paths stay with [refuted] prefix and
    negative strength. The neuron itself stays in the graph too.
    Returns the number of paths refuted.
    """
    refuted = 0
    paths_to = brain.path_repo.get_paths_to(candidate.neuron_id)
    paths_from = brain.path_repo.get_paths_from(candidate.neuron_id)
    for p in list(paths_to) + list(paths_from):
        if p.source_text and not p.source_text.startswith("["):
            # Reconstruct a positive form from the source_text and refute it
            try:
                result = brain.refute(p.source_text)
                if result is not None:
                    refuted += 1
            except Exception:
                pass
    if refuted > 0:
        brain.conn.commit()
    return refuted


def _review_category(
    brain: Brain,
    candidates: list[PollutionCandidate],
    label: str,
    explanation: str,
) -> None:
    """Walk through a category of pollution candidates with per-item review.

    Sara never bulk-refutes. Each item requires explicit user choice.
    """
    if not candidates:
        return
    print()
    print(f"  {label} review (each requires explicit confirmation):")
    print(f"  {explanation}")
    for c in candidates[:50]:
        print()
        print(f"    Candidate: {c.label!r} ({c.path_count} paths)")
        choice = input("    [r]efute paths / [k]eep / [s]kip / [q]uit category: ").strip().lower()
        if choice == "q":
            print("    Quitting this category.")
            break
        if choice == "r":
            refuted = refute_neuron_paths(brain, c)
            print(f"    Refuted {refuted} path(s).")
        elif choice == "k":
            print("    Kept.")
        else:
            print("    Skipped.")


def _print_candidates(label: str, candidates: list[PollutionCandidate]) -> None:
    if not candidates:
        print(f"  No {label} candidates found.")
        return
    print(f"\n  {len(candidates)} {label} candidate(s):")
    for c in candidates[:30]:
        if c.canonical:
            print(
                f"    [{c.path_count:3d} paths] {c.label!r}  → "
                f"likely typo of {c.canonical!r} "
                f"({c.canonical_path_count} paths, edit distance {c.edit_distance})"
            )
        else:
            print(f"    [{c.path_count:3d} paths] {c.label!r}")
    if len(candidates) > 30:
        print(f"    ... and {len(candidates) - 30} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sara-cleanup",
        description="Find and refute pollution in Sara's brain. "
                    "Never deletes — refuted paths stay with negative "
                    "strength as evidence of past parser mistakes.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Brain database path (default: {default_db_path()})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be refuted without making any changes.",
    )
    # Note: there are NO --auto flags. Every refutation is per-item
    # interactive. What looks like a typo in English may be a valid
    # word in Haitian Creole, Jamaican Patois, AAVE, or other dialects.
    # Sara has no authority to silently erase a user's language.
    parser.add_argument(
        "--show-typos",
        action="store_true",
        help="Also scan for suspected content-word typos. NEVER auto-refuted "
             "— always shown for manual review.",
    )
    args = parser.parse_args()

    db_path = args.db or default_db_path()
    brain = Brain(db_path)

    print()
    print("  Sara Brain Cleanup")
    print(f"  Database: {db_path}")
    s = brain.stats()
    print(f"  Brain: {s['neurons']} neurons, {s['paths']} paths")
    print()

    article_typos = find_article_typo_neurons(brain)
    pronouns = find_pronoun_neurons(brain)
    question_typos = find_question_word_typos(brain)
    stopwords = find_stopword_subject_neurons(brain)
    sentences = find_sentence_subject_neurons(brain)
    punct = find_punctuation_artifact_neurons(brain)

    _print_candidates("article-typo", article_typos)
    _print_candidates("pronoun", pronouns)
    _print_candidates("question-word typo (waht/hwat/etc)", question_typos)
    _print_candidates("stopword-subject (not/it/like/etc)", stopwords)
    _print_candidates("sentence-subject (long phrases)", sentences)
    _print_candidates("punctuation-artifact (label ends in punct)", punct)

    typos: list[PollutionCandidate] = []
    if args.show_typos:
        print()
        print("  Scanning for suspected content-word typos (slow)...")
        typos = find_suspected_typo_neurons(brain)
        _print_candidates("suspected typo", typos)

    if args.dry_run:
        print()
        print("  Dry run — no changes made.")
        brain.close()
        return

    # Per-category interactive review. Sara never bulk-refutes.
    # A user's "tteh" may be intentional in their dialect.
    _review_category(
        brain, article_typos, "article-typo",
        "This may be a typo of an English article, OR it may be a real word in your dialect.",
    )

    _review_category(
        brain, pronouns, "pronoun-subject",
        "Pronouns can never be standalone subjects — these are old parser bugs.",
    )
    _review_category(
        brain, question_typos, "question-word typo",
        "These are typos of question words (waht/hwat) that became subjects.",
    )
    _review_category(
        brain, stopwords, "stopword-subject",
        "Bare stopwords (not/it/like/type) should never be standalone subjects.",
    )
    _review_category(
        brain, sentences, "sentence-subject",
        "Long phrases that got captured as a single neuron — usually digester errors.",
    )
    _review_category(
        brain, punct, "punctuation-artifact",
        "Sentence fragments with trailing punctuation that became subjects.",
    )

    # Suspected content typos: extra-careful review
    if typos:
        print()
        print("  Suspected typo review (each requires explicit confirmation):")
        print("  Sara will NEVER auto-merge — confirming refutes the typo's paths,")
        print("  but the canonical version stays as the source of truth.")
        for c in typos[:20]:
            print()
            print(f"    Candidate: {c.label!r} ({c.path_count} paths)")
            print(f"    Likely canonical: {c.canonical!r} ({c.canonical_path_count} paths)")
            print(f"    Edit distance: {c.edit_distance}")
            choice = input("    [r]efute typo paths / [k]eep both / [s]kip / [q]uit: ").strip().lower()
            if choice == "q":
                break
            if choice == "r":
                refuted = refute_neuron_paths(brain, c)
                print(f"    Refuted {refuted} path(s).")
            elif choice == "k":
                print("    Kept. Both neurons remain.")
            else:
                print("    Skipped.")

    print()
    s = brain.stats()
    print(f"  Final state: {s['neurons']} neurons, {s['paths']} paths")
    brain.close()


def _confirm(prompt: str) -> bool:
    answer = input(f"  {prompt} [y/N] ").strip().lower()
    return answer in ("y", "yes")


if __name__ == "__main__":
    main()
