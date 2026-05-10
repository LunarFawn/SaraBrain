"""Rule-based (subject, relation, object) extractor over spaCy dependency parse.

Stub extractor for the book-ingest pipeline. Used until the neural
extraction head is trained. Same protocol both implementations follow:
input is a single clause string + a spaCy-loaded `nlp` object, output is
zero or more (subject, relation, object) triples.

Compound subjects ("RNA aptamers"), prepositional verbs ("fold into",
"binds to"), and conjoined objects ("hairpins and ligands") are handled.
Sentences with no clean s/r/o (questions, fragments, declaratives without
a verb) emit zero triples — silence beats noise per auto-commit semantics.
"""
from __future__ import annotations

from dataclasses import dataclass

# spaCy types are duck-typed; we don't import them here to keep the
# module importable without spaCy installed. The `nlp` callable is
# passed in by the caller.


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_clause: str


_AUX_DEPS = {"aux", "auxpass"}
_NEG_DEPS = {"neg"}
_SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
_OBJ_DEPS = {"dobj", "attr", "oprd", "acomp"}
_PREP_OBJ_DEPS = {"pobj"}
_CONJ_DEP = "conj"
_VERB_LIKE_POS = {"VERB", "AUX"}
_VERB_LIKE_ROOT_DEPS = {"ROOT", "ccomp", "xcomp", "advcl", "relcl"}


def _subtree_text(token) -> str:
    """Concatenate the token's full subtree in source order, lowercased,
    punctuation stripped at the edges."""
    spans = sorted([t for t in token.subtree], key=lambda t: t.i)
    text = " ".join(t.text for t in spans if not t.is_punct)
    return _normalize(text)


_NP_MODIFIER_DEPS = {"compound", "amod", "nmod", "poss", "nummod", "quantmod", "advmod"}
_NP_LEADING_DETS = {"the", "a", "an"}


def _np_phrase(head_token) -> str:
    """Build a noun phrase from a head token: head + left-side modifiers
    (compound/amod/nmod/poss/nummod) + any prep+pobj subtree attached
    (catches hyphen-of constructions like "path-of-thought").

    Strips leading "a"/"an"/"the". Keeps possessives ("its target")
    because they are semantically meaningful.
    """
    keep_indices: set[int] = {head_token.i}
    for child in head_token.children:
        if child.dep_ in _NP_MODIFIER_DEPS:
            keep_indices.update(t.i for t in child.subtree)
        elif child.dep_ == "prep":
            # Include prep + its pobj subtree only if the prep sits
            # tightly adjacent to the head (covers "path-of-thought" /
            # "way of life" patterns), but skip wide-ranging prep PPs
            # that introduce new arguments.
            if abs(child.i - head_token.i) <= 2:
                keep_indices.update(t.i for t in child.subtree)
    doc = head_token.doc
    parts = [doc[i] for i in sorted(keep_indices)]
    tokens = [t.text for t in parts if not t.is_punct]
    while tokens and tokens[0].lower() in _NP_LEADING_DETS:
        tokens.pop(0)
    return _normalize(" ".join(tokens))


# Back-compat alias for internal callers.
_compound_phrase = _np_phrase


def _normalize(s: str) -> str:
    s = " ".join(s.lower().split())
    s = s.strip(" ,.;:!?\"'‘’“”")
    s = s.replace(" - ", "-")
    s = s.replace(" 's", "'s")
    s = s.replace(" n't", "n't")
    return s


def _verb_with_particles(verb) -> str:
    """Build the relation phrase: verb + prt/prep/agent particles attached."""
    parts = [verb]
    for child in verb.children:
        if child.dep_ in {"prt", "prep", "agent"}:
            parts.append(child)
    parts.sort(key=lambda t: t.i)
    out = []
    for t in parts:
        if t == verb:
            # Use lemma if spaCy gave us one; fall back to surface for mistags.
            lemma = t.lemma_.lower() if t.lemma_ and t.lemma_ != "-PRON-" else t.text.lower()
            out.append(lemma)
        else:
            out.append(t.text.lower())
    return _normalize(" ".join(out))


def _find_subject(verb):
    for child in verb.children:
        if child.dep_ in _SUBJ_DEPS:
            return child
    return None


def _find_objects(verb) -> list:
    """Direct objects, attribute objects, and prepositional objects.

    For prepositional objects, returns the pobj of any prep child of the
    verb. Conjoined objects ("hairpins and ligands") are expanded.
    """
    objects = []
    for child in verb.children:
        if child.dep_ in _OBJ_DEPS:
            objects.append(child)
            for cc in child.children:
                if cc.dep_ == _CONJ_DEP:
                    objects.append(cc)
        elif child.dep_ == "prep":
            for pc in child.children:
                if pc.dep_ in _PREP_OBJ_DEPS:
                    objects.append(pc)
                    for cc in pc.children:
                        if cc.dep_ == _CONJ_DEP:
                            objects.append(cc)
    return objects


def _is_negated(verb) -> bool:
    return any(c.dep_ in _NEG_DEPS for c in verb.children)


def extract_triples(clause: str, nlp) -> list[Triple]:
    """Run rule-based extraction on a single clause.

    Returns zero or more triples. Empty list when no clean s/r/o is
    found (questions, fragments, copular fragments, etc.).
    """
    if not clause or not clause.strip():
        return []
    doc = nlp(clause)

    # Verb-like predicates: any token tagged as VERB/AUX, plus any ROOT
    # token that has a subject child (catches mistags where spaCy labels
    # an inflected verb as NOUN — e.g. "Creed 2 builds on Creed 1").
    verbs = []
    for t in doc:
        if t.pos_ in _VERB_LIKE_POS and t.dep_ in _VERB_LIKE_ROOT_DEPS:
            verbs.append(t)
        elif t.dep_ == "ROOT" and any(c.dep_ in _SUBJ_DEPS for c in t.children):
            if t not in verbs:
                verbs.append(t)
    if not verbs:
        return []

    triples: list[Triple] = []
    for verb in verbs:
        if _is_negated(verb):
            # Negated facts are out of scope for v1 auto-commit.
            continue
        subj_tok = _find_subject(verb)
        if subj_tok is None:
            continue
        relation = _verb_with_particles(verb)
        if not relation:
            continue
        subject = _compound_phrase(subj_tok)
        if not subject:
            continue

        obj_tokens = _find_objects(verb)
        if not obj_tokens:
            # Copular case: VERB="be" with attr child carrying object
            attrs = [c for c in verb.children if c.dep_ in {"attr", "acomp", "oprd"}]
            if attrs:
                obj_tokens = attrs
            else:
                continue

        for obj_tok in obj_tokens:
            obj = _compound_phrase(obj_tok)
            if not obj:
                continue
            # Skip pronouns as objects — no useful triple ("X bound it")
            if obj_tok.pos_ == "PRON":
                continue
            triples.append(Triple(
                subject=subject,
                relation=relation,
                object=obj,
                source_clause=clause.strip(),
            ))

    return triples


__all__ = ["Triple", "extract_triples"]
