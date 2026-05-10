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


_NP_MODIFIER_DEPS = {
    "compound", "amod", "nmod", "poss", "nummod", "quantmod", "advmod", "det",
    # Gerund subjects ("noticing a limitation") have a dobj child;
    # appositives ("Lee Smith, the captain") have an appos child.
    "dobj", "appos",
}


def _np_phrase(head_token) -> str:
    """Build a noun phrase from a head token: head + left-side modifiers
    (compound/amod/nmod/poss/nummod/det) + any prep+pobj subtree attached
    (catches "K_d for the binding" / "path-of-thought" / "way of life"
    patterns where the PP modifies the head noun).

    Keeps leading determiners ("the"/"a"/"an"), possessives ("its
    target"), and numeric modifiers ("Creed 2"). The trained head and
    Sara substrate downstream want the full surface phrase as the
    canonical neuron label — strip nothing.
    """
    keep_indices: set[int] = {head_token.i}
    for child in head_token.children:
        if child.dep_ in _NP_MODIFIER_DEPS:
            keep_indices.update(t.i for t in child.subtree)
        elif child.dep_ == "prep":
            # Include prep + its pobj subtree as part of the NP. Real
            # paper prose has many "X for/of/on Y" patterns where the
            # PP modifies the head noun ("K_d for the binding").
            keep_indices.update(t.i for t in child.subtree)
    doc = head_token.doc
    parts = [doc[i] for i in sorted(keep_indices)]
    tokens = [t.text for t in parts if not t.is_punct]
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


def _verb_with_particles(verb, *, has_dobj: bool) -> str:
    """Build the relation phrase. Particle vs. oblique-PP heuristic:

    - When the verb has NO direct object, treat `prep`/`prt`/`agent`
      children as part of the relation (e.g. "fold INTO hairpins" —
      relation = "fold into", object = pobj of into).
    - When the verb HAS a direct object, prepositions attached to the
      verb are oblique modifiers (manner / instrument / temporal),
      NOT particles. The relation is the verb alone, lemmatized
      ("predicts kdoff with p<0.05" → relation="predict", object="kdoff").
    """
    parts = [verb]
    if not has_dobj:
        for child in verb.children:
            if child.dep_ in {"prt", "prep", "agent"}:
                parts.append(child)
    else:
        # Even with a dobj, a phrasal-verb particle (`prt`) is still
        # part of the verb (e.g. "pick UP the book"). Keep `prt` only.
        for child in verb.children:
            if child.dep_ == "prt":
                parts.append(child)
    parts.sort(key=lambda t: t.i)
    out = []
    for t in parts:
        if t == verb:
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


_FALLBACK_SUBJECT_POS = {"NOUN", "PROPN", "PRON", "X", "ADJ", "NUM", "INTJ"}


def _find_subject_fallback(verb):
    """Recovery for copular AUX-roots where spaCy mislabeled the subject
    (we've seen `intj` / `dep` / `appos` for unfamiliar tokens like
    `K_d`). If the verb is a copula, look back to the nearest content-
    bearing token before the verb that's a child of `verb` and treat
    it as subject. INTJ is included because spaCy frequently dumps
    unknown content words there.
    """
    if verb.lemma_ not in {"be", "is", "was", "were", "are", "been", "being"}:
        return None
    candidates = [
        c for c in verb.children
        if c.i < verb.i and c.pos_ in _FALLBACK_SUBJECT_POS
        and c.dep_ not in _OBJ_DEPS  # don't grab the attr/dobj as subject
    ]
    if not candidates:
        return None
    # Closest one to the verb wins (rightmost candidate before verb).
    return max(candidates, key=lambda t: t.i)


def _find_objects(verb) -> list:
    """Direct objects + conjuncts. When no direct object, fall back to
    the pobj of a prep child as the object (case: "fold into hairpins").

    When a direct object IS present, prepositions on the verb are
    oblique modifiers (manner / instrument), NOT additional objects.
    Skip them to avoid emitting "X verb with Y" as a separate triple.
    """
    direct_objects = []
    for child in verb.children:
        if child.dep_ in _OBJ_DEPS:
            direct_objects.append(child)
            for cc in child.children:
                if cc.dep_ == _CONJ_DEP:
                    direct_objects.append(cc)
    if direct_objects:
        return direct_objects

    # No direct object — promote pobj of prep to be the object.
    for child in verb.children:
        if child.dep_ == "prep":
            for pc in child.children:
                if pc.dep_ in _PREP_OBJ_DEPS:
                    direct_objects.append(pc)
                    for cc in pc.children:
                        if cc.dep_ == _CONJ_DEP:
                            direct_objects.append(cc)
    return direct_objects


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
        subj_tok = _find_subject(verb) or _find_subject_fallback(verb)
        if subj_tok is None:
            continue

        obj_tokens = _find_objects(verb)
        if not obj_tokens:
            continue
        # Direct objects (dobj/attr/oprd/acomp) determine whether the
        # relation should swallow prep children. attr/acomp/oprd come
        # from copular constructions and count as "direct" for this
        # heuristic.
        has_direct_dobj = any(
            t.dep_ in _OBJ_DEPS for t in obj_tokens
        )
        relation = _verb_with_particles(verb, has_dobj=has_direct_dobj)
        if not relation:
            continue
        subject = _np_phrase(subj_tok)
        if not subject:
            continue

        for obj_tok in obj_tokens:
            obj = _np_phrase(obj_tok)
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
