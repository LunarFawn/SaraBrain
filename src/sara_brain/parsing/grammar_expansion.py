"""Grammar expansion — decompose a rich source sentence into multiple
clean SVO sub-facts using spaCy dep parse.

One source sentence yields one or more ParsedStatement objects:
  - Primary SVO with bare head-noun atoms as subject/object
  - Adjective modifiers: `<noun> has_property <adj>`
  - Prep-phrase modifiers: `<noun> <prep> <prep_obj_head>`
  - Adverb modifiers: `<verb> has_manner <adv>`
  - Subordinate clauses: recursively extracted as their own SVO

Atoms are token lemmas (head nouns only). Modifier relations may be
novel verbs (prepositions, "has_property", "has_manner"); callers
should register them via `brain.teach_verb()` before teaching the
expansion.
"""
from __future__ import annotations

from .statement_parser import ParsedStatement


_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "NUM"}


def _lemma(tok) -> str:
    return tok.lemma_.lower().strip()


def _head_noun(tok):
    """Return the head noun token of a noun-phrase subtree. Walks down
    compound/amod/nmod modifiers to the final NOUN/PROPN."""
    if tok.pos_ not in {"NOUN", "PROPN", "PRON"}:
        return tok
    return tok  # spaCy already gives us the head; modifiers are children


def _adjective_children(noun_tok):
    for child in noun_tok.children:
        if child.pos_ == "ADJ" and child.dep_ in {"amod", "acomp"}:
            yield child


def _prep_phrase_children(tok):
    """Yield (prep_word, prep_obj_head_tok) for each prep modifier."""
    for child in tok.children:
        if child.dep_ == "prep":
            for pobj in child.children:
                if pobj.dep_ == "pobj" and pobj.pos_ in {
                    "NOUN", "PROPN", "PRON"
                }:
                    yield child.text.lower(), pobj


def _adverb_children(verb_tok):
    for child in verb_tok.children:
        if child.pos_ == "ADV" and child.dep_ == "advmod":
            yield child


def _subordinate_clauses(tok):
    """Yield subordinate verb tokens (relative clauses, clausal
    modifiers)."""
    for child in tok.children:
        if child.dep_ in {"relcl", "ccomp", "xcomp", "advcl", "acl"}:
            if child.pos_ in {"VERB", "AUX"}:
                yield child


def _nsubjs(verb):
    for child in verb.children:
        if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {
            "NOUN", "PROPN", "PRON"
        }:
            yield child


def _dobjs(verb):
    """Direct objects + attributes + complements."""
    for child in verb.children:
        if child.dep_ in {"dobj", "attr", "acomp", "oprd"}:
            if child.pos_ in {"NOUN", "PROPN", "ADJ"}:
                yield child


def _make_stmt(subject: str, relation: str, obj: str,
               original: str, negated: bool = False) -> ParsedStatement:
    return ParsedStatement(
        subject=subject,
        relation=relation,
        obj=obj,
        original=original,
        negated=negated,
    )


def expand_statement(sentence: str, nlp,
                     include_modifiers: bool = True,
                     ) -> list[ParsedStatement]:
    """Return every SVO sub-fact spaCy can pull from `sentence`.

    Modifier facts (adjectives, prep phrases, adverbs) are emitted in
    addition to the primary SVO when `include_modifiers=True`.
    """
    doc = nlp(sentence)
    out: list[ParsedStatement] = []
    seen = set()

    def add(subj: str, rel: str, obj: str, negated: bool = False) -> None:
        if not (subj and rel and obj):
            return
        key = (subj.lower(), rel.lower(), obj.lower())
        if key in seen:
            return
        seen.add(key)
        out.append(_make_stmt(subj, rel, obj, sentence, negated))

    for sent in doc.sents:
        for verb in sent:
            if verb.pos_ not in {"VERB", "AUX"}:
                continue
            subjs = list(_nsubjs(verb))
            if not subjs:
                continue
            objs = list(_dobjs(verb))
            verb_lemma = _lemma(verb)
            negated = any(
                c.dep_ == "neg" for c in verb.children
            )

            for subj in subjs:
                subj_atom = _lemma(subj)

                # Primary SVO: subject + verb + each object
                for obj in objs:
                    obj_atom = _lemma(obj)
                    add(subj_atom, verb_lemma, obj_atom, negated=negated)

                # Prep-phrase objects on the verb: subject + verb +
                # prep_obj_head, relation = verb_prep
                for prep, pobj in _prep_phrase_children(verb):
                    pobj_atom = _lemma(pobj)
                    # Use the preposition itself as a relation word so
                    # the atom relationship is preserved without
                    # inventing a compound verb.
                    add(subj_atom, prep, pobj_atom)

                if not include_modifiers:
                    continue

                # Subject modifiers
                for adj in _adjective_children(subj):
                    add(subj_atom, "is", _lemma(adj))
                for prep, pobj in _prep_phrase_children(subj):
                    add(subj_atom, prep, _lemma(pobj))

            # Object modifiers
            if include_modifiers:
                for obj in objs:
                    obj_atom = _lemma(obj)
                    for adj in _adjective_children(obj):
                        add(obj_atom, "is", _lemma(adj))
                    for prep, pobj in _prep_phrase_children(obj):
                        add(obj_atom, prep, _lemma(pobj))

            # Adverbs on the verb: verb has_manner adverb
            if include_modifiers:
                for adv in _adverb_children(verb):
                    add(verb_lemma, "is", _lemma(adv))

            # Subordinate clauses — recurse by pulling sub-verb SVO
            if include_modifiers:
                for sub_verb in _subordinate_clauses(verb):
                    sub_subjs = list(_nsubjs(sub_verb))
                    sub_objs = list(_dobjs(sub_verb))
                    sub_rel = _lemma(sub_verb)
                    # Relative clauses: the subject is usually a relative
                    # pronoun ("that", "which", "who") that refers to the
                    # parent noun. Substitute the parent.
                    if sub_verb.dep_ == "relcl":
                        parent = sub_verb.head
                        if parent.pos_ in {"NOUN", "PROPN"}:
                            for sub_obj in sub_objs:
                                add(_lemma(parent), sub_rel, _lemma(sub_obj))
                    for ss in sub_subjs:
                        if ss.pos_ == "PRON":
                            continue  # already handled above for relcl
                        for so in sub_objs:
                            add(_lemma(ss), sub_rel, _lemma(so))

    return out


def head_verbs_in(statements: list[ParsedStatement]) -> set[str]:
    """Return every distinct verb (relation) used by a list of expanded
    statements. Caller can feed these to brain.teach_verb() before
    teaching the statements.
    """
    return {s.relation for s in statements if s.relation}
