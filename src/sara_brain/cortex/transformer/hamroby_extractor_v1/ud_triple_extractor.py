"""Extract (subject, relation, object) triples directly from gold UD trees.

Replaces "run rule stub on a spaCy parse" with "walk the gold-annotated
UD dependency tree." The treebanks ship with human-curated parses;
that's a much higher-quality signal than what spaCy infers from
scratch on each sentence. The extractor's job is just to read the
gold tree and pull out the canonical (s, r, o) for each clause.

Algorithm (per sentence):

  1. Find predicates: VERB-rooted clauses + copular constructions.
  2. For each predicate, locate `nsubj` / `nsubj:pass`.
  3. Object: `obj`, `iobj`, `obl` (objects of obliques like prep phrases),
     or `xcomp` for "want to X". For copular, the head IS the predicate
     and the cop child carries the surface verb.
  4. Subject = subtree of the nsubj head (rooted at it, descending
     through all its dep-children).
  5. Object = subtree of the object head.
  6. Relation = predicate token + immediate `case` / `compound:prt` /
     particle children located between subject and object.

Conjuncts ("X and Y") get their own triples — each conjunct of an
nsubj or obj is paired with the predicate independently.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .. import ud as ud_mod


@dataclass
class GoldTriple:
    subject: str
    relation: str
    object: str
    sentence: str   # surface text of the original UD sentence


# UD relation labels that count as a subject. spaCy/CLearNLP path is
# normalized upstream; UD's own annotation uses these directly.
_SUBJ_DEPRELS = frozenset({"nsubj", "csubj"})
# Primary object-bearing UD deprels. These are always treated as objects
# of the predicate.
_PRIMARY_OBJ_DEPRELS = frozenset({"obj", "iobj", "xcomp", "ccomp"})
# Oblique modifiers. UD tags `obl` for PP-modifiers like "by similarity"
# or "during loading" — semantically these are adjuncts, not arguments.
# Only treat `obl` as a triple object when the predicate has NO primary
# object (e.g. "fold INTO HAIRPINS" — `hairpins` is `obl`, no `obj`, so
# it IS the object). When a primary `obj` exists, `obl` is an oblique
# modifier and should be skipped — matches the rule stub's behavior.
_OBLIQUE_DEPRELS = frozenset({"obl"})
# Kept for backwards reference; not used directly in the algorithm.
_OBJ_DEPRELS = _PRIMARY_OBJ_DEPRELS | _OBLIQUE_DEPRELS
# Particles/preps attached to the verb that should be glued onto the
# relation (e.g. "fold into" — `into` is `case` attached to the obl).
_RELATION_PARTICLE_DEPRELS = frozenset({
    "compound:prt", "advmod:prt",
})


def _strip_subtype(deprel: str) -> str:
    """UD subtype labels look like `nsubj:pass`. Strip the subtype."""
    return deprel.split(":", 1)[0]


def _build_children_index(tokens) -> dict[int, list[int]]:
    """Map token-index → list of child token-indices (1-indexed UD heads
    converted to 0-indexed). Returns indices into `tokens`."""
    out: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(tokens):
        # UD `head` is 1-indexed; 0 means ROOT.
        if 0 < t.head <= len(tokens):
            out[t.head - 1].append(i)
    return out


def _subtree_indices(
    root_idx: int,
    children: dict[int, list[int]],
    *,
    exclude: set[int] | None = None,
) -> set[int]:
    """All token indices in the subtree rooted at `root_idx`.
    Excludes any index in `exclude` and never descends through it."""
    exclude = exclude or set()
    nodes: set[int] = {root_idx}
    queue = [root_idx]
    while queue:
        cur = queue.pop()
        for c in children.get(cur, ()):
            if c in exclude or c in nodes:
                continue
            nodes.add(c)
            queue.append(c)
    return nodes


def _join_subtree(tokens, indices: set[int]) -> str:
    """Render the surface text of a set of token indices in original order."""
    return " ".join(tokens[i].form for i in sorted(indices) if tokens[i].form)


def extract_triples_from_ud(sentence) -> list[GoldTriple]:
    """Pull canonical (subject, relation, object) triples from the gold
    UD parse of a single sentence."""
    tokens = sentence.tokens
    if not tokens:
        return []
    children = _build_children_index(tokens)
    surface = " ".join(t.form for t in tokens if t.form)

    triples: list[GoldTriple] = []

    # Track predicates: VERB heads or any token with a `cop` child.
    for pred_idx, pred in enumerate(tokens):
        cop_children = [c for c in children.get(pred_idx, ())
                        if tokens[c].dep == "cop"]
        is_verb = pred.upos == "VERB"
        is_copular = bool(cop_children)
        if not (is_verb or is_copular):
            continue

        # All conjuncts of nsubj/obj count too. Collect subject heads:
        subj_heads: list[int] = []
        for c in children.get(pred_idx, ()):
            base = _strip_subtype(tokens[c].dep)
            if base in _SUBJ_DEPRELS:
                subj_heads.append(c)
                # Conjuncts of the subject (e.g. "X and Y verbed Z" → X, Y)
                for cc in children.get(c, ()):
                    if _strip_subtype(tokens[cc].dep) == "conj":
                        subj_heads.append(cc)
        if not subj_heads:
            continue

        # Object heads. For copular, the predicate IS the object slot.
        obj_heads: list[int] = []
        if is_copular and not is_verb:
            obj_heads.append(pred_idx)
        # Two-pass: prefer primary objects (obj/iobj/xcomp/ccomp) over
        # obliques (obl). When a predicate has at least one primary
        # object, treat `obl` children as oblique modifiers (adjuncts)
        # and skip them — they're not additional triple objects. When
        # the predicate has NO primary object, fall back to `obl` (e.g.
        # "fold INTO HAIRPINS" — no `obj`, `hairpins` IS the object).
        primary_obj_children: list[int] = []
        oblique_children: list[int] = []
        for c in children.get(pred_idx, ()):
            base = _strip_subtype(tokens[c].dep)
            if base in _PRIMARY_OBJ_DEPRELS:
                primary_obj_children.append(c)
            elif base in _OBLIQUE_DEPRELS:
                oblique_children.append(c)
        # For copular predicates, the predicate noun IS the "object" —
        # oblique modifiers should NOT add extra triples.
        if is_copular and not is_verb:
            use_obl = False
        elif primary_obj_children:
            use_obl = False
        else:
            use_obl = True
        candidate_children = primary_obj_children + (
            oblique_children if use_obl else []
        )
        for c in candidate_children:
            obj_heads.append(c)
            for cc in children.get(c, ()):
                if _strip_subtype(tokens[cc].dep) == "conj":
                    obj_heads.append(cc)
        if not obj_heads:
            continue

        # Relation. For copular, use the cop child (the surface verb).
        # For verb predicates, use the verb token + any particle/prep
        # children that come before the first object.
        if is_copular and not is_verb:
            rel_indices = sorted(cop_children)
        else:
            rel_indices = [pred_idx]
            for c in children.get(pred_idx, ()):
                base = _strip_subtype(tokens[c].dep)
                if base in _RELATION_PARTICLE_DEPRELS:
                    rel_indices.append(c)
                # `case` attached to obl can act as a verb particle —
                # but only when it sits right after the verb. Skip
                # broader case attachments to avoid pulling in the
                # whole prepositional phrase.
        rel_text = _join_subtree(tokens, set(rel_indices)).strip()
        if not rel_text:
            continue

        # Build subject and object texts. Cross-pair every subject
        # conjunct with every object conjunct (handles "X and Y verbed
        # A and B" → 4 triples).
        for s_head in subj_heads:
            # Don't descend into other subject conjuncts when collecting
            # this subject's subtree — keeps each conjunct as its own
            # span.
            other_subj = set(subj_heads) - {s_head}
            s_indices = _subtree_indices(
                s_head, children,
                exclude=other_subj | set(obj_heads) | {pred_idx},
            )
            s_text = _join_subtree(tokens, s_indices).strip()
            if not s_text:
                continue
            for o_head in obj_heads:
                if o_head == s_head:
                    continue
                other_obj = set(obj_heads) - {o_head}
                o_excl = (other_obj | set(subj_heads) | set(rel_indices))
                # For copular, predicate IS the object — don't exclude self.
                if o_head != pred_idx:
                    o_excl.add(pred_idx)
                o_indices = _subtree_indices(
                    o_head, children, exclude=o_excl,
                )
                o_text = _join_subtree(tokens, o_indices).strip()
                if not o_text:
                    continue
                triples.append(GoldTriple(
                    subject=s_text,
                    relation=rel_text,
                    object=o_text,
                    sentence=surface,
                ))

    return triples


def iter_gold_triples(
    treebanks: list[str] | None = None,
    splits: tuple[str, ...] = ("train",),
    *,
    max_sentences: int = 0,
):
    """Yield GoldTriple records by walking UD treebanks. Each yielded
    triple includes the original sentence text on `.sentence`."""
    if treebanks is None:
        treebanks = list(ud_mod.ENGLISH_ALL)
    seen = 0
    for tb in treebanks:
        for split in splits:
            try:
                path = ud_mod.ensure_split(tb, split)
            except Exception:
                continue
            for sent in ud_mod.parse_conllu(path):
                if max_sentences and seen >= max_sentences:
                    return
                seen += 1
                for tri in extract_triples_from_ud(sent):
                    yield tri


__all__ = [
    "GoldTriple",
    "extract_triples_from_ud",
    "iter_gold_triples",
]
