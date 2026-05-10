"""Generate (delexicalized real prose, delexicalized triple) training
pairs from the cached UD treebanks.

Labels come from the GOLD UD dependency tree, not from running a fresh
parser + rule stub. UD treebanks ship with human-curated parses; that's
much higher quality than what spaCy infers on its own. The
`ud_triple_extractor` module reads gold deprels (`nsubj`, `obj`, `obl`,
`cop`, `conj`, particles) and produces canonical (s, r, o) per clause.

Pipeline per UD sentence:

  1. Reconstruct surface text from the UD `.conllu` token forms.
  2. Walk the gold parse → list of (subject, relation, object) triples
     from `extract_triples_from_ud`.
  3. Build a delexicalization mapping that grows across the whole
     corpus — same surface word always maps to the same nonsense.
  4. Apply the map to both prose AND triple parts.
  5. Emit Pair-shaped records (compatible with synthetic_pairs.Pair).

Content-orthogonality is preserved: the model never sees the original
words, only their consistent nonsense substitutes. Real syntactic
distribution is preserved: each pair carries the gold structural
relationships from the treebank.
"""
from __future__ import annotations

import sys

from ..v2 import synthetic_pairs as sp
from .delexicalizer import DelexMapping, delexicalize_phrase, delexicalize_text
from .ud_triple_extractor import extract_triples_from_ud
from .. import ud as ud_mod


def _reconstruct_surface(sentence) -> str:
    return " ".join(t.form for t in sentence.tokens if t.form)


def _find_spans(prose: str, subject: str, relation: str, obj: str
                ) -> tuple[tuple[int, int], tuple[int, int],
                           tuple[int, int]] | None:
    """Locate (subject, relation, object) char spans in prose, reusing
    synthetic_pairs's order-independent span finder."""
    p = sp._build_pair_strings(
        prose, subject, relation, obj,
        template_name="real_prose", qualifiers=[],
    )
    if p is None:
        return None
    return p.subject_span, p.relation_span, p.object_span


def generate_real_prose_pairs(
    nlp,
    *,
    treebanks: list[str] | None = None,
    splits: tuple[str, ...] = ("train",),
    max_sentences: int = 0,
    seed: int = 0,
    verbose: bool = True,
) -> list[sp.Pair]:
    """Walk UD treebanks → gold-tree triples → delexicalize → emit Pairs.

    Args:
      nlp: spaCy nlp (used only by the delexicalizer to drop closed-
        class function words from substitution). Should be the
        domain-aware nlp from `feature_extractor.load_domain_nlp()`.
      treebanks: UD English treebanks. None = all six.
      splits: UD splits, default "train".
      max_sentences: cap (0 = no cap).
      seed: nonsense generator seed.
    """
    import random
    if treebanks is None:
        treebanks = list(ud_mod.ENGLISH_ALL)
    mapping = DelexMapping(rng=random.Random(seed))
    pairs: list[sp.Pair] = []
    sentences_seen = 0
    sentences_with_triple = 0

    for tb in treebanks:
        for split in splits:
            try:
                path = ud_mod.ensure_split(tb, split)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"[real-prose] skip {tb}/{split}: {e}",
                          file=sys.stderr)
                continue
            for sent in ud_mod.parse_conllu(path):
                if max_sentences and sentences_seen >= max_sentences:
                    break
                sentences_seen += 1
                surface = _reconstruct_surface(sent)
                if not surface or len(surface.split()) < 3:
                    continue
                # Step 1: extract gold triples directly from the UD parse.
                gold_triples = extract_triples_from_ud(sent)
                if not gold_triples:
                    continue
                for tri in gold_triples:
                    sentences_with_triple += 1
                    # Step 2: delexicalize the prose. Mapping accumulates
                    # across the corpus so repeated words stay consistent.
                    delex_prose, mapping = delexicalize_text(
                        surface, nlp, mapping=mapping,
                    )
                    # Step 3: delexicalize each triple part using the
                    # same mapping (closed-class words pass through).
                    delex_subject = delexicalize_phrase(tri.subject, mapping)
                    delex_relation = delexicalize_phrase(tri.relation, mapping)
                    delex_object = delexicalize_phrase(tri.object, mapping)
                    # Step 4: re-find spans in the delex prose.
                    spans = _find_spans(
                        delex_prose, delex_subject, delex_relation, delex_object,
                    )
                    if spans is None:
                        continue
                    s_span, r_span, o_span = spans
                    pairs.append(sp.Pair(
                        prose=delex_prose,
                        subject=delex_subject,
                        relation=delex_relation,
                        obj=delex_object,
                        subject_span=s_span,
                        relation_span=r_span,
                        object_span=o_span,
                        template=f"real_prose_{tb}",
                        qualifiers=[],
                    ))
            if max_sentences and sentences_seen >= max_sentences:
                break
        if max_sentences and sentences_seen >= max_sentences:
            break

    if verbose:
        print(
            f"[real-prose] sentences_seen={sentences_seen} "
            f"with_triple={sentences_with_triple} pairs_emitted={len(pairs)} "
            f"corpus_vocab_size={len(mapping.word_to_nonsense)}",
            file=sys.stderr,
        )
    return pairs


__all__ = ["generate_real_prose_pairs"]
