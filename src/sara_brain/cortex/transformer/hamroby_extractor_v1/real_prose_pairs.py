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
from .feature_extractor import ParsedSentence, _CLOSED_CLASS_POS
from .vocab import (
    DEP_TO_ID, NONE_FUNCWORD_ID, POS_TO_ID,
    UNK_DEP_ID, UNK_POS_ID,
    encode_funcword, encode_head_offset,
)
from .. import ud as ud_mod


# UPOS tags considered closed-class for delexicalization. Mirrors
# delexicalizer.py's _CLOSED_CLASS_POS — closed-class words pass through
# verbatim in delex prose; open-class get nonsense substitutes.
_DELEX_CLOSED_CLASS_UPOS: frozenset[str] = frozenset({
    "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PRON", "PART",
})


def _reconstruct_surface(sentence) -> str:
    return " ".join(t.form for t in sentence.tokens if t.form)


def _ud_delexicalize(ud_sent, mapping: DelexMapping
                    ) -> tuple[str, list[str]]:
    """Delexicalize a UD sentence directly from UD tokens (no spaCy).

    Returns:
      delex_prose: space-joined delexicalized forms (non-empty tokens).
      delex_forms: per-UD-token delex form, aligned 1:1 with
                   ud_sent.tokens. Empty UD tokens yield empty entries
                   (filtered when joining the prose).

    Aligning delex forms to UD tokens by construction lets the caller
    attach gold UD features (POS, dep, head) to each delex word without
    re-tokenizing.
    """
    delex_forms: list[str] = []
    for tok in ud_sent.tokens:
        if not tok.form:
            delex_forms.append("")
            continue
        if tok.upos == "PUNCT" or tok.upos in _DELEX_CLOSED_CLASS_UPOS:
            delex_forms.append(tok.form)
        else:
            # Open-class content word — substitute via the corpus mapping.
            delex_forms.append(mapping.substitute(tok.form))
    delex_prose = " ".join(f for f in delex_forms if f)
    return delex_prose, delex_forms


def _ud_to_parsed(ud_sent, delex_forms: list[str],
                  delex_prose: str) -> ParsedSentence:
    """Build a ParsedSentence from gold UD features.

    Mirrors `parse_sentence(..., drop_punct=True)` behavior: PUNCT
    tokens are skipped from words/feature_ids/char_offsets. Head offset
    is computed relative to the kept (non-PUNCT) word index.
    """
    # Pass 1: build a UD-token-index → output-word-index map (skip PUNCT
    # and empty-form tokens, same rule as parse_sentence).
    keep: list[bool] = []
    for tok in ud_sent.tokens:
        keep.append(bool(tok.form) and tok.upos != "PUNCT")
    ud_to_output_idx: dict[int, int] = {}
    next_idx = 0
    for i, k in enumerate(keep):
        if k:
            ud_to_output_idx[i] = next_idx
            next_idx += 1

    # Pass 2: walk all non-empty UD tokens. Track cumulative char
    # position in the joined delex_prose (the prose is the non-empty
    # delex_forms joined by single spaces). Emit feature tuples and
    # char offsets only for kept tokens; PUNCT still advances the
    # cursor since it occupies a slot in the joined prose.
    words: list[str] = []
    feature_ids: list[tuple[int, int, int, int]] = []
    char_offsets: list[tuple[int, int]] = []
    cum = 0
    for i, (tok, dform) in enumerate(zip(ud_sent.tokens, delex_forms)):
        if not tok.form:
            continue
        token_start = cum
        token_end = cum + len(dform)
        cum = token_end + 1  # +1 for the space-join separator

        if not keep[i]:
            continue  # PUNCT — slot in prose, but not in words/features

        pos_id = POS_TO_ID.get(tok.upos, UNK_POS_ID)
        dep_id = DEP_TO_ID.get(tok.dep, UNK_DEP_ID)

        # Head offset relative to the kept-words array. UDToken.head is
        # 1-indexed (0 = root). Self-pointing fallback when the head was
        # dropped (e.g. PUNCT root, rare).
        if tok.head and tok.head > 0:
            head_ud_idx = tok.head - 1
            if head_ud_idx in ud_to_output_idx:
                head_offset = (ud_to_output_idx[head_ud_idx]
                               - ud_to_output_idx[i])
            else:
                head_offset = 0
        else:
            head_offset = 0
        offset_id = encode_head_offset(head_offset)

        # Funcword stream: encode lowercased form for closed-class POS,
        # NONE otherwise. Mirrors parse_sentence exactly. (Closed-class
        # forms pass through delex unchanged, so dform == tok.form.)
        if tok.upos in _CLOSED_CLASS_POS:
            funcword_id = encode_funcword(dform.lower())
        else:
            funcword_id = NONE_FUNCWORD_ID

        words.append(dform)
        feature_ids.append((pos_id, dep_id, offset_id, funcword_id))
        char_offsets.append((token_start, token_end))

    return ParsedSentence(
        text=delex_prose,
        words=words,
        feature_ids=feature_ids,
        char_offsets=char_offsets,
    )


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
                # Step 2: delexicalize the surface ONCE per sentence using
                # UD tokens directly (not spaCy). UD-aligned delex forms
                # let us attach gold UD features to each delex word.
                delex_prose, delex_forms = _ud_delexicalize(sent, mapping)
                if not delex_prose:
                    continue
                # Step 3: build a single ParsedSentence with gold UD
                # features. Shared across all Pairs from this sentence —
                # the parse is the same for every triple in the clause.
                gold_parsed = _ud_to_parsed(sent, delex_forms, delex_prose)
                if not gold_parsed.words:
                    continue
                # Step 4: group gold triples by (subject, relation) so that
                # conj-of-dobj triples sharing the same predicate are
                # emitted as ONE Pair with multiple object spans, not as
                # N competing per-conjunct Pairs. The model trains on
                # multi-B-O labels directly instead of averaging away
                # which conjunct to tag.
                grouped: dict[tuple[str, str], list] = {}
                group_order: list[tuple[str, str]] = []
                for tri in gold_triples:
                    delex_s = delexicalize_phrase(tri.subject, mapping)
                    delex_r = delexicalize_phrase(tri.relation, mapping)
                    key = (delex_s, delex_r)
                    if key not in grouped:
                        grouped[key] = []
                        group_order.append(key)
                    grouped[key].append((tri, delex_s, delex_r))

                for key in group_order:
                    bucket = grouped[key]
                    delex_subject, delex_relation = key
                    # Per-object spans: walk the bucket in order and look
                    # up each object's span via _find_spans. The first
                    # successfully-located object becomes the primary;
                    # subsequent ones go into additional_object_spans.
                    primary_obj: str | None = None
                    primary_s_span = primary_r_span = primary_o_span = None
                    extra_o_spans: list[tuple[int, int]] = []
                    extra_obj_strs: list[str] = []
                    for tri, _, _ in bucket:
                        delex_object = delexicalize_phrase(tri.object, mapping)
                        spans = _find_spans(
                            delex_prose,
                            delex_subject, delex_relation, delex_object,
                        )
                        if spans is None:
                            continue
                        s_span, r_span, o_span = spans
                        if primary_obj is None:
                            primary_obj = delex_object
                            primary_s_span = s_span
                            primary_r_span = r_span
                            primary_o_span = o_span
                        else:
                            extra_o_spans.append(o_span)
                            extra_obj_strs.append(delex_object)
                    if primary_obj is None:
                        continue
                    sentences_with_triple += len(bucket)
                    # `obj` field is human-readable; join all object
                    # strings with " | " so logs/JSON dumps show all
                    # labeled objects.
                    obj_label = primary_obj
                    if extra_obj_strs:
                        obj_label = " | ".join([primary_obj] + extra_obj_strs)
                    pairs.append(sp.Pair(
                        prose=delex_prose,
                        subject=delex_subject,
                        relation=delex_relation,
                        obj=obj_label,
                        subject_span=primary_s_span,
                        relation_span=primary_r_span,
                        object_span=primary_o_span,
                        template=f"real_prose_{tb}",
                        qualifiers=[],
                        pre_parsed=gold_parsed,
                        additional_object_spans=extra_o_spans or None,
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
