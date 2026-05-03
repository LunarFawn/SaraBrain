# v030 — `ud.py` lexicalization (second slice of L2-en)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v028_multi_layer_cortex_architecture.md](v028_multi_layer_cortex_architecture.md),
[v029_vocab_en_plan.md](v029_vocab_en_plan.md)

## Context

Per v028/v029, the L1→L2 build sequence is:

1. ✅ `vocab_en.py` (commit `6d19e97`) — defines L2-en's vocabulary.
2. **(this doc)** `ud.py` extension — emit lexicalized sequences when asked.
3. (next) `train_l2.py` — train the L2 layer on lexicalized UD.
4. (later) Integrate into the synthesizer pipeline.

The L1 training path produces token streams like
`nsubj NOUN cop AUX det DET amod ADJ nmod NOUN`. For L2 training we need
the same skeleton but with English function-word literals where they
appear:
`nsubj NOUN cop is det the amod ADJ nmod NOUN`.

This is the change `to_input_tokens()` needs to support — opt-in, off by
default so existing L1 training is byte-for-byte unchanged.

## What changes

Three small edits to [src/sara_brain/cortex/transformer/ud.py](../src/sara_brain/cortex/transformer/ud.py):

1. **`UDToken` gains a `form: str = ""` field.** Lowercased surface
   form. Defaults to empty for safety and back-compat (existing
   constructors that don't pass `form` still work).

2. **`parse_conllu()` populates `form`.** The lowercased form is
   already computed at line 102 (`form_lower = parts[1].lower()`) for
   the question/negation marker detection. We just store it in the
   resulting `UDToken`.

3. **`to_input_tokens()` gains two opt-in parameters:**
   ```python
   def to_input_tokens(
       sent: UDSentence,
       max_tokens: int = 32,
       lexicalize_function_words: bool = False,
       function_word_set: frozenset[str] | None = None,
   ) -> list[str]:
   ```
   When both the flag is on and a set is provided, each token emits
   `t.dep` + (literal `t.form` if it's in the set, else `t.upos`).
   The structural skeleton (the `dep` half) is unchanged — only the
   second half flips between UPOS and literal form per token.

## What does NOT change

- `parse_conllu`'s output shape (still yields `UDSentence`).
- `to_input_tokens` default behaviour (no flag = identical output).
- `UDStreamDataset` in [synthetic.py](../src/sara_brain/cortex/transformer/synthetic.py).
  It still calls `to_input_tokens(sent, max_tokens=...)` with no extra
  args, so the L1 training path is byte-for-byte the same.
- `vocab.py`. Still the L1 vocabulary; L1 checkpoints keep loading.
- `vocab_en.py`. Imported only by callers that want lexicalized
  output (none yet — the L2 trainer slice will be the first consumer).

## Files

**Modified:**
- [src/sara_brain/cortex/transformer/ud.py](../src/sara_brain/cortex/transformer/ud.py)

**Read (no edits):**
- [src/sara_brain/cortex/transformer/synthetic.py](../src/sara_brain/cortex/transformer/synthetic.py)
  — confirm L1 dataset still works.
- [src/sara_brain/cortex/transformer/vocab_en.py](../src/sara_brain/cortex/transformer/vocab_en.py)
  — used in the verification step.

## Verification

1. **Existing L1 path unaffected.** Build a `UDStreamDataset(split="dev")`
   with no flag and confirm `len()`, average sequence length, and a
   sample of streams match what they were before this commit.
2. **Lexicalization opt-in.** Take one parsed UD sentence, run
   `to_input_tokens(sent, lexicalize_function_words=True,
   function_word_set=EN_FUNCTION_WORD_SET)` and confirm function-word
   forms appear in the output where they appeared in the source
   sentence; UPOS tags appear elsewhere.
3. **Backward-compat constructors.** `UDToken(upos="NOUN", dep="nsubj",
   head=1, is_q_marker=False, is_neg=False)` still works (no `form`
   required).

## Out of scope

- `train_l2.py` (next slice).
- Modifying `UDStreamDataset` to pass the flag through (the L2 trainer
  will either extend it or build a sibling).
- Spanish or other-language ingestion (same recipe; just needs a
  `vocab_es.py` and a different UD treebank list).
