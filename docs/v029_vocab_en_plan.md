# v029 — `vocab_en.py` plan (first slice of L2-en)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v028_multi_layer_cortex_architecture.md](v028_multi_layer_cortex_architecture.md)

## Context

Per v028, HamRobyLLM is being decomposed into three layers:

- **L1** — universal grammatical capacity. Today's grammar transformer is a
  prototype: 76 structural tokens (UPOS + UD deps + slots), no actual words.
- **L2** — per-language function-word overlay. Doesn't exist yet.
- **L3** — substrate. Already exists (`brain.db`).

The v027 article-heuristic problem (`Cat is mammal` should be `Cat is a mammal`)
is the visible symptom of L2 missing. Root cause: L1's vocabulary literally
cannot emit `a` / `an` / `the` because those tokens aren't in its world. No
amount of L1 training fixes this — it's a vocabulary problem, not a capacity
problem.

`vocab_en.py` is the **first concrete file** in the L1 → L2 build sequence. It
is purely a vocabulary definition — no model code, no training code. Once it
exists, `ud.py` can be extended to emit lexicalized sequences (function words
literal, content words abstracted), which feeds the L2 trainer.

This doc covers **only** `vocab_en.py`. Subsequent slices (`ud.py` extension,
`train_l2.py`, integration) get their own docs.

## What `vocab_en.py` does

A new file at [src/sara_brain/cortex/transformer/vocab_en.py](src/sara_brain/cortex/transformer/vocab_en.py).
It produces a **superset vocabulary** that re-exports every L1 token at its
existing ID, then appends ~100 English function-word literals at IDs ≥76.

Concretely the file exports:

| Symbol | Type | Purpose |
|---|---|---|
| `VOCAB_EN` | `list[str]` | L1 `VOCAB` + English function words, in that order |
| `TOK2ID_EN` | `dict[str, int]` | token → id |
| `ID2TOK_EN` | `dict[int, str]` | id → token |
| `VOCAB_SIZE_EN` | `int` | `len(VOCAB_EN)` (~175) |
| `ENGLISH_FUNCTION_WORDS` | `tuple[str, ...]` | The closed-class allowlist itself |
| `EN_FUNCTION_WORD_SET` | `frozenset[str]` | Lookup helper for `ud.py` |
| `is_english_function_word(token: str) -> bool` | function | Lowercased lookup |
| Re-exported: `PAD_ID`, `BOS_ID`, `EOS_ID`, `SEP_ID`, `UNK_ID` | int | Same values as L1 |

**Critical invariant:** every L1 token keeps its existing ID. The L1
checkpoint at `src/sara_brain/cortex/checkpoints/grammar_base_015000.pt`
stores `vocab_size=76` in its config (saved by [train.py:217](../src/sara_brain/cortex/transformer/train.py#L217)).
We never renumber. New tokens append above. Special IDs 0–4 are referenced by
hardcoded integer in inference loops ([inference.py:45](../src/sara_brain/cortex/transformer/inference.py#L45),
[inference.py:56](../src/sara_brain/cortex/transformer/inference.py#L56)) — they
**must** remain `PAD=0`, `BOS=1`, `EOS=2`, `SEP=3`, `UNK=4`.

### The function-word allowlist

Closed-class English function words, ~100 tokens, conservative first cut.
Grouped (groups are documentation only — order in `ENGLISH_FUNCTION_WORDS`
is sequential and stable):

```
DETERMINERS  (12): a an the this that these those some any every each no
PREPOSITIONS (19): of in on at by for with to from as into over under between
                   through against about before after
CONJUNCTIONS (14): and or but nor so yet because although if when while since
                   unless until
AUXILIARIES  (22): is are was were be been being has have had do does did
                   can could will would shall should may might must
PRONOUNS     (15): it they them their its which who whose whom he she him her
                   his hers
NEG/PARTICLES(10): not n't also only just then now here there 's
QUANTIFIERS  ( 7): more less much many few several all
```

Total: ~99. Treated as a single ordered tuple so IDs are stable.

Notes / decisions:
- `n't` and `'s` are split tokens in UD (`don't` → `do` + `n't`). Included as
  separate literals because that's how the treebank produces them.
- Personal pronouns (`he`, `she`, `him`, `her`, ...) included even though
  they refer to specific entities — they're closed-class, finite, and
  grammatically required by L2 to produce "It is..." / "They are..." prose.
- Numbers (`one`, `two`, ...) **excluded** — they look closed-class but the
  set is unbounded; treat numbers as content words handled by L3.
- Common adverbs (`really`, `very`, `quite`) **excluded** — open-class.
  Conservative cut; revisit after first L2 training run.
- Removed/bare-form lemmas only. No inflectional variants beyond what the
  treebank actually emits (e.g., we list `is` / `are` / `was` / `were`
  separately because UD doesn't lemmatize them at the token level).

Easy to extend later: append more tokens at the end. Never reorder.

## How this gets us to L1 + L2

`vocab_en.py` is the foundation; the rest of the build sequence is what
turns it into a working L2 layer.

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1 (THIS PLAN): vocab_en.py                             │
│   defines what L2-en's vocabulary IS                        │
│   IDs 0-75 inherited from L1, 76-~175 are English literals  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2 (next plan): ud.py extension                         │
│   add `lexicalize_function_words: bool` flag to             │
│   `to_input_tokens()` at ud.py:138-139                      │
│   when on, check `is_english_function_word(form)`:          │
│     yes → append the literal form                           │
│     no  → append the UPOS tag (existing behaviour)          │
│   single hook point, ~5 lines of change                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3 (next plan): train_l2.py                             │
│   load L1 checkpoint (vocab_size=76)                        │
│   project into L2 model (vocab_size=~175):                  │
│     - copy L1 embedding rows 0..75 verbatim                 │
│     - randomly init rows 76..174 (the new function words)   │
│     - same for LM head                                      │
│   freeze L1 transformer blocks (optional first try)         │
│   train on lexicalized UD corpus                            │
│   target: dev perplexity ≤ L1's structural ppl on the       │
│   structural-tokens-only subset, plus a learnable function  │
│   word distribution                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4 (later): integrate into synthesizer pipeline         │
│   replace v027 article heuristic                            │
│   labeler in synth_data.py emits L2-grammatical prose       │
│   HamRobySum (path 2) trains on top of (L1, L2-en)           │
└─────────────────────────────────────────────────────────────┘
```

The L1 / L2 separation in v028 is realized exactly by this chain:

- L1 model = today's checkpoint, `vocab_size=76`, frozen scaffold.
- L2 model = same architecture, `vocab_size=~175`, knows English function-word
  grammar.
- Anyone who wants Spanish creates `vocab_es.py` with the same shape and
  rebuilds steps 2–3 against Spanish UD treebanks. L1 is never retrained.

`vocab_en.py` itself is small and standalone. Its value is that **every
later step has a single import path for "what L2-en knows about words."**
No magic strings sprinkled across `ud.py` and the trainer; one source of
truth.

## Files

**New:**
- [src/sara_brain/cortex/transformer/vocab_en.py](../src/sara_brain/cortex/transformer/vocab_en.py)
  — the new module described above. ~150 lines including the allowlist
  and the helper function.

**Read-only references** (re-exported from):
- [src/sara_brain/cortex/transformer/vocab.py](../src/sara_brain/cortex/transformer/vocab.py)
  — `VOCAB`, `TOK2ID`, `ID2TOK`, `PAD_ID`, `BOS_ID`, `EOS_ID`, `SEP_ID`,
  `UNK_ID`, `VOCAB_SIZE`. **Not modified.** The L1 vocab stays exactly
  as it is, so existing L1 checkpoints continue to load unchanged.

**Not touched in this slice:**
- `model.py`, `train.py`, `inference.py`, `router_head.py`, `ud.py`,
  `synthetic.py`, `train_router.py` — no L1 consumer changes. The new
  module is opt-in; nothing imports it yet.

## Verification

End-to-end checks for `vocab_en.py` on its own (no model training needed
for this slice):

1. **Import check** — module loads, all expected names exist.
2. **L1 ID stability** — every L1 token keeps its L1 ID under `TOK2ID_EN`.
3. **Special IDs unchanged** — `(PAD, BOS, EOS, SEP, UNK) == (0, 1, 2, 3, 4)`.
4. **Allowlist properties** — no duplicates, all lowercase, count matches expected.
5. **Helper round-trip** — `is_english_function_word("the") == True`,
   `is_english_function_word("cat") == False`, case-insensitive.
6. **L1 checkpoint still loads unchanged** — sanity that we didn't disturb L1.

If all six pass, `vocab_en.py` is in place and ready for the next slice
(`ud.py` extension).

## Out of scope

- `ud.py` lexicalization (next plan).
- `train_l2.py` (later plan).
- Checkpoint weight projection from L1 → L2 (handled inside `train_l2.py`).
- Adding adverbs / numbers / open-class extensions (deferred until first
  L2 training run shows whether the conservative allowlist suffices).
- Spanish or other-language vocab files (the same recipe; copy + adapt
  after L2-en is validated).
