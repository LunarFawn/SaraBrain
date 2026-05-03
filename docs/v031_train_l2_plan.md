# v031 — `train_l2.py` (third slice of L2-en)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v028_multi_layer_cortex_architecture.md](v028_multi_layer_cortex_architecture.md),
[v029_vocab_en_plan.md](v029_vocab_en_plan.md),
[v030_ud_lexicalize_plan.md](v030_ud_lexicalize_plan.md)

## Context

Per v028, the L1→L2 build sequence is:

1. ✅ `vocab_en.py` (commit `6d19e97`) — defines L2-en's vocabulary.
2. ✅ `ud.py` lexicalization (commit `6bc080c`) — emit lexicalized sequences when asked.
3. **(this doc)** `train_l2.py` — train the L2 layer on lexicalized UD.
4. (later) Integrate into the synthesizer pipeline.

L2 training is intentionally a small, scoped step: load an L1 checkpoint,
extend its vocabulary to L2-en (175 tokens vs 76), copy every L1 weight
verbatim, leave the new function-word embedding rows at random init, then
train **only** the embedding (which is weight-tied to the LM head, see
[model.py:71](../src/sara_brain/cortex/transformer/model.py#L71)) on
lexicalized UD English.

## What changes

Two files:

1. **`synthetic.UDStreamDataset` extended** with three opt-in parameters:
   `lexicalize_function_words`, `function_word_set`, `vocab_table`.
   Defaults preserve byte-for-byte L1 behaviour; L2 callers pass all
   three. The module-level `_encode` is kept for back-compat callers;
   the dataset now uses an instance method `self._encode` that respects
   `vocab_table`.

2. **`train_l2.py` (new)** — the L2 trainer.

## Architecture of `train_l2.py`

- **Load L1 checkpoint** (`--grammar-ckpt`, required). Reads
  `state_dict`, `config` (vocab_size=76).
- **Build L2 model** with the same architecture but `vocab_size=175`
  (`vocab_en.VOCAB_SIZE_EN`). Identical `d_model`, `n_heads`, `n_layers`,
  `d_ff`, `dropout`, `pad_id` as L1.
- **Project weights** via `project_l1_into_l2`:
  - `tok_embed.weight`: copy rows [0, 76) from L1; rows [76, 175) stay
    at the L2 model's random init (`N(0, 0.02²)` per `_init_weights`).
  - `head.weight`: skipped — tied to `tok_embed.weight` already.
  - Everything else (pos_embed, all blocks, ln_f): copied verbatim.
- **Freeze L1 layers** (default; toggle with `--unfreeze-l1`):
  only `tok_embed.weight` requires_grad=True. 134,400 trainable params
  vs 127,656,960 frozen — the entire L2-en overlay rides in 0.1% of the
  model.
- **L2 dataset**: `UDStreamDataset(lexicalize_function_words=True,
  function_word_set=EN_FUNCTION_WORD_SET, vocab_table=TOK2ID_EN)`.
- **Training loop**: cosine LR with warmup (reused from
  `train.cosine_lr`), AdamW(β=(0.9, 0.95), wd=0.1), bf16 autocast on
  CUDA, dev-perplexity eval on the same lexicalized corpus, periodic
  checkpoint saves at `l2_{lang}_{step:06d}.pt`. Same shape and
  conventions as `train.py`.
- **Pre-train sanity eval** at step 0 logs the dev ppl with random
  function-word embeddings — that's the headroom the trainer is closing.

## Defaults

| Flag | Default |
|---|---|
| `--grammar-ckpt` | required |
| `--steps` | 3000 |
| `--batch` | 32 |
| `--max-seq` | 96 |
| `--lr` | 3e-4 → 1e-5 (cosine) |
| `--warmup` | 100 |
| `--eval-every` | 500 |
| `--ckpt-every` | 500 |
| `--lang` | `en` (only `en` supported until other `vocab_*.py` files exist) |
| `--unfreeze-l1` | off |

## Verification — actual 3000-step run

Final results (RTX 3070, ~5.5 minutes wall):

```
[eval] step=0 (pre-train)  dev_ppl=38.462
[eval] step=500            dev_ppl=4.237
[eval] step=1000           dev_ppl=4.129
[eval] step=2500           dev_ppl=4.103
[eval] step=3000           dev_ppl=4.127
final_loss=1.3807  final_dev_ppl=4.127
```

**Pre-train 38.46 → trained 4.13 dev ppl** with only the 134K-param
embedding adapter. For comparison L1's structural-only dev ppl was
2.806 — L2 carries 99 extra tokens to predict, so a higher absolute
ppl is expected; what matters is the convergence ratio.

Sample output (`inference_l2.py --sample 5`, recognisable English
skeletons with function words in plausible positions):

```
[2]  nsubj they  root VERB  obj NOUN  cc and  amod ADJ  conj NOUN  punct PUNCT
     ≈ "they [VERB] [NOUN] and [ADJ] [NOUN] ."

[3]  root PRON  cop are  det the  amod ADJ  nsubj NOUN  amod ADJ
     case from  nmod PROPN  case to  nmod PROPN  flat PROPN
     ≈ "[PRON] are the [ADJ] [NOUN] [ADJ] from [PROPN] to [PROPN] ..."

[4]  nsubj PRON  aux AUX  aux AUX  root VERB  mark to  xcomp VERB
     obj it  punct PUNCT  mark that  nsubj PRON  advcl VERB  obj it  punct PUNCT
     ≈ "[PRON] [AUX] [AUX] [VERB] to [VERB] it . that [PRON] [VERB] it ."
```

L2 has learned which English function words slot into which
structural positions: `det the` after a noun phrase head, `case
from / to` as preposition cases, `cop are` as the copula, `mark to /
that` for subordinating clauses, `cc and` for coordination.

Dev-set scoring (`--score-dev 50`):
```
mean ppl over 50 EWT dev sentences = 4.222
```

Checkpoint at `src/sara_brain/cortex/checkpoints/l2_en_003000.pt`
(512 MB — most of which is the redundantly-saved frozen L1 weights;
a future slice can deduplicate by storing only the trainable rows
plus an L1 ckpt reference).

## Inference

`inference_l2.py` mirrors `inference.py` but loads against the L2-en
vocabulary. Two modes:

- `--sample N` — sample N tag streams from the trained model
- `--score-dev N` — score N lexicalized EWT dev sentences (clamped to
  the model's `max_seq`)

## Out of scope

- Spanish or other-language vocab/training (copy `vocab_en.py` →
  `vocab_es.py` with the Spanish allowlist; copy this trainer with
  `--lang=es` and a Spanish-treebank corpus).
- Sampling / inference using L2 (next slice — small wrapper that
  loads an L2 checkpoint and exposes a generation API).
- Synthesizer-pipeline integration (replaces v027 article heuristic
  once L2 inference is wired up).
- `--unfreeze-l1` regression suite (the toggle exists; using it costs
  L1's universality, so it's an opt-in escape hatch, not a default).
