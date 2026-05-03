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

## Verification (smoke test, 50 steps on 3070)

```
[l2] loading L1 checkpoint: src/sara_brain/cortex/checkpoints/grammar_base_015000.pt
[l2] projected L1 -> L2: copied=219 padded=1 skipped=1
[l2] trainable params: 134,400  frozen: 127,656,960  (tok_embed only)
[ud-lm/L2] split=train sentences=36404 ...
[ud-lm/L2] split=dev sentences=5922 ...
[eval] step=0 (pre-train)  dev_ppl=30.903
     1  3.8590   47.42  ...
    50  2.8239   16.84  ...
[eval] step=50  dev_ppl=16.449
[ckpt] /tmp/l2_smoke/l2_en_000050.pt
```

Pre-train dev ppl 30.9 → 16.4 in 50 steps. The 134K-param adapter is
learning the function-word distribution. A full 3000-step run on a 3070
takes a few minutes and should bring dev ppl close to L1's structural
ppl (2.8 — though comparing across vocab sizes is loose, the trend
direction is what matters).

Checkpoints round-trip:
```
ck['lang'] == 'en'
ck['frozen_l1'] == True
ck['config']['vocab_size'] == 175
GrammarModel.load_state_dict(ck['state_dict']) — clean
```

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
