# v034 — HamRobySum v1: unfreeze top blocks + repetition penalty + multi-brain corpus

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v033_reproducible_hamroby_loop.md](v033_reproducible_hamroby_loop.md)

## Context

HamRobySum v0 (commit `0f74acb` status) trained end-to-end but doesn't
ship: dev_loss=4.13 vs train_loss=1.83 (10× perplexity gap = clear
overfit), and inference produces repetition-loop output on long
clusters ("rna stability affect and is a rna aptamer stability affect
for rna stability affect for ...").

Three root causes, all fixable without leaving the no-API
reproducibility path of [v033_reproducible_hamroby_loop.md](v033_reproducible_hamroby_loop.md):

1. **Frozen base wrong-distribution.** L1+L2 transformer blocks were
   trained on conversational UD English; HamRobySum's input is a
   structured fact dump. Frozen blocks can't adapt to the new task.
2. **No decoder-side defense against loops.** Vanilla greedy / sampling
   decoding has no mechanism to break out of repetition attractors.
3. **Tiny corpus.** 600 training pairs from one brain (~9.4k loss
   positions) against 134K trainable params — upside-down for
   generalization. The repo has 11 substrate files in the root —
   ~60 MB of additional supervision sitting there unused.

This slice addresses all three.

## What changes

Three files modified; no new files added.

### `src/sara_brain/cortex/transformer/train_synth.py`

- New CLI arg `--unfreeze-top-n N` (default 2). Replaces the binary
  `--unfreeze-base` flag (which stays as a deprecated alias for
  `--unfreeze-top-n -1` meaning "all blocks").
- `--pairs PATH` becomes repeatable (`action='append'`) so multiple
  per-brain JSONLs can be passed if pre-generated separately. (Most
  of the time the multi-brain corpus is built by `synth_data` and
  passed as one combined JSONL — see the `synth_data` change below.)
- `freeze_base_params(model, top_n)` extended: keeps `tok_embed`
  trainable AND unfreezes the last `top_n` `TransformerBlock` modules
  in `model.blocks`. Each block adds ~7M trainable params (12-head
  attention + 768→3072→768 FF).
- Trainer logs trainable count broken down by category.

### `src/sara_brain/cortex/transformer/inference_synth.py`

- New CLI arg `--repetition-penalty FLOAT` (default 1.2).
- New CLI arg `--no-repeat-ngram-size N` (default 3) — bans any n-gram
  that has already appeared in the generated prose tail.
- Modify `synthesize_cluster`'s decode loop:
  - Before argmax/sample, divide logits of any token that appeared in
    the last `repetition_window` positions by `repetition_penalty`.
    (HuggingFace-style penalty applied in log space.)
  - After the candidate token is picked, optionally veto if it would
    close a repeating n-gram and pick the next best alternative.
- Both penalties default to off when their args sit at sentinel
  values (`repetition_penalty=1.0`, `no_repeat_ngram_size=0`) so v0
  output stays reproducible bit-for-bit.

### `src/sara_brain/cortex/transformer/synth_data.py`

- `--brain` becomes repeatable (`action='append'`). Walks each in
  turn; the per-brain vocab union builds one combined sidecar
  `<out>.vocab.json`.
- New `--augment-multiplier N` (default 2). For clusters with ≥2
  edges, emit `N` variants by shuffling edge order before rendering.
  The rendered prose changes (cluster ordering propagates through
  `render_edges`'s sentence combining), giving the trainer multiple
  legitimate (edges, prose) pairs from the same source.
- Existing sub-sampling for very-large clusters (`max_edges_per_cluster=12`)
  stays; augmentation interacts cleanly (each chunk gets its own
  shuffles).

## Reusable existing pieces

- [`generate_examples()`](../src/sara_brain/cortex/transformer/synth_data.py)
  and `cluster_by_subject()` — multi-brain wrapper just calls them
  per brain.
- [`build_brain_vocab()`](../src/sara_brain/cortex/transformer/vocab_synth.py)
  — already takes any iterable of words; multi-brain corpus
  extraction fits without modification.
- `project_base_into_synth()` in train_synth.py — already pads
  embeddings; no change needed for the larger vocab.
- Model's `loss_mask` support
  ([model.py:113](../src/sara_brain/cortex/transformer/model.py#L113))
  — unchanged; new training rows use it the same way.

## Order of operations

1. Save plan + commit (this commit).
2. Implement train_synth.py changes (`--unfreeze-top-n`, repeatable
   `--pairs`).
3. Implement inference_synth.py repetition penalty + n-gram veto.
4. Implement synth_data.py multi-brain `--brain` + augmentation.
5. Generate v1 corpus from all available brains. Expect ~3000–10000
   rows depending on brain sizes and augmentation.
6. Train v1 with `--unfreeze-top-n 2 --steps 3000`. Single ckpt for
   the trained behavior.
7. Sample with repetition penalty enabled, compare against v0 output
   on the same clusters with the same seed.
8. Bundle slices 4g+4h+4i in one commit (small per-file diffs, all
   part of the same v1 quality pass).
9. Update v033 v0 status section to reflect v1 results.

## Verification

End-to-end, in this order:

### 1. Multi-brain corpus generation

```bash
.venv/bin/python -m sara_brain.cortex.transformer.synth_data \
  --brain /tmp/sara_demo.db \
  --brain aptamer_full.db.bak \
  --brain brain.db.bulk_reteach_backup \
  --brain brain.db.flatten_lift_backup \
  --brain brain.db.hand_curated_nopartof \
  --brain brain.db.hand_faithful \
  --serialize-out /tmp/synth_pairs_v1.jsonl \
  --augment-multiplier 2 --max-seq 256
```

Expect: `written` significantly > 600; `vocab_size` larger than 1170;
`unk_in_corpus` still 0.

### 2. v1 training

```bash
.venv/bin/python -m sara_brain.cortex.transformer.train_synth \
  --l2-ckpt src/sara_brain/cortex/checkpoints/l2_en_003000.pt \
  --pairs /tmp/synth_pairs_v1.jsonl \
  --unfreeze-top-n 2 --steps 3000
```

Expect:
- Log line `[synth] trainable params: ~14M+ (tok_embed + top 2 blocks)`
- Final dev_loss meaningfully below 4.13
- Train/dev gap narrower than v0's 1.83 vs 4.13

### 3. v1 inference comparison

Run inference_synth against the v0 ckpt and the v1 ckpt on the same
8 random clusters with the same seed:

```bash
.venv/bin/python -m sara_brain.cortex.transformer.inference_synth \
  --ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_synth_pairs_002000.pt \
  --brain /tmp/sara_demo.db --n 8 --seed 42 \
  > /tmp/v0_out.txt

.venv/bin/python -m sara_brain.cortex.transformer.inference_synth \
  --ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_synth_pairs_v1_*.pt \
  --brain /tmp/sara_demo.db --n 8 --seed 42 \
  --repetition-penalty 1.2 --no-repeat-ngram-size 3 \
  > /tmp/v1_out.txt

diff -y /tmp/v0_out.txt /tmp/v1_out.txt
```

Pass criteria (judged by eye):
- No more `rna rna rna rna` loops on the long clusters
- At least 4 of 8 clusters produce coherent multi-clause prose
- Short-cluster outputs stay at v0 quality or better

### 4. No-API reproducibility check

Confirm the entire v1 build path uses zero API calls:

```bash
grep -rn "anthropic\|openai" src/sara_brain/cortex/transformer/ | grep -v "\.venv"
```

Should return nothing.

### 5. Backward compatibility

v0 ckpt still loads cleanly under the modified `inference_synth`
(default `--repetition-penalty 1.0` matches v0 behavior).

## Out of scope (deferred to slice 5+)

- Chat REPL `--use-hamrobysum` flag — gated on v1 quality clearing
  the v032 template baseline.
- Eval script (slice 4f) — same gating.
- Per-brain vocab adapters or shared synth across brains with vocab
  routing — bigger architecture change; revisit if v1 quality still
  insufficient after this slice.
- Path 2 (frontier distillation) — explicitly off the table per v033;
  only revisit if Path 1 tuning plateaus below the template baseline.
