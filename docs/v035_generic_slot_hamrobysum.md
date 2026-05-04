# v035 — Generic slot-based HamRobySum (correcting the v033 per-brain leak)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v033_reproducible_hamroby_loop.md](v033_reproducible_hamroby_loop.md),
[v034_hamrobysum_v1_quality.md](v034_hamrobysum_v1_quality.md)
**Supersedes:** the per-brain vocab assumption in v033 / v034

## The mistake

v033's "Path 1" loop made HamRobySum a *per-brain* model: each brain
contributed its content words into the model's emission vocabulary,
giving you `vocab_size = 192 (base) + N (this brain's content words)`.
v034 tried to scale this by training on 9 brains at once → vocab grew
to 13991 → model couldn't form strong word-level priors for any of
them → output collapsed to `rna rna rna rna...` regardless of
repetition penalty.

The architectural error: **the per-brain vocab forces per-brain
models**. That breaks the entire v028/v033 commitment. L1 is universal,
L2 is per-language (universal across users of that language), L3
substrate is per-user — but HamRobySum was per-substrate, which is
strictly *narrower than* per-user (one user's brain evolves, every
`/teach` invalidates the model). Wrong.

## The correction — slot-based generic vocabulary

HamRobySum's vocabulary contains **no substrate content words at
all**. The model learns structural patterns over a small fixed
vocabulary; substrate labels get inserted at inference time via slot
expansion.

Vocabulary (final, fixed, never grows per-brain):

| Group | Source | Count |
|---|---|---|
| L1 structural tokens | `vocab.py` | 76 |
| L2-en function words | `vocab_en.py` | 99 |
| Synth delimiters | this file | 10 |
| Synth punctuation | this file | 7 |
| **Slot tokens (NEW)** `<C0>` ... `<C31>` | this file | 32 |
| **Substrate verbs (NEW)** common predicate verbs | this file | ~30 |
| **Total** | | **~254** |

No `build_brain_vocab` call. No per-example vocab extension. **One
trained model works against every brain**, including brains it has
never seen.

## Mechanics

### Training

Every (edges, prose) example is **slot-substituted** before
serialization:

1. Collect every distinct substrate label that appears in the
   cluster (sources, targets). Assign each a slot id `<C0>`, `<C1>`,
   ... in encounter order. Maximum 32 distinct labels per cluster.
2. In the facts side, replace `e.src` / `e.tgt` token sequences with
   the corresponding `<Cn>` token.
3. In the prose side, find any case-insensitive occurrence of those
   labels and substitute the same `<Cn>`. (Sort by length descending
   so longer labels match before shorter ones — prevents `rna` from
   eating `rna stability`.)
4. Encode the result against the fixed `TOK2ID_SYNTH`. No
   brain-extended vocab, no UNK explosion.

The prose might still contain non-slot content the model needs to
emit literally — articles, predicate verbs, punctuation. Those are
all already in vocab_synth.

Example.

Edge cluster:
```
src='inertia in rna' rel='is' tgt='tendency to maintain current state' [attr]
src='ribosome'       rel='part_of' tgt='cell'
```

Per-cluster mapping (assigned per-example):
```
<C0> = "inertia in rna"
<C1> = "tendency to maintain current state"
<C2> = "ribosome"
<C3> = "cell"
```

Serialized facts:
```
<facts>
  <subj> <C1> <pred> is <obj> <C0> <attr> <edge_sep>
  <subj> <C2> <pred> part of <obj> <C3> <edge_sep>
<prose>
```
(src and tgt swapped for the attr edge because that's how
`render_edges` flips it via `_ATTR_TEMPLATES`.)

Slot-substituted prose target:
```
<C0> is a <C1> . <C2> is part of <C3> .
</prose>
```

The model learns: given that frame, emit `<Cn>` slots in the right
places, glued by `is`, `a`, `part of`, periods.

### Inference

1. Take an edge cluster.
2. Build the same per-cluster slot mapping.
3. Format the facts prefix with slots.
4. Decode prose tokens until `</prose>`.
5. Replace each `<Cn>` token in the decoded output with the
   corresponding substrate label.
6. Detokenize → final string.

Step 5 means substrate content **never enters the model's
weights** — it round-trips through the model purely as a position
marker. The model's job is "given these positions, produce a
grammatical sentence connecting them in this order."

This is the v028 thesis cleanly realized: knowledge in substrate,
structure in weights.

## What this changes from v033/v034

| File | Change |
|---|---|
| `vocab_synth.py` | Add 32 slot tokens + ~30 substrate verbs. **Drop `build_brain_vocab` entirely** — never used again. `VOCAB_SIZE_SYNTH` becomes a fixed `~254`, never per-brain. |
| `synth_data.py` | Add `slot_substitute()` helper. `serialize_example` calls it before encoding. Saves `slot_mapping` in the JSONL row alongside `input_ids` / `loss_mask`. `write_serialized_jsonl` no longer builds or writes a sidecar `.vocab.json`. |
| `train_synth.py` | Drop the brain-vocab loading code. Vocab is just `vocab_synth.VOCAB_SIZE_SYNTH`. The projection logic stays — only the per-brain expansion goes. |
| `inference_synth.py` | Add slot expansion: build mapping at start, decode, swap `<Cn>` tokens for their substrate strings before detokenizing. |
| Trainer `--ckpt-every` defaults | Bumped to 5000 across all trainers (was 500) so we don't recreate the v034 100+ GB intermediate-ckpt sprawl. Single-checkpoint runs become the default; users can lower if they actually want per-step snapshots. |

The v0 (`hamroby_sum_synth_pairs_002000.pt`) and v1
(`hamroby_sum_v1_003000.pt`) checkpoints are **deprecated**. Their
per-brain vocab format won't load against v035's generic vocab; they
stay on disk as historical artifacts until the v035 model proves
better, then can be deleted (T3 in the disk cleanup).

## Substrate verb list (initial cut)

Drawn from `synthesizer._TEMPLATES` and `_ATTR_TEMPLATES` predicate
slots — the verbs that appear in rendered prose:

```
measures evaluates assesses leverages incorporates integrates
validates validate offers states means stands acts applies focuses
indicates produces influences simulates simulate provide provides
drops requires defined related analogous synonym known
abbreviation expressed caused results associated described
```

~30 closed-ish substrate-relevant verbs. Same append-only stability
rule as the function-word list in `vocab_en.py`.

## Order of operations

1. Save this doc + commit.
2. Modify `vocab_synth.py` — add slots + verbs, drop `build_brain_vocab`.
3. Modify `synth_data.py` — `slot_substitute`, drop brain-vocab pipeline.
4. Modify `train_synth.py` — drop brain-vocab handling, bump
   `--ckpt-every` default.
5. Modify `inference_synth.py` — slot expansion at decode.
6. Bump `--ckpt-every` defaults in `train.py` and `train_l2.py` too
   (same justification).
7. Single bundled commit for the code changes (small per-file diffs,
   one architectural shift).
8. **You launch** the v2 retrain on the multi-brain corpus.
9. Eval on a brain not seen during training (the genericness proof).

## Verification

### 1. Vocab fixed-size sanity

```bash
.venv/bin/python -c "
from sara_brain.cortex.transformer.vocab_synth import VOCAB_SIZE_SYNTH
print(f'VOCAB_SIZE_SYNTH = {VOCAB_SIZE_SYNTH}')
assert VOCAB_SIZE_SYNTH < 300, 'vocab should not blow up — generic'
"
```

### 2. Slot substitution round-trip

```bash
.venv/bin/python -c "
from sara_brain.cortex.transformer.synth_data import slot_substitute
prose = 'Inertia in rna is a tendency to maintain current state.'
mapping = {'<C0>': 'inertia in rna', '<C1>': 'tendency to maintain current state'}
out = slot_substitute(prose, mapping)
print(repr(out))
assert '<C0>' in out and '<C1>' in out, 'slots not substituted'
assert 'inertia' not in out.lower(), 'original content leaked through'
"
```

### 3. Multi-brain corpus generation (NO sidecar vocab)

```bash
.venv/bin/python -m sara_brain.cortex.transformer.synth_data \
  --brain /tmp/sara_demo.db --brain aptamer_full.db.bak \
  --serialize-out /tmp/synth_pairs_v2.jsonl \
  --augment-multiplier 2 --max-seq 256
```

Expect: same row count, no `.vocab.json` sidecar, `vocab_size`
reported as the fixed `VOCAB_SIZE_SYNTH`.

### 4. v2 training (you launch)

```bash
.venv/bin/python -m sara_brain.cortex.transformer.train_synth \
  --l2-ckpt src/sara_brain/cortex/checkpoints/l2_en_003000.pt \
  --pairs /tmp/synth_pairs_v2.jsonl \
  --unfreeze-top-n 2 --steps 3000 \
  --ckpt-name hamroby_sum_v2
```

Expect:
- Trainable params **smaller** than v1's 25M (vocab is ~254 not
  13991, so tok_embed is much smaller)
- Final dev_ppl much lower than v1's 66 (smaller vocab is easier to
  predict)

### 5. Cross-brain inference

```bash
.venv/bin/python -m sara_brain.cortex.transformer.inference_synth \
  --ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_v2_*.pt \
  --brain brain.db.hand_faithful \
  --n 8 --seed 42
```

The model was *not trained* against `hand_faithful`, but if generic
works it should still produce reasonable prose. **This is the
genericness test.**

### 6. No per-brain artifact regressions

```bash
grep -rn "build_brain_vocab\|brain_vocab\b\|.vocab.json" \
  src/sara_brain/cortex/transformer/ | grep -v "\.venv"
```

Should find no callers of `build_brain_vocab`, no references to the
sidecar vocab file. Old code paths fully removed.

## Out of scope

- The "go bigger" question (300M / 500M / 1B). v035 keeps current
  125M sizing. If generic at 125M clears the v032 template baseline,
  we may not need to scale. If it doesn't, scale next slice.
- Predicate slotting (`<P0>`, `<P1>`). v035 emits predicate verbs as
  literal English from a fixed list. If this proves limiting (rare
  domain-specific verbs not in the list), revisit.
- Chat REPL `--use-hamrobysum` — same gating as before: needs to
  beat templates first.
