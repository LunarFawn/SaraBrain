# v036 — Curriculum-train HamRobySum on synthetic nonsense substrates

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v035_generic_slot_hamrobysum.md](v035_generic_slot_hamrobysum.md),
[papers/sara_as_instrument_paper_rev7.md](../papers/sara_as_instrument_paper_rev7.md) §3
**Reuses:** [papers/instrument_validation/generate_synthetic_substrate.py](../papers/instrument_validation/generate_synthetic_substrate.py)

## Why

v035 made HamRobySum architecturally generic (substrate content as
slots, never in weights). v2 (`hamroby_sum_v2_003000.pt`) confirmed
it works — clean prose on `brain.db.drifted_s1` (a brain it never
saw): `Sea urchin is a urchin.`, `Cell division is a division.`

But "trained on real brains" still has a confound. Even with slot
substitution, the model sees real English substrate-prose content
during training (substrate verbs, predicate phrases). It might be
implicitly learning real-world co-occurrence patterns — `<Cn>` slots
in rna-flavored contexts tend to combine with `<Cm>` slots about
stems, because the templated prose has those co-occurrences.

The instrument paper proposed the cleanest possible test of
architectural orthogonality: **synthetic substrates with
pronounceable nonsense-word labels** (`zilkrap`, `bortle`, `milvon
doplis`, ...). By construction, no model has been or could be
trained on them — they didn't exist before the substrate was
generated.

If a HamRobySum trained ONLY on synthetic nonsense substrates
produces clean prose for real brains it has never seen, the v035
architectural claim is proven beyond any leak concern.

## Curriculum (option B)

Three training phases, each resuming from the previous phase's
checkpoint. Anti-forgetting: each phase trains on a corpus that
*includes the prior phase's data plus new harder substrates*, so the
model doesn't unlearn the simpler patterns when it sees the harder
ones.

| Phase | Substrates trained on | Bucket sizes |
|---|---|---|
| 1 | small only | 60 substrates × (10 concepts, 30 triples) |
| 2 | small + medium | + 30 substrates × (30 concepts, 80 triples) |
| 3 | small + medium + large | + 10 substrates × (100 concepts, 250 triples) |

Each phase: ~1500–2000 steps starting from the prior ckpt. The
sizing is conservative; if quality plateaus we increase steps per
phase or add more substrates per bucket.

## Code changes

**`train_synth.py`** — one new flag:

- `--resume-from PATH` — load directly from a prior synth checkpoint
  (skip L2-en projection since the vocab is already correct). Used by
  phase 2 and phase 3.
- Existing `--l2-ckpt` path remains for cold-start (phase 1).
- Mutually exclusive: pass exactly one of `--l2-ckpt` / `--resume-from`.

**`scripts/build_synthetic_corpus.sh`** (new):

Generates `N_small + N_medium + N_large` synthetic substrates in
`/tmp/synth_brains/` (deterministic seeds for reproducibility), then
builds three cumulative JSONLs:

- `/tmp/synth_pairs_phase1.jsonl` — small substrates only
- `/tmp/synth_pairs_phase2.jsonl` — small + medium substrates
- `/tmp/synth_pairs_phase3.jsonl` — small + medium + large substrates

CPU-only, runs in seconds.

**`scripts/train_hamrobysum.sh`** — already takes `PAIRS=...` and
`CKPT_NAME=...` env vars; just needs an additional `RESUME_FROM=...`
passthrough so phases 2/3 can chain. Small change.

## Order of operations

1. Save plan + scripts + train_synth resume support → commit.
2. **You run** `./scripts/build_synthetic_corpus.sh` (CPU).
3. **You run phase 1** in tmux:
   ```
   PAIRS=/tmp/synth_pairs_phase1.jsonl \
   CKPT_NAME=hamroby_sum_v3_phase1 \
   STEPS=1500 \
   SESSION=sara-synth-p1 \
   ./scripts/train_hamrobysum.sh
   ```
4. **You run phase 2** (resumes from phase 1):
   ```
   PAIRS=/tmp/synth_pairs_phase2.jsonl \
   CKPT_NAME=hamroby_sum_v3_phase2 \
   STEPS=2000 \
   SESSION=sara-synth-p2 \
   RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_v3_phase1_001500.pt \
   ./scripts/train_hamrobysum.sh
   ```
5. **You run phase 3** (resumes from phase 2):
   ```
   PAIRS=/tmp/synth_pairs_phase3.jsonl \
   CKPT_NAME=hamroby_sum_v3_phase3 \
   STEPS=2000 \
   SESSION=sara-synth-p3 \
   RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_v3_phase2_002000.pt \
   ./scripts/train_hamrobysum.sh
   ```
6. When done, ping me; I run inference on the same benchmark
   clusters used for v0/v1/v2 plus the cross-brain genericness test
   on `brain.db.drifted_s1`.

## Verification

Pass criteria for v3:

1. Each phase trains cleanly. dev_loss decreases or holds across
   phases (no catastrophic forgetting).
2. v3 produces clean prose on the demo brain — at least as good as
   v2 on the same 8 clusters.
3. v3 produces clean prose on `brain.db.drifted_s1` — equal or
   better than v2's "Sea urchin is a urchin."
4. **The architectural proof:** v3 has demonstrably never seen any
   real English substrate content during training. If real-brain
   outputs are clean, "knowledge in substrate, not in weights" is
   validated as cleanly as it can be.

## Out of scope

- Mixing synthetic + real substrates in training. If v3 underperforms
  v2 on real brains, that's a v036.1 experiment.
- Per-bucket separate learning rate schedules. Default cosine schedule
  per phase is enough for v0.
- Chat REPL `--use-hamrobysum` integration. Still gated on v3
  clearing the v032 template baseline.
