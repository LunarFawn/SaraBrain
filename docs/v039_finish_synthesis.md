# v039 — finish synthesis (v037.1 + article post-processor + chat REPL integration)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v037_layered_synth_architecture.md](v037_layered_synth_architecture.md),
[v038_state_and_directions.md](v038_state_and_directions.md)
**Path:** option 1 from v038 — finish synthesis as a real
user-visible feature in the chat REPL.

## Context

v037 shipped the layered Core + EN architecture. Three concrete
gaps remain between EN-as-it-stands and EN-as-shippable:

1. **Predicate vocabulary.** EN trained on the 12 relations in
   `_RELATIONS_POOL`. Real brains use ~1928 distinct relations
   across the repo; the top ~50 are real English verbs (`contains`,
   `causes`, `activates`, `triggers`, `forms`, ...). Outside the
   trained 12, EN substitutes a pool verb instead of the right one
   (e.g. `forms → requires`).
2. **Article heuristic gap.** v032 templates apply
   `_maybe_article` *before* formatting. HamRoby-Sum's slot expansion
   happens at inference time, after the model has emitted
   `<C0> is a <C1> .` → expanded to `Multicellular organism is a
   organism.` (should be `is an organism`).
3. **Chat REPL integration.** `chat.py` calls
   `synthesizer.synthesize()` in five places; v032 templates are
   still the runtime renderer.

## The three slices

### Slice 1 — extend `_RELATIONS_POOL` and retrain EN

Edit `papers/instrument_validation/generate_synthetic_substrate.py`
to extend `_RELATIONS_POOL` from 12 to ~50 entries (top by
cross-brain frequency, hand-curated).

Verbs to add (after filtering noun-shaped noise like `chromatids`,
`fibers`, `ring` and substrate plumbing like `refutation_of`,
`refutation_status`, `describes`):

```
contains has causes activates becomes triggers forms begins
releases breaks contributes occurs encodes binds creates remains
prevents during starts maintains detects includes provides lacks
ensures joins disrupts means reduces accumulates monitors
comprises controls undergo separates determines involves increases
adds form associates precedes gets allow throughout within
applies_to stores supports holds
```

Already in pool: `is_a, has_property, part_of, produces, requires,
interacts_with, used_for, described_by, predicts, contains, opposes,
enables`.

After the edit, user reruns:
```
./scripts/build_layered_corpus.sh
PAIRS=/tmp/synth_pairs_en.jsonl CKPT_NAME=hamroby_sum_en STEPS=2500 \
  SESSION=sara-synth-en \
  RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_core_002500.pt \
  ./scripts/train_hamrobysum.sh
```

Core stays unchanged (its training never touched the pool because
of `--nonsense-relations`). Only EN retrains, ~10 min on a 3070.

### Slice 2 — article post-processor

In [src/sara_brain/cortex/transformer/inference_synth.py](../src/sara_brain/cortex/transformer/inference_synth.py),
add `_fix_articles(text: str) -> str` that scans detokenized output
for `\b(a|an)\s+(\w+)` and swaps `a ↔ an` based on vowel-onset of
the following word. Apply inside `synthesize_cluster` *after*
`_detokenize`.

Conservative: only fix clear `a → an` / `an → a` mismatches.
Doesn't insert or remove articles — the model's emission decides
"should there be an article here"; we only fix obvious vowel
agreement.

### Slice 3 — chat REPL `--use-hamrobysum`

In [src/sara_brain/cortex/transformer/chat.py](../src/sara_brain/cortex/transformer/chat.py):

- Add CLI args `--use-hamrobysum` + `--hamrobysum-ckpt`
  (default `src/sara_brain/cortex/checkpoints/hamroby_sum_en_002500.pt`).
- Load the model at session start when the flag is set, via
  `inference_synth.load_synth_checkpoint`.
- Add `_synthesize(self, question, gathered)` method:
  1. If HamRoby-Sum loaded: parse `gathered` to edges via
     `synthesizer.parse_edges_from_gathered`, cluster by subject,
     run each cluster through `inference_synth.synthesize_cluster`,
     concatenate.
  2. If any cluster's output is empty / degenerate, fall back to
     v032 `synthesizer.synthesize` for that cluster.
  3. If HamRoby-Sum not loaded: pass through to v032 directly
     (current behaviour).
- Replace all 5 call sites of `synthesize(...)` with
  `self._synthesize(...)`.

## Reused functions (no duplication)

- `inference_synth.load_synth_checkpoint` — loads + verifies
  `vocab_flavor=v035-generic`.
- `inference_synth.synthesize_cluster` — takes edges, emits prose.
- `synthesizer.parse_edges_from_gathered` — parses brain_explore
  output into Edge objects.
- `synth_data.cluster_by_subject` — groups edges by subject.
- `synthesizer.synthesize` (v032 templates) — fallback path.

## Order of operations

1. Save plan + commit (this commit).
2. **Slice 1a:** edit `generate_synthetic_substrate.py`. Single
   targeted edit. Commit.
3. **Slice 1b:** *user runs* `./scripts/build_layered_corpus.sh`
   (CPU, ~minute) then the EN retrain command in tmux (~10 min).
4. **Slice 1c:** I run inference comparison — predicates that were
   substituted should now resolve correctly. Brief commit with the
   findings.
5. **Slice 2:** edit `inference_synth.py` for article
   post-processor. Test inline. Commit.
6. **Slice 3:** edit `chat.py` for `--use-hamrobysum`. Smoke test
   the chat REPL with the flag. Commit.
7. Update v028 status section: mark synth slice 4 done.

## Verification

1. **Slice 1c** — sample EN on demo brain. Predicates that were
   `<unk>` in v037 (`formed`, `applies_to`, `forms`) resolve
   correctly. Wrong-predicate substitution rate drops below 20% on
   a sample of 20 clusters.
2. **Slice 2** — `is a organism` → `is an organism`, `is a apple`
   → `is an apple`. Conservative: never inserts or removes
   articles, only swaps a↔an.
3. **Slice 3** — `chat.py --brain /tmp/sara_demo.db --cortex-router
   --use-hamrobysum` answers a "what is X" question with
   HamRoby-Sum prose. Without the flag still uses v032 templates
   (no regression).
4. **No-API reproducibility** — `grep -rn "anthropic\|openai"
   src/sara_brain/cortex/transformer/` returns nothing.

## Out of scope

- Open-class English noun mining. The slot mechanism handles
  content; only predicate verbs need coverage.
- Multi-language overlays (`hamroby_sum_es`, etc.). Recipe in v037;
  separate slice.
- Long-cluster (`more` cluster) improvements. Long clusters fall
  back to template rendering via slice 3 fallback path.
- Eval script. Same gating as before; do it after v039 ships and
  competes meaningfully against v032 templates.
