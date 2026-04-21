# v023 — Session State 2026-04-19 / 2026-04-20

Report written before shutdown. Captures the current state of brain,
code, benchmarks, memory, and open questions so the next session can
resume without losing context.

---

## 1. Brain state on disk

| File | Contents | When |
|---|---|---|
| `brain.db` | **Current brain.** 47-chapter biology2e taught via grammar-expansion. 12,370 neurons, 44,936 paths, 38,312 segments, 933 learned verbs. | 2026-04-20 |
| `brain.db.reverse` | Reverse-direction mirror of current brain (subject → relation → object chain). Written in parallel to every teach. | 2026-04-20 |
| `brain.db.ch10_expanded` | Prior state: ch10-only taught via grammar-expansion. 721 neurons, 698 paths. | 2026-04-20 |
| `brain.db.hand_faithful` | Ch10 taught sentence-by-sentence by Claude-as-surrogate. 1039 neurons, 371 paths. Session earlier. | 2026-04-19 |
| `brain.db.openie_verbs` | Two-pass OpenIE + verb-teach. 844 neurons, 358 paths. | 2026-04-19 |
| `brain.db.openie_no_verb_teach` | Single-pass OpenIE, no verb teaching. 585 neurons, 280 paths. | 2026-04-19 |
| `brain.db.hand_curated_nopartof` | Curated hand-teach after removing `_link_sub_concepts`. 940 neurons, 629 paths. | 2026-04-19 |
| `brain.db.with_part_of_leak` | Pre-part_of-removal hand-curated teach. Has the `cell → muscle_cell` sideways leak. | 2026-04-19 |
| `brain.db.bulk_reteach_backup` | Bulk reteach via parser (broken pattern per hand_teach_not_bulk memory). | 2026-04-19 |
| `brain.db.bulk_ch10_nopartof` | Bulk ch10 after removing part_of. | 2026-04-19 |
| `brain.db.flatten_lift_backup` | Original flatten + lift from biology2e.db. 48,575 neurons, 358 paths. | 2026-04-19 |

`biology2e.db` — unchanged source of truth, 25,315 paths across 48
region-prefixed chapters. Never modified.

---

## 2. Code changes this session

### New modules

- `src/sara_brain/storage/hierarchical_backend.py` — three-tier SQLite (v022 shelved)
- `src/sara_brain/tools/flatten_monolithic.py` — biology2e → flat brain.db
- `src/sara_brain/tools/lift_compound_neurons.py` — spaCy compound → IS-A lift
- `src/sara_brain/tools/migrate_chapters_to_hierarchy.py` — shelved path, kept
- `src/sara_brain/core/query_resolver.py` — compound-aware query seeds
- `src/sara_brain/core/wavefront_scorer.py` — confluence scoring, negation-aware, witness-dilution
- `src/sara_brain/core/reverse_learner.py` — parallel reverse-direction writer
- `src/sara_brain/parsing/grammar_expansion.py` — spaCy dep decomposition → atomic SVO
- `src/sara_brain/teaching/prompts.py` — LLM teacher cascade prompt templates (shelved as failed experiment)
- `src/sara_brain/teaching/ollama_client.py` — HTTP client for Ollama
- `src/sara_brain/teaching/cascade.py` — LLM teacher cascade orchestrator
- `src/sara_brain/teaching/openie.py` — rule-based OpenIE extraction + sensor filters

### Modified existing code

- `src/sara_brain/core/recognizer.py` — IS-A edges excluded from wavefront propagation; added `inherit_definitions()` for upward IS-A walk at scoring time.
- `src/sara_brain/core/learner.py` — compound IS-A emission on ingest; `_link_sub_concepts` **removed** (was creating bogus `cell → part_of → muscle_cell` sideways leaks).
- `src/sara_brain/core/brain.py` — added `teach_verb()`, `teach_expanded()`; wired reverse-learner; parser now consults `neuron_repo.is_verb`.
- `src/sara_brain/storage/neuron_repo.py` — added `is_verb(label)` query.
- `src/sara_brain/parsing/statement_parser.py` — compound detection; learned-verb consultation; `verb_unknown` field on ParsedStatement; removed the verb-suffix fallback that silently accepted any -s/-ed word.

### New benchmarks / tools

- `benchmarks/run_teacher_cascade.py` — LLM teacher cascade (failed experiment, kept as evidence)
- `benchmarks/run_openie_cascade.py` — OpenIE sensor cascade
- `benchmarks/teach_openie_triples.py` — triples → teach
- `benchmarks/teach_openie_triples_with_verbs.py` — two-pass with verb registration
- `benchmarks/reteach_flat.py` — reteach from fact files
- `benchmarks/reteach_curated.py` — reteach from curated SVO list
- `benchmarks/teach_ch10_expanded.py` — ch10 via grammar expansion
- `benchmarks/teach_all_chapters_expanded.py` — all 47 chapters via grammar expansion
- `benchmarks/run_wavefront_ch10.py` — benchmark driver (now uses `pick_choice`)

### Commits this session (branch: `feature/hierarchical-concept-storage`)

```
653e055 Scorer: negation-aware ranking + dilution-based witness weighting
bc6f6da Grammar expansion + reverse-direction mirror + bidirectional scorer
974bca0 Hand-faithful ch10 teach — sentence-by-sentence walk of ch10_facts.txt
64948ff Teacher cascade + OpenIE pipeline + learned-verb mechanism
0372290 Wavefront scorer + Ch10 baseline — 9/33 unoptimized
d429641 Compound-aware query resolver + polysemy smoke tests
fa3f484 Flat brain.db + IS-A hierarchy + non-propagating IS-A semantics
```

---

## 3. Benchmark state (current brain.db)

### Ch10 (33 questions, biology2e chapter 10 test bank)

| Run | correct | wrong | abstain | tie | coverage | precision |
|---|---:|---:|---:|---:|---:|---:|
| Flatten+lift brain (2026-04-19 early) | 9 | 23 | 0 | 1 | 97.0% | 27.3% |
| Bulk-taught, `part_of` removed | 0 | 2 | 30 | 2 | 6.1% | 0.0% |
| Hand-curated (with `part_of` leak) | 5 | 12 | 11 | 5 | 51.5% | 29.4% |
| Hand-curated (no `part_of` leak) | 2 | 6 | 22 | 3 | 24.2% | 25.0% |
| OpenIE + verb-teach | 0 | 2 | 28 | 3 | 6.1% | 0.0% |
| Hand-faithful walk of ch10_facts | 1 | 5 | 25 | 2 | 18.2% | 16.7% |
| Grammar expansion ch10-only | 8 | 12 | 13 | 0 | 60.6% | 40.0% |
| **Grammar expansion ALL 47 chapters + scorer fixes (current)** | **8** | **24** | **1** | **0** | **97.0%** | **25.0%** |

### MMLU biology (310 questions, current brain)

- correct: 74
- wrong: 203
- abstained: 8
- ties: 25
- coverage: 89.4%
- precision: 26.7%

### Benchmark artifacts in `benchmarks/`

- `ch10_wavefront_hand_faithful.json` — hand-faithful result
- `ch10_wavefront_expanded.json` — ch10-only expansion
- `ch10_wavefront_full.json` — current (all chapters + scorer fixes)
- `ch10_wavefront_bidir.json`, `ch10_wavefront_openie_verbs.json`, `ch10_wavefront_nopartof.json`, `ch10_wavefront_ch10only.json`, `ch10_wavefront_chapters.json`, `ch10_wavefront.json` — intermediate runs
- `mmlu_wavefront_full.json` — current MMLU
- `mmlu_wavefront_openie_verbs.json`, `mmlu_wavefront_reteach.json`, `mmlu_biology_chapters.json` — prior

---

## 4. Architectural findings

### What worked

- **Compound-aware query resolver** — multi-word compounds that exist as neurons resolve to a single high-power seed; others fall back to bare-token seeds.
- **IS-A edges excluded from wavefront propagation** — prevents polysemy leakage via shared head nouns (`nerve_cell` can't leak to `battery_cell` via `cell`).
- **`inherit_definitions()` upward walk** — ancestors reachable for definition lookup without contaminating propagation.
- **Learned-verb mechanism** — `brain.teach_verb(word)` registers a verb; parser consults innate ∪ learned.
- **Grammar expansion** — spaCy dep parse decomposes rich English into atomic SVO sub-facts (primary + adjective + prep phrase + adverb + relative clause). This was the single largest win (1 → 8 correct on ch10).
- **Bidirectional scorer** — wavefronts walk both incoming and outgoing edges so subject-seeded queries reach facts taught about them.

### What didn't work (kept as evidence, not reverted)

- **LLM teacher cascade** (llama3.2:1b → 3b → final). Both sizes injected content not in source (invented "Jurassic period" from a sea-urchin sentence; paraphrased "organism" as "animal"). Confirms the paper's thesis on new models. Artifacts in `benchmarks/cascade_smoke/`.
- **Hierarchical per-concept storage (v022)** — topic-grain (4849 DBs, query hung); chapter-grain (48 DBs, 14/23 on ch10). Underperformed monolithic. Shelved.
- **Raw textbook prose → Sara parser** — produces mangled multi-clause neurons ("The Mitotic Phase The mitotic phase is a multistep..." as a subject).
- **Margin-based abstain** — tested at 15% threshold, cost more correct (-5) than saved wrong. Reverted.
- **Hub-drop** (drop seeds with 5× median out-degree) — made Ch10 worse (8 → 6). Reverted.
- **Compound-hit bonus** — no measurable effect; compound matches don't fire often enough in the test set to move picks. Reverted.

### Open architectural questions

1. **Cross-chapter content bleed.** Full-biology teach gave score to every previously-abstained question, but the extra signal is typically noise. Transition matrix: 11 abstain→wrong, 3 correct→wrong, 2 wrong→correct, 1 abstain→correct. Net same correct count, many more wrong.
2. **Hub-dominated wavefronts.** Tokens like `cell`, `chromosome`, `organism` appear in every teaching and every question. Their wavefronts flood every choice with similar score. Witness-dilution helps but doesn't fully separate correct from wrong.
3. **NOT/EXCEPT questions.** Scorer detects negation and picks argmin. Works when Sara has signal on the non-outlier choices, but if she has no content for the topic at all, argmin picks the first zero-score choice arbitrarily.
4. **Multi-word label atomicity.** When "chromosomes toward opposite poles" is taught as a single noun-phrase object, it becomes a leaf neuron with no connection to the bare `chromosome` atom. Grammar expansion fixed this for new teaches but the prior hand-faithful brain still has these leaves.

---

## 5. Memory updates this session

New feedback memories (in `/Users/grizzlyengineer/.claude/projects/-Users-grizzlyengineer-repo-sara-brain/memory/`):

- `user_experimental_method.md` — Jennifer runs known-bad experiments deliberately; measurement beats consensus; JKD "I am the experience, not the experiencer"
- `feedback_finish_before_evaluate.md` — don't interpret benchmarks mid-plan
- `feedback_technician_not_engineer.md` — execute the simple spec, no optional knobs/caches/trade-off commentary
- `feedback_hand_teach_not_bulk.md` — teaching is per-fact judgment, not a batch script
- `feedback_avoid_qwen.md` — qwen imposes training; default to llama family for small-model roles
- `feedback_failed_experiments_are_data.md` — predicted failures are contributions, not mistakes
- `feedback_teach_faithfully_sentence_by_sentence.md` — walk every sentence in order; no judgment on importance

New project memories:

- `project_hierarchy_shelved.md` — v022 shelved with evidence
- `project_brain_db_checkpoint.md` — flat brain.db checkpoint
- `project_grammar_only_cortex.md` — north-star: train a small LLM on grammar alone, no facts

---

## 6. How to resume

### If you want to keep iterating on the current brain

Current `brain.db` is the full-biology expansion-taught brain. Benchmarks are stored. To re-run:

```bash
.venv/bin/python benchmarks/run_wavefront_ch10.py --db brain.db \
    --questions benchmarks/ch10_test_questions.json
```

### If you want to rebuild from scratch

```bash
# Wipe current
mv brain.db brain.db.old
rm -f brain.db-shm brain.db-wal brain.db.reverse*

# All 47 chapters, atomic expansion
.venv/bin/python benchmarks/teach_all_chapters_expanded.py --db brain.db
# (~3 min)

# Ch10 benchmark
.venv/bin/python benchmarks/run_wavefront_ch10.py --db brain.db \
    --questions benchmarks/ch10_test_questions.json \
    --output benchmarks/ch10_latest.json

# MMLU biology
.venv/bin/python benchmarks/run_wavefront_ch10.py --db brain.db \
    --questions benchmarks/mmlu_biology_full.json \
    --output benchmarks/mmlu_latest.json
```

### If you want a specific slice

- `teach_expanded.py --db X --source benchmarks/biology2e_facts/ch10_facts.txt` for ch10 only
- Any `.txt` file with one sentence per line works

### Known failing tests (expected, don't fix without user direction)

- `tests/test_refutation.py::test_why_marks_refuted_paths` — pre-existing failure before this session's changes. 378/379 pass (1 skipped, 1 deselected).

---

## 7. Outstanding design decisions Jennifer may want to revisit

1. **Grammar-only cortex** — saved as north star in `project_grammar_only_cortex.md`. Training a small LLM (50–500M params) on syntax only, no world facts. She floated the i9 / 3080 box for training.
2. **Hardware deployment shape** — M2 mini for Sara Brain + dev, i9/3070 as Ollama inference server over LAN once the 3070 box is brought online. Windows updates + data backup required first.
3. **Reverse-direction DB** — currently written to `brain.db.reverse` in parallel. Not queried yet. Future work could use it for subject-seeded forward traversal that bypasses the primary direction's backward-walk requirement.
4. **Benchmark expectations** — current 25–40% precision on Ch10 is well below the paper's 80% result. The paper used a different brain (45 hand-taught facts, MMLU-specific) and a different benchmark. Whether a full-biology brain should hit comparable precision on the broader ch10 test is unresolved.

---

## 8. Git state

Branch: `feature/hierarchical-concept-storage`

Commits ahead of origin: 7 (need `git push` when ready)

Working tree clean except for:
- Uncommitted: `tests/test_polysemy_cell.py` may need verification on a fresh brain
- Several untracked `.json`, `.aux`, `.log` artifacts that are gitignored or build output

Never-delete rule per `feedback_never_delete_branches`: the `feature/hierarchical-concept-storage` branch holds all shelved work; do not delete.
