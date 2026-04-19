# v022 — Hierarchical Per-Concept Storage

**Date:** 2026-04-19
**Status:** Infrastructure complete. Migration and end-to-end benchmark runs pending.

---

## What changed

Sara's storage moves from one monolithic SQLite file (`biology2e.db`, 252 prefixed tables for 48 regions) to a three-tier hierarchy of SQLite files:

```
brain_root/
    brain.db                        ← subjects index
    subjects/
        biology.db                  ← concepts index + concept_vocab + bridges
    concepts/
        biology/
            mitosis.db              ← self-contained neurons/segments/paths
            dna_replication.db
            heterotroph.db
            …
            _unclassified.db        ← debug bucket for unrouted facts
```

Grain is **topic** from day one — not chapter, not section. A concept is "mitosis", "DNA replication", "evaluate a blood sample". Chapters are a quirk of one source, not a reasoning boundary. Overlap is designed: a fact about anaphase lives in `mitosis.db`, `meiosis.db`, and `anaphase.db` simultaneously.

---

## Why

1. **Routing mismatch.** With all 48 chapters in one corpus, IDF over the full book is near-useless — "cell" appears in every chapter. Per-concept IDF computed only over the opened concept DBs is much more discriminating.
2. **Second-source reinforcement requires topic-grain.** When Wikipedia biology is ingested as a second source, its mitosis facts must land in the same `mitosis.db` as Biology 2e's for Sara's two-witness strength bump to fire. Chapter-grain would split the same content into different files across sources.
3. **Inspectability.** `sqlite3 concepts/biology/mitosis.db "SELECT source_text FROM paths LIMIT 20"` shows exactly what Sara knows about mitosis — nothing else.
4. **Lazy loading.** The scorer opens only 3–10 concept DBs per question instead of loading all 48 regions at startup.

---

## New files

| File | Purpose |
|---|---|
| `src/sara_brain/storage/hierarchical_backend.py` | `HierarchicalBackend` class. Manages brain.db, subjects/{s}.db, concepts/{s}/{c}.db. Provides `route_teach`, `route_query`, `concept_conn`, `update_concept_vocab`. LRU cap (256) prevents FD exhaustion with thousands of concept DBs. |
| `src/sara_brain/tools/__init__.py` | Makes tools a package. |
| `src/sara_brain/tools/extract_biology2e_topics.py` | One-shot topic extractor. Mines noun lemmas from per-chapter fact logs + section headings from raw textbook TOC. Writes to brain_root via backend. |
| `src/sara_brain/tools/migrate_to_hierarchy.py` | Fact-migration CLI. Reads each path in biology2e.db, routes it to target concept DBs, writes neurons/segments/paths/path_steps/segment_sources into each. Carries bridges. Builds concept_vocab after migration. |
| `tests/test_hierarchical_backend.py` | 22 tests: slugify, three-tier init, concept registration, route_teach, route_query, overlap routing, 10 hand-graded biology routing queries, repo parity, LRU FD safety. |

## Modified files

| File | Change |
|---|---|
| `src/sara_brain/core/brain.py` | `Brain(path)` detects a directory → sets `self.backend = HierarchicalBackend(path)`. Monolithic `.db` unchanged. |
| `benchmarks/run_spacy_ch10.py` | Added `load_concept_paths(conn, nlp)` (unprefixed concept DB variant of `load_region_paths`). Added `select_concepts_from_backend`. In `main()`: hierarchical mode detects `brain.backend is not None`, routes per question via `route_query`, loads only routed concept DBs, computes per-question IDF. |

---

## Key design details

### Two vocab tables (subject DB)

- **`concept_lemmas`** — trigger set for teach-time routing. Seeded from noun frequency across fact logs. `route_teach(lemmas)` intersects against this.
- **`concept_vocab`** — observed vocabulary built from the paths actually written into the concept DB. Used at query time by `route_query` for best-concept selection and per-question IDF. Rebuilt after migration via `_build_concept_vocab`.

### LRU connection cache

Opening ~6000 concept DB files simultaneously would exhaust OS file descriptors (each SQLite WAL DB uses 3 FDs). The backend keeps an `OrderedDict` cache capped at 256; the oldest open concept DB is closed on eviction. This is transparent to callers — `concept_conn()` always returns a valid connection.

### Fan-out write

A single path whose lemmas match `mitosis`, `chromosome`, and `cell_cycle` is written into all three concept DBs. ID remapping is done per destination because each concept DB has its own integer namespace. Upsert semantics (INSERT OR IGNORE on neurons and segments by label/triple) prevent duplicates when a concept accumulates paths from multiple regions.

### Unclassified bucket

Facts whose lemmas match no concept trigger are routed to `_unclassified.db`. This bucket is never lost and never silently dropped — it is inspectable after migration to tune trigger lemmas.

---

## Migration command (run after topic extraction)

```bash
# Step 1 — extract topics
.venv/bin/python -m sara_brain.tools.extract_biology2e_topics \
    --facts-dir benchmarks/biology2e_facts \
    --toc /path/to/Biology2e-WEB.txt \
    --subject biology \
    --dest brain_root/ \
    --min-freq 5

# Step 2 — migrate paths
.venv/bin/python -m sara_brain.tools.migrate_to_hierarchy \
    --source biology2e.db \
    --subject biology \
    --dest brain_root/ \
    --verify

# Step 3 — MMLU parity benchmark
.venv/bin/python benchmarks/run_spacy_ch10.py \
    --db brain_root/ \
    --questions benchmarks/mmlu_biology_full.json \
    --mode property --top-k 5 --use-idf \
    --output benchmarks/mmlu_biology_hier.json
```

---

## Success thresholds (from plan)

| Metric | Target |
|---|---|
| MMLU correct | ≥ 100 (baseline 96 with IDF+gate) |
| OR precision | ≥ 52% at comparable coverage |
| Unclassified rate | < 20% of source paths |
| Ch10 exam | 28/33 (100% precision / 84.8% coverage) unchanged |
| Full test suite | 375 / 376 pass (pre-existing failure in test_why_marks_refuted_paths unrelated) |

---

## Invariants preserved

- Zero LLM in any pipeline step.
- All compounding — existing biology2e.db is untouched; `Brain("biology2e.db")` still works.
- Sara never forgets — unclassified facts land in `_unclassified.db`, not discarded.
- Provenance intact — each concept path carries the original `source_text`.

---

## Separately-tracked follow-ups

- **Strength-biasing in foundation** — recognizer already averages signed segment strengths along traversals; MMLU scorer currently bypasses `brain.recognize()` entirely. Future refactor: have the scorer call `brain.recognize()` to inherit strength-weighting.
- **Multi-source ingest** — biology2e is single-source; no two-witness bumps have fired. Topic-grain storage (this plan) is the prerequisite. Next plan: Wikipedia biology + Khan Academy as second sources.

---

*Versioned per `feedback_versioned_docs.md`. Previous in series: v021_biology2e_session_summary.md.*
