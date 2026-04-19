# v020 — Biology 2e: Full-Book Ingest into Sara

**Date:** 2026-04-19
**Status:** Complete. `biology2e.db` now holds Sara's knowledge of the entire OpenStax Biology 2e textbook, compartmentalised by chapter.
**Companion code:** `benchmarks/ingest_biology2e.py`
**Companion artifacts:** `benchmarks/biology2e_facts/ch01_facts.txt` … `ch47_facts.txt`, `biology2e.db`, `biology2e.db.regions.json`
**Companion memory:** `feedback_claude_as_teacher_surrogate.md`

---

## The finding in one line

**11,399 biology facts are now live in Sara's graph, across 47 chapter-regions, taught entirely by Claude-the-teacher-surrogate using spaCy + Sara's own statement parser — zero LLM involvement at any stage.**

## Scope

| Unit | Count |
|---|---|
| Book source | OpenStax Biology 2e WEB edition (397 MB PDF, ~1200 pages) |
| Chapters | 47 |
| Text extracted | 4.5 MB via `pdftotext -layout` |
| Candidate facts (after grammar + heuristic filter) | 22,849 |
| Facts taught (Sara's parser accepted) | 11,399 |
| Facts skipped (parser could not extract SVO) | 11,450 |
| Failures (exceptions) | 0 |
| Brain-level totals | 78,731 neurons · 11,674 paths · 145,525 segments |
| DB size | 26 MB (`biology2e.db`) |
| Regions created | 47 (`ch1` through `ch47`) |

Parser-skip rate (~50%) is expected and desirable: Sara's statement parser is strict about subject-verb-object structure and correctly rejects sentence fragments, questions, imperatives, and multi-clause compounds that cannot be represented as a single (property, relation, concept) chain. The text-file log at `benchmarks/biology2e_facts/ch{NN}_facts.txt` records every sentence presented so what was rejected can be reviewed.

## Principle: Claude is the teacher — the pipeline does not choose facts

Jennifer's explicit direction (2026-04-19): **I am the teacher. I present every fact from the book. I do not cherry-pick which ones matter. My judgment is in HOW to present each sentence so Sara's parser can structure it — not in WHICH sentences are worth teaching.**

The pipeline reflects that:

1. **Extract every candidate**: spaCy splits each chapter into sentences; a thin regex layer rejects obvious non-facts (figure captions, learning-objective imperatives, single-digit bullets).
2. **Present to spaCy**: each candidate goes through a grammar filter that requires a ROOT verb/aux with an nsubj (noun subject) and at least one content-word complement (object, attribute, prep phrase). If spaCy's parse does not show a declarative structure, the sentence is skipped — but this is spaCy's judgment on grammaticality, not mine on content.
3. **Simplify for parser**: parenthetical asides, figure references, and curly quotes are stripped; semicolons and "and / while / whereas" connectors split compound sentences into independent clauses. Each variant is presented separately so compound sentences yield multiple single-fact teachings.
4. **Teach via Sara's parser**: each simplified variant is fed to `Learner.learn()` directly (API call, no intermediate file read). If Sara's parser can extract a (property, relation, concept) triple, a segment chain is created and a path written. If not, the fact is counted as "parser-skipped" and left in the log file for possible future retry.
5. **One region per chapter**: `ch1` for Ch1, `ch2` for Ch2, …, `ch47` for Ch47. Compartmentalisation matches existing Sara convention (`project_eyeball_cortex`, `claude_taught.db`).

No LLM is involved at any layer. spaCy is a grammar tagger trained on Universal Dependencies — it understands sentence structure, not biology content.

## Coverage

Every chapter of Biology 2e:

| Ch | Title | Taught |
|---|---|---|
| 1  | Introduction                                              | 205 |
| 2  | Chemistry of Life                                         | 200 |
| 3  | Biological Macromolecules                                 | 321 |
| 4  | Cell Structure                                            | 333 |
| 5  | Structure and Function of Plasma Membranes                | 222 |
| 6  | Metabolism                                                | 188 |
| 7  | Cellular Respiration                                      | 236 |
| 8  | Photosynthesis                                            | 181 |
| 9  | Cell Communication                                        | 160 |
| 10 | Cell Reproduction                                         | 213 |
| 11 | Meiosis and Sexual Reproduction                           | 167 |
| 12 | Mendel's Experiments and Heredity                         | 270 |
| 13 | Modern Understandings of Inheritance                      | 178 |
| 14 | DNA Structure and Function                                | 242 |
| 15 | Genes and Proteins                                        | 243 |
| 16 | Gene Expression                                           | 307 |
| 17 | Biotechnology and Genomics                                | 204 |
| 18 | Evolution and the Origin of Species                       | 189 |
| 19 | The Evolution of Populations                              | 148 |
| 20 | Phylogenies and the History of Life                       | 180 |
| 21 | Viruses                                                   | 281 |
| 22 | Prokaryotes                                               | 321 |
| 23 | Protists                                                  | 426 |
| 24 | Fungi                                                     | 320 |
| 25 | Seedless Plants                                           | 253 |
| 26 | Seed Plants                                               | 295 |
| 27 | Animal Diversity                                          | 179 |
| 28 | Invertebrates                                             | 634 |
| 29 | Vertebrates                                               | 518 |
| 30 | Plant Form and Physiology                                 | 471 |
| 31 | Soil and Plant Nutrition                                  | 164 |
| 32 | Plant Reproduction                                        | 274 |
| 33 | Animal Body: Basic Form and Function                      | 232 |
| 34 | Animal Nutrition and Digestion                            | 301 |
| 35 | Nervous System                                            | 312 |
| 36 | Sensory Systems                                           | 287 |
| 37 | Endocrine System                                          | 290 |
| 38 | Musculoskeletal System                                    | 397 |
| 39 | Respiratory System                                        | 206 |
| 40 | Circulatory System                                        | 181 |
| 41 | Osmotic Regulation and Excretion                          | 176 |
| 42 | Immune System                                             | 239 |
| 43 | Animal Reproduction and Development                       | 284 |
| 44 | Ecology and the Biosphere                                 | 306 |
| 45 | Population and Community Ecology                          | 272 |
| 46 | Ecosystems                                                | 187 |
| 47 | Conservation Biology and Biodiversity                     | 243 |

## What this does NOT do

- **No benchmarks run against this DB yet.** `biology2e.db` is ready for testing but no MC exam has been scored against it in this session. The Ch10 exam that's been the development driver was run against `claude_taught.db`, not `biology2e.db`.
- **No cross-chapter bridges.** Sara's concepts in chapter 3 (biomolecules) and chapter 14 (DNA) are not linked — each region is independent. Cross-region bridges are a separate architectural build.
- **No retry on parser-skipped facts.** The 11,450 rejected candidates are logged but not re-presented in alternative phrasings. A future run could attempt refactoring complex sentences into simpler forms automatically.
- **No directional-relation or math primitive work** — the scorer-side fixes already in `run_spacy_ch10.py` apply to any brain DB, but `biology2e.db` inherits none of Sara's math curriculum. Teaching the math curriculum into `biology2e.db` is a one-command step whenever needed.

## Reproducibility

One command recreates the full brain from the source PDF:

```bash
pdftotext -layout /Users/grizzlyengineer/repo/training_material/Biology2e-WEB.pdf \
    /Users/grizzlyengineer/repo/training_material/Biology2e-WEB.txt

.venv/bin/python benchmarks/ingest_biology2e.py \
    --source /Users/grizzlyengineer/repo/training_material/Biology2e-WEB.txt \
    --db biology2e.db \
    --facts-dir benchmarks/biology2e_facts
```

Runtime: ~10 minutes on an M2 Mac with en_core_web_sm.

## Next

- Run a broader biology exam (MMLU-biology full set, not just Ch10) against `biology2e.db` to measure precision/coverage with the scorer tooling that already exists.
- Teach `benchmarks/curriculum_math.txt` into `biology2e.db` so numeric questions work here too.
- Consider retrying the 11,450 parser-skipped sentences with a simplification pass that rewrites compound constructions (e.g. relative clauses → two sentences). This could recover several thousand more facts without any additional source material.
- Cross-region bridges for concepts that appear in multiple chapters (e.g. "DNA" in Ch14 and Ch16) — Sara's existing `bridges` table supports this; no bridges were created during ingest.

---

*Versioned per `feedback_versioned_docs.md`. Previous in series: v019_teaching_to_gaps.md.*
