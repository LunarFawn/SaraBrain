# v021 — Biology 2e: Post-Ingest Session Summary

**Date:** 2026-04-19
**Status:** Sara now has the entire Biology 2e textbook in her graph, plus MMLU-biology benchmarked against it. Documenting honest numbers and follow-up surface.

---

## Bottom-line numbers

**Brain (`biology2e.db`):**

| Metric | Count |
|---|---|
| Chapter regions | 47 (ch1 … ch47) + 1 math region |
| Neurons | 78,731 |
| Paths | ~13,233 (11,399 from first pass + 1,834 from aggressive retry) |
| Segments | 145,525 |
| Cross-region bridges | 466,524 (7,559 unique shared concepts) |
| DB size | 63 MB |
| Math curriculum taught | 14 composite operations in the `math` region |

**MMLU-Biology full-set benchmark (310 questions), `biology2e.db`, zero LLM:**

| Config | Correct | Wrong | Abstain | Precision (of answered) | Coverage |
|---|---|---|---|---|---|
| default (threshold=1, top_k=3) | 51 | 99 | 160 | 34.0% | 48.4% |
| threshold=3, top_k=3 | 31 | 48 | 231 | 39.2% | 25.5% |
| threshold=1, top_k=5 | 53 | 101 | 156 | 34.4% | 49.7% |

Runtime: 12-15 s total (0.04 s / question).

## What this proves

- **Scale of Sara's graph:** the Biology 2e knowledge base compresses ~1,200 book pages into 13k declarative path-facts, zero LLM in the loop.
- **Teaching speed at scale:** pdftotext → spaCy → Sara's parser ingested the full book in about 10 minutes.
- **Architecture robustness:** the scorer built for 9 Ch10 regions ran unchanged against 47 chapter regions; only a `--top-k` CLI flag was added.
- **Precision limits at scale:** 34% precision on MMLU-biology vs 100% on the 33-question Ch10 exam reflects that auto-ingest facts are lexically cruder than the 4 rounds of hand-targeted teaching that got Ch10 to perfect.

## What this does NOT prove

- **Bridges are not yet consumed by the scorer.** 466k bridges sit in `biology2e.db.bridges` as infrastructure. Making the scorer bridge-aware is a separate architectural task.
- **MMLU-biology tests beyond Ch10 content** (genetics depth, biochemistry, ecology, plant biology) — Sara has auto-taught facts there but no hand-targeted teaching rounds. Sample wrong answers show the scorer picking on surface overlap rather than semantic fit.
- **Scorer semantics are still lexical.** Word-bag scoring cannot distinguish "heterotrophs obtain energy from autotrophs" from "heterotrophs obtain energy by consuming autotrophs" at the MCQ level, even though the book contains the distinction.

## Gap report on MMLU-biology failures

Of 259 failing questions:

| Kind | Count | Teach-action |
|---|---|---|
| distinction_gap | 191 | Teach finer discriminating facts per question (expensive) |
| vocab_gap | 46 | Teach definitions of unfamiliar key terms |
| relation_gap | 22 | Many are meta-choice architectural ("Both A and C") — unreachable without meta-choice support |

## Tools shipped this session

| File | Purpose |
|---|---|
| `benchmarks/ingest_biology2e.py` | PDF-text → per-chapter facts → teach |
| `benchmarks/reteach_skipped_biology2e.py` | Aggressive simplification for parser-skipped sentences |
| `benchmarks/bridge_biology2e.py` | Cross-region bridge builder |
| `benchmarks/biology2e_facts/ch01..ch47_facts.txt` | Per-chapter extracted fact log |
| `docs/v020_biology2e_ingest.md` | Full-book ingest design doc |
| `docs/v021_biology2e_session_summary.md` | This file |

## Recommended next work (not yet executed)

In priority order of compounding leverage:

1. **Bridge-aware scoring** — teach `score_choice_by_property` to consume the `bridges` table so a question routed to ch3 can see `DNA`-related content in ch14/ch16 via the `same_as` bridges. Expected: precision rises without losing coverage.
2. **Distinction-gap cleanup teach** — pick the top 30 distinction-gap questions, teach a finer-grained disambiguating fact per question. 30 facts × 1 teach each. Expected: +30 correct answers if chosen well.
3. **Auto-ingest pollution audit** — scan the 13k auto-taught facts for obviously-wrong-simplified pieces and refute them. Reduces noise and ties.
4. **Meta-choice handler** — code-level support for "Both A and C" / "A, B, and C only" / "None of the above" answer patterns. Closes ~10 of the relation gaps architecturally.

## Invariants preserved

- Zero LLM in any pipeline step (pdftotext + spaCy + regex + Sara's parser).
- All compounding, nothing throwaway (every teach, every bridge, every script is persisted).
- Provenance chain intact — every answer Sara gives on MMLU can be traced back to a specific path, which traces back to a specific textbook sentence, which is in the per-chapter fact log.

---

*Versioned per `feedback_versioned_docs.md`. Previous in series: v020_biology2e_ingest.md.*
