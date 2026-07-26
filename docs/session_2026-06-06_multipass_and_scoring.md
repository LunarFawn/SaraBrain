# Session Report: Multi-Pass Teaching & Wavefront Scoring — 2026-06-06

## What We Did

Picked up from DESIGN_VISION.md's "Next Engineering Priority: Multi-Pass Teaching" and worked through the full pipeline from teaching to benchmark scoring, uncovering and fixing several issues along the way.

---

## New Components Built

### 1. Multi-Pass Teaching (`src/sara_brain/cortex/transformer/v2/multipass.py`)

Three focused extraction passes over the same document:

- **Pass 1 (Definitions):** keeps triples where relation is `is_a` or `be` — what IS this thing?
- **Pass 2 (Relationships):** keeps non-definitional triples — what does it DO?
- **Pass 3 (Bridges):** re-scans and commits triples where both subject and object already exist as neurons in the substrate

Usage:
```bash
sara-teach-book document.txt --brain my.db --extractor sara --multipass
```

Also wired into MCP: `brain_ingest(source, multipass=True)`

The bridge pass gains its real value on longer documents or multi-document ingests — concepts introduced early get cross-references strengthened later.

### 2. Label Normalization (`src/sara_brain/cortex/transformer/v2/normalize.py`)

Post-extraction cleanup applied to every triple the sara extractor emits:

- Strips framing punctuation
- Rejects pure-punctuation/symbol tokens
- Rejects stopword-only labels
- Strips leading stopwords ("the cell cycle" → "cell cycle")
- Preserves trailing Roman numerals (I, II, III) and particles (over, up, out)
- Normalizes plurals (cells → cell, chromosomes → chromosome)
- Rejects single-character labels

### 3. Math Wired into Wavefront Scorer (`src/sara_brain/core/wavefront_scorer.py`)

- `NumberExtractor` pulls numeric values from question text (e.g., "2n = 96")
- Checks `operation_tag` on segments reachable from question seeds (e.g., `multiply:0.5` from "reduces by half")
- Computes results and boosts choices whose text matches (e.g., 96 × 0.5 = 48 → boosts "48")
- Also fixed `brain.teach_triple()` to run `MathResolver` on `source_text` so future teaches get operation_tags automatically

### 4. Fixed Query Resolver (`src/sara_brain/core/query_resolver.py`)

The spaCy-free `resolve_query_nospacy` now:

- Probes substrate with raw bigrams BEFORE stopword filtering (catches "prophase i", "meiosis ii")
- Handles hyphenated words by trying space-joined variants ("crossing-over" → "crossing over")
- Captures single uppercase letters in tokenization (Roman numerals: I, V, X)

### 5. Unblocked `is_a` Propagation (`src/sara_brain/core/recognizer.py`)

`_NON_PROPAGATING_RELATIONS` was `{"is_a"}` — meaning the wavefront could never follow definitional edges. This blocked the most important connections (e.g., "meiosis → prophase_attribute → prophase"). Now empty — all relations propagate.

---

## Benchmark Results

33 MMLU biology questions, wavefront confluence scoring (no LLM), brain taught from Biology 2e chapters 10+11 using sara extractor + multipass + normalization.

| Configuration | Correct | Precision | Abstains | Notes |
|--------------|---------|-----------|----------|-------|
| Rules extractor + spaCy resolver, ch10 only, depth 3 | 10/22 | 45.5% | 10 | Previous baseline |
| Sara extractor + nospacy, ch10+11, dirty (no normalize), depth 3 | 5/22 | 22.7% | 5 | Hub noise from garbage neurons |
| Sara extractor + normalize, depth 3 | 7/22 | 31.8% | 9 | Normalization helps |
| Sara extractor + normalize + is_a unblocked, depth 1 | 2/9 | 22.2% | 17 | Too tight, starves |
| **Sara extractor + all fixes, depth 2** | **8/22** | **36.4%** | **7** | Best with sara extractor |

Q278 ("crossing-over occurs during which phase?") — previously always wrong — is now correct with all fixes at both depth 1 and depth 2.

---

## Remaining Problems

### 1. Depth 2 vs Depth 3 Tradeoff

- Depth 1: too tight, most questions can't find convergence (the `property → _attribute → concept` chain requires 2 hops minimum)
- Depth 2: sweet spot — bridges through attribute nodes without flooding
- Depth 3: floods — every seed reaches hundreds of nodes and convergence becomes meaningless

The architecture stores triples as 3-node chains. Depth 2 is the natural hop distance for one fact lookup. Depth 3 is "thinking harder" — following chains of facts — but the current scorer can't discriminate in the resulting noise.

### 2. Scorer Can't Discriminate in Dense Convergence

When the question and multiple choices all share convergence through hub neurons ("cell" = 118 connections), the wrong choice can outscore the right one by having more total overlapping nodes. The scorer sums power at ALL shared nodes equally — it doesn't weight specific convergence higher than generic convergence.

Potential fixes:
- Weight convergence inversely by node connectivity (specific nodes count more)
- Only count nodes reached by MULTIPLE question seeds (not just one)
- Topic-gated propagation (only follow edges that lead toward other question seeds)

### 3. Extractor Still Produces Some Garbage

Even with normalization, numbers (0, 1, 2, 10, 14...) and a few stopwords still leak through as neurons. The model was trained on synthetic nonsense words and has no experience rejecting real English noise. The proper fix is training data reform — add real English noise to synthetic paragraphs so the model learns to skip it.

### 4. Training Data Reform Needed (Tasks 4-6 from plan)

The generator (`scripts/generate_extractor_v2_data.py`) uses clean synthetic words exclusively. The model needs:
- Real English stopwords/pronouns/numbers in input paragraphs that it must NOT copy into triples
- Plural variants in input with singular in output targets
- Punctuation noise in input

Then retrain (~2hrs on RTX 3070).

### 5. Many Questions Are Outside Ch10+11 Content

Only ~14 of the 33 test questions are actually about cell division/meiosis (chapters 10-11). The rest cover genetics, ecology, evolution, virology, endosymbiosis — content from other chapters the brain was never taught. This creates unavoidable abstains/wrong answers.

---

## Architecture Insight from This Session

The wavefront convergence mechanism works correctly when:
1. The seeds resolve to the right neurons (fixed resolver)
2. The propagation follows all edge types (unblocked is_a)
3. The depth matches the graph's chain length (depth 2 for this triple structure)
4. The neurons are clean (normalization)

The mechanism breaks when the graph is too dense relative to the depth — convergence becomes meaningless overlap rather than meaningful intersection. This is the "cell has 118 connections" problem: not a bug, but a density/discrimination issue that the scorer needs to handle.

The human brain equivalent: you hear "cell" in a meiosis question and immediately context-gate it — you only think about biological cells, not prison cells. The wavefront doesn't have that context gate yet.

---

## Files Modified This Session

- `src/sara_brain/cortex/transformer/v2/multipass.py` — NEW: pass filters
- `src/sara_brain/cortex/transformer/v2/normalize.py` — NEW: label normalization
- `src/sara_brain/core/wavefront_scorer.py` — math boost integration
- `src/sara_brain/core/brain.py` — teach_triple runs MathResolver on source_text
- `src/sara_brain/core/recognizer.py` — unblocked is_a propagation
- `src/sara_brain/core/query_resolver.py` — fixed nospacy resolver (hyphens, Roman numerals, raw bigram probing)
- `src/sara_reader/cli_teach_book.py` — multipass flag, normalization wired into sara extractor

---

## Next Steps

1. **Fix the scorer's hub discrimination** — the biggest remaining accuracy gap
2. **Training data reform** — add English noise so model learns to reject garbage at extraction time
3. **Retrain the 115M extractor** — on cleaned data
4. **Teach more chapters** — cover the full question set
5. **Set default depth to 2** — for the scorer/benchmark runner
