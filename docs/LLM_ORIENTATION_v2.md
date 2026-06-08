# Sara Brain — LLM Orientation (2026-06-07)

**For:** any LLM session picking up this project mid-stream.
Read this in full before touching code.

---

## 1. What Sara Brain Is

Sara Brain is a **persistent path-of-thought knowledge store**. It pairs with a stateless LLM (the "cortex") to form a two-layer architecture:

- **Hippocampus** = Sara Brain. SQLite graph of triples. Persistent. Inspectable. Correctable. All domain knowledge lives here.
- **Cortex** = LLM or from-scratch 115M model. Stateless. Does language processing only. Zero domain knowledge in weights.

The brain's defining mechanism is **wavefront convergence**: seeded neurons emit parallel wavefronts across the graph. Where wavefronts intersect is the recognition signal. This is not one retrieval option among many — it IS the brain.

```
Teaching:   Document → [Sara Extractor 115M] → triples → Sara Brain (SQLite)
Retrieval:  Question → Seed Resolution → Wavefront Propagation → Convergence → Answer
```

## 2. How Knowledge Is Stored

Triples are stored as 3-node directed chains:

```
property_neuron → relation_attribute_neuron → concept_neuron
```

Example: "meiosis involves prophase" becomes:
```
meiosis (concept) → prophase_attribute (relation) → prophase (concept)
```

The `_build_chain` function in `src/sara_brain/core/learner.py` handles this. Every fact creates a recorded `Path` with source text provenance.

**Key implication:** traversing one fact requires depth 2 (concept → attribute → concept). This is why depth 1 starves and depth 3 floods.

## 3. How Retrieval Works (Wavefront Scoring)

For multiple-choice questions (`src/sara_brain/core/wavefront_scorer.py`):

1. Resolve question text → seed neurons (substrate-aware, checks what neurons actually exist)
2. Resolve each choice → seed neurons
3. Launch wavefronts from question seeds (BFS, bidirectional, depth 2)
4. Launch wavefronts from each choice's seeds
5. Score = sum of power at nodes where BOTH wavefronts converge
6. Highest-scoring choice wins; zero-score = abstain

Seed resolution (`src/sara_brain/core/query_resolver.py`): uses regex tokenization + bigram/trigram substrate probing. No spaCy. Handles hyphens ("crossing-over" → tries "crossing over"), Roman numerals ("Prophase I" → "prophase i"), and compound terms.

## 4. The Teaching Pipeline

### From-Scratch Extractor (115M, copy-mechanism transformer)
```bash
sara-teach-book document.txt --brain my.db --extractor sara
```

The model reads text and emits structured triples:
```
t_start meiosis t_rel is_a t_obj nuclear division t_end
t_start crossing over t_rel occurs_in t_obj prophase i t_end
```

Trained on synthetic nonsense words (domain-agnostic). Checkpoint: `models/sara-extractor-115m-v2/best.pt`

### Multi-Pass Teaching
```bash
sara-teach-book document.txt --brain my.db --extractor sara --multipass
```

Three passes over the same document:
- **Pass 1 (Definitions):** is_a triples only — what IS this thing
- **Pass 2 (Relationships):** non-is_a triples — what does it DO
- **Pass 3 (Bridges):** re-commits triples where both endpoints already exist as neurons (strengthens connections between known concepts)

### Post-Extraction Normalization (`src/sara_brain/cortex/transformer/v2/normalize.py`)

Applied to every triple before it enters the brain:
- Rejects stopwords, punctuation, single-char garbage
- Normalizes plurals (cells → cell)
- Preserves Roman numerals (prophase I stays as "prophase i")
- Preserves phrasal particles ("crossing over" not stripped to "crossing")

### MCP Server (`src/sara_brain/mcp_server.py`)
Sara as a tool provider for any LLM client. 14+ tools including teach, query, explore, refute, ingest.

## 5. Current Performance (2026-06-06 Benchmark)

33 MMLU biology questions, pure wavefront scoring (no LLM), brain taught from Biology 2e chapters 10+11:

| Configuration | Precision | Notes |
|--------------|-----------|-------|
| Rules extractor + spaCy, depth 3 | 45.5% (10/22) | Previous baseline (sparse graph) |
| Sara extractor + all fixes, depth 2 | **36.4% (8/22)** | Current best with sara extractor |
| Sara extractor, depth 3 | 31.8% (7/22) | Flooding degrades discrimination |
| Sara extractor, depth 1 | 22.2% (2/9) | Too tight, most questions starve |

Reference: random = 25%, hand-curated 45 facts = 80% (the architecture works when facts are precise)

## 6. Known Problems (What to Work On)

### Problem 1: Scorer Hub Discrimination
"Cell" has 118 connections. When the question mentions "cell", the wavefront floods through that hub and every choice converges. The scorer sums ALL convergence equally — a convergence at "cell" (generic) counts the same as convergence at "sister chromatid separation" (specific). Need: weight convergence inversely by node connectivity, or topic-gate propagation.

### Problem 2: Extractor Still Produces Garbage
Numbers (0, 1, 2...) and some stopwords leak through as neurons despite normalization. Root cause: training data uses only synthetic words — model has no experience rejecting real English noise. Fix: add English noise to training data, retrain.

### Problem 3: Depth 2 vs Depth 3 Tradeoff
The triple chain structure (concept → attribute → concept) means:
- Depth 1 can't bridge between concepts (only reaches intermediate nodes)
- Depth 2 bridges exactly one fact (sweet spot for current benchmark)
- Depth 3 bridges fact-chains but floods (every seed reaches half the brain)

The architecture's backwave propagation (hit a concept end node → bounce backward to find connections) works at depth 2. Depth 3 is for "thinking harder" but needs scorer improvements to be useful.

### Problem 4: Test Questions Span Multiple Chapters
Only ~14 of the 33 test questions are actually ch10-11 material. The rest cover genetics, ecology, evolution, virology — chapters not taught. This creates unavoidable abstains.

## 7. Critical Code Paths

| What | Where |
|------|-------|
| Triple storage | `src/sara_brain/core/learner.py` → `_build_chain` |
| Wavefront propagation | `src/sara_brain/core/recognizer.py` → `_propagate` |
| MC scoring | `src/sara_brain/core/wavefront_scorer.py` → `score_choices` |
| Seed resolution (no spaCy) | `src/sara_brain/core/query_resolver.py` → `resolve_query_nospacy` |
| Sara extractor inference | `src/sara_reader/cli_teach_book.py` → `_sara_extract` |
| Label normalization | `src/sara_brain/cortex/transformer/v2/normalize.py` |
| Multi-pass filters | `src/sara_brain/cortex/transformer/v2/multipass.py` |
| Math computation | `src/sara_brain/core/math.py` |
| MCP server | `src/sara_brain/mcp_server.py` |
| Training data generator | `scripts/generate_extractor_v2_data.py` |
| Extractor model architecture | `scripts/train_sara_extractor_scratch.py` |
| Benchmark runner | `benchmarks/run_wavefront_ch10.py` |

## 8. Architecture Rules (Non-Negotiable)

| Rule | Meaning |
|------|---------|
| Wavefront IS the brain | Cannot be demoted to one tool option. Runs first, always. |
| Knowledge in substrate, not weights | Models do language; Sara stores knowledge. Neither does the other's job. |
| Depth 2 is the default | Matches the triple-chain structure. Depth 3 only with scorer improvements. |
| `is_a` propagates | Was previously blocked (`_NON_PROPAGATING_RELATIONS`). Now propagates. Required for convergence to work. |
| No spaCy in the hot path | Seed resolution and scoring use regex + substrate probing. The sara extractor is the from-scratch 115M model. |
| Smaller models are more faithful | 115M is the right size for structural tasks. Bigger models hedge/refuse. |
| Don't ship partial as done | Lead with remaining failures, not the wins. |

## 9. The Math System

Segments can carry `operation_tag` (e.g., `multiply:0.5` from "reduces by half"). When teaching via `brain.teach_triple(source_text="meiosis reduces chromosome number by half")`, `MathResolver` detects the arithmetic phrase and `MathLinker` stores the tag.

The wavefront scorer checks for `operation_tag` on segments reachable from question seeds, applies the operation to numbers extracted from the question (via `NumberExtractor`), and boosts choices matching the computed result.

## 10. Environment

- **Repo:** `/home/grizzlyengineer/repo/SaraBrain/`
- **Venv:** `.venv/` (Python 3.12)
- **GPU:** RTX 3070 (8GB VRAM) — for training only
- **Extractor checkpoint:** `models/sara-extractor-115m-v2/best.pt` (462MB)
- **Test brains:** `/tmp/bio_sara_fix_20260606.db` (ch10+11, normalized, current best)
- **OS:** WSL2 Linux on Windows

## 11. Running Things

```bash
# Teach a book
.venv/bin/python -m sara_reader.cli_teach_book FILE --brain DB --extractor sara --multipass --no-dictionary

# Run the benchmark (pure wavefront, no LLM)
.venv/bin/python -c "
from sara_brain.core.brain import Brain
from sara_brain.core.wavefront_scorer import score_choices, pick_choice
brain = Brain('path.db')
brain.recognizer.max_depth = 2
# ... score_choices(question, choices, None, brain.recognizer, brain.neuron_repo)
"

# Check brain stats
.venv/bin/python -c "
from sara_brain.core.brain import Brain
b = Brain('path.db'); print(b.stats())
"

# Tests (289 pass, 1 fails due to missing anthropic SDK)
.venv/bin/pytest tests/ -x -q
```

## 12. Next Engineering Priorities

1. **Scorer hub discrimination** — convergence at specific nodes should outweigh convergence at hubs
2. **Training data reform** — add English noise so extractor learns to reject garbage
3. **Retrain 115M extractor** — on cleaned data (~2hrs RTX 3070)
4. **Teach more chapters** — cover the full 33-question test set
5. **Set depth 2 as default** in recognizer

---

*Last updated: 2026-06-07. Previous orientation: docs/LLM_ORIENTATION.md (2026-05-12)*
