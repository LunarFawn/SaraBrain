# Sara as Instrument paper — reading list

A working list of citations to consider engaging with in future revisions of `sara_as_instrument_paper_rev6.md`. Each topic has candidate references with one-line descriptions and one-line "why it matters" notes. Stars mark the single most-useful read per topic.

## Why this list exists

Rev 6 of the instrument paper makes claims that touch on three adjacent literatures — hallucination taxonomy, RAG evaluation, and training-data contamination — without engaging them. §8's scoping paragraph names this honestly. This doc is the reading queue for if/when future revisions choose to engage.

**Discipline:** do not cite anything from this list without reading it first. The list is for prioritization, not for direct citation. Verify citation details (year, venue, exact title) when you do read each one — the entries below are working notes from a reviewer's recollection, not copy-pasted bibliography lines.

---

## Hallucination taxonomy

- ★ **Ji et al. 2023** — *Survey of Hallucination in Natural Language Generation.* The standard survey across NLG tasks.
  - Why it matters here: would let you position the "confabulation" subtypes in Cases 5.6 (acronym-expansion) and 5.7 (format-imitation) within an existing taxonomy rather than coining standalone vocabulary.
- **Huang et al. 2023** — *A Survey on Hallucination in Large Language Models.* The LLM-specific update to Ji.
  - Why it matters here: more recent and specifically frames hallucination in transformer architectures, which is closer to the paper's scope.
- **Maynez et al. 2020** — *On Faithfulness and Factuality in Abstractive Summarization.* The foundational distinction between faithfulness (to source) and factuality (to world).
  - Why it matters here: the paper's "substrate fidelity" axis is essentially Maynez's faithfulness, applied to a structured substrate. Engaging would clarify what's new and what's a known concept under a new substrate type.
- **Min et al. 2023** — *FActScore* (per-claim atomic factuality scoring).
  - Why it matters here: closest prior art for the per-triple grading the paper proposes. Engaging would sharpen the novelty positioning — the differentiator is the structured substrate, not the per-claim grading idea itself.
- **Manakul et al. 2023** — *SelfCheckGPT* (self-consistency-based hallucination detection).
  - Why it matters here: adjacent black-box detection technique; useful for positioning the substrate-grounded approach against zero-resource methods.

## RAG evaluation

- ★ **Gao et al. 2024** — *Retrieval-Augmented Generation for Large Language Models: A Survey.* Single anchor for the RAG-eval space.
  - Why it matters here: the four-property criterion would be sharper if positioned against what RAG-eval frameworks already do at coarser grain (and don't do at per-triple grain).
- **Es et al. 2024** — *RAGAs* (open-source automated RAG eval).
  - Why it matters here: most-used RAG-eval framework; positioning the four-property criterion vs. RAGAs' faithfulness/relevance metrics would clarify the gap.
- *Additional candidates* — ARES (Saad-Falcon et al. 2024), CRAG (Yang et al. 2024). Verify titles/venues before citing.

## Training-data contamination

- ★ **Sainz et al. 2023** — *NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each Benchmark.* Short, accessible standard read on the contamination problem.
  - Why it matters here: would empirically support Property 4's necessity by showing how badly contaminated common benchmarks already are.
- **Magar & Schwartz 2022** — *Data Contamination: From Memorization to Exploitation.* Mechanistic study of how contamination affects downstream performance.
  - Why it matters here: directly relevant to the argument that training-orthogonal substrates measure something different from contaminated ones.
- **Carlini et al. 2021/2023** — *Quantifying Memorization Across Neural Language Models.* Memorization scaling with model size.
  - Why it matters here: supports the broader argument that public corpora are unsafe substrates for measurement.
- *Additional candidates* — Golchin & Surdeanu 2023 (*Time Travel in LLMs*), Roberts et al. 2023 (data contamination through time). Verify before citing.

## Sycophancy / format compliance (relevant to Case 5.7)

- **Sharma et al. 2023** — *Towards Understanding Sycophancy in Language Models.* Anthropic.
  - Why it matters here: Case 5.7 (format-imitation under detailed prompt instructions) is structurally a format-compliance failure; sycophancy is the closest existing literature to position it against.

## KG-augmented LMs / GraphRAG

- **Edge et al. 2024** — *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* Microsoft GraphRAG.
  - Why it matters here: closest contemporary system to Sara in spirit; positioning vs. GraphRAG would clarify what wavefront propagation adds over graph-augmented retrieval.
- *Additional candidates* — Think-on-Graph (Sun et al. 2024), KG-RAG variants. Verify before citing.

## MCP citation (needed for §2.3)

- **Anthropic 2024** — *Model Context Protocol* spec / announcement.
  - Why it matters here: §2.3 leans on MCP as the cross-vendor reader interface but provides no citation. One-line addition once you have the canonical URL or spec reference.

---

## How to use this list

1. When you're ready to engage one topic, read the starred entry first. It's enough to anchor a §8 subsection.
2. After reading, decide if it actually fits the paper's scope before citing — many of these may turn out to be tangential.
3. If it does fit, add to the appropriate §8 subsection of the paper and either remove it from this list or annotate with "engaged in rev N."
4. If it doesn't fit, leave a brief note here saying why, so you don't have to re-read to remember the decision.

The "Additional candidates" entries are noted for completeness but I'm less confident of their exact titles/venues — verify before reading or citing.
