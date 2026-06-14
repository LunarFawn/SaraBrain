# Sara Brain — Work In Progress Log

**Purpose:** This file tracks engineering decisions, model selections, and benchmark progress across AI sessions. It is the persistent "memory" of the project's current state.

---

## Current Status (2026-06-13)

### 1. The Extractor Confusion
*   **Models found:**
    *   `sara-extractor-115m-v2` (June 3): The previous standard.
    *   `sara-extractor-v2-clean` (June 9): Includes a new noise-reduction system (synthetic training with English noise). **Suspected current production model.**
    *   `sara-extractor-340m-v2` (June 10): Large model, but logs show potential divergence/instability. **Likely not ready.**
*   **Action Needed:** Verify if `v2-clean` is the intended extractor for the full biology build.

### 2. The Persistence Failure
*   **Event:** A 115M biology brain (8 hours of ingestion) was lost because it was stored in `/tmp/`.
*   **New Rule:** All brain databases MUST be persistent. **Location:** `data/` or project root.
*   **Status:** Full biology ingestion is ON HOLD until the extractor is verified.

### 3. The Interface (Benchmark)
*   **Current Goal:** Run the MMLU Biology benchmark against a Chapter 10 brain to verify quality.
*   **Interface Options:** `cli_teach_book` (ingestion), `sara-ask-stateless` (retrieval/benchmarking), MCP Server (tools).
*   **Action Needed:** Confirm which interface is the "current" production path for running benchmarks.

### 4. Known Conflicts
*   `GEMINI.md` previously mandated 115M, then I incorrectly updated it to 340M, then reverted it.
*   The project has shifted toward an MCP-first architecture, but `cli_teach_book.py` is still the primary ingestion tool.

## Consensus (2026-06-13, 15:15)
*   **Decision:** Move forward with the `v2-clean` (115M, June 9) extractor.
*   **Reasoning:** It is the newest model (pre-340M) and includes the specific "English noise" training needed for cleaner extraction.
*   **Test Case:** Ingest Chapter 10 ONLY into `data/ch10_v2_clean.db`.
*   **Next Step:** Run MMLU Biology benchmark against this brain.

## Consensus (2026-06-13, 17:00)
*   **Prose Injection Failure:** Template-based prose injection for Llama 3.2:3b scored **48.5%**, performing worse than raw wavefront (60.6%) and baseline (66.7%).
*   **Hypothesis:** Basic template-based synthesis clutters the prompt. The 3B model is strong enough to knowledge-mine the raw neuron list better than clunky templates.
*   **Verified Extractor:** The `v2-clean` extractor remains verified as the production standard.

---

## Next Steps (Agreed)
1. Launch the full 47-chapter ingestion into `data/biology_full_v2_clean.db` using the verified `v2-clean` extractor.
2. Refine the reader pipeline: try **Neural Synthesis** (115M synthesizer model) or **Fine-tuning** the cortex to actually gain lift on 3B+ models.
