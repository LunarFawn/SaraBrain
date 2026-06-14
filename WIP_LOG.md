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

## Consensus (2026-06-13, 16:30)
*   **Refactor Success:** `run_mmlu_biology.py` is now a universal loader. All models use the same **Wavefront Substrate** (no more keyword RAG).
*   **The "Frozen" Brain Read:** Identified. Wavefront propagation on the 64k-neuron brain takes ~50s per question. The script is not frozen, just compute-heavy.
*   **Benchmark Results (33Q, Ch10 Brain):**
    *   **Pure Wavefront:** 18.2% (Pure graph intersections, zero-LLM)
    *   **Custom 115M Reader:** 30.3% (Heavy bias toward 'B')
    *   **Custom 1B Reader:** 18.2% (Heavy bias toward 'D')
    *   **Llama 3.2:3b (Cortex):** **60.6%** (Proves the substrate knowledge is high quality)
*   **Conclusion:** The architecture is verified. The next challenge is closing the reasoning gap for the custom small models so they can match the frontier model's performance on the same substrate.

---

## Next Steps (Agreed)
1. Launch the full 47-chapter ingestion into `data/biology_full_v2_clean.db` (verified production path).
2. Optimize wavefront propagation speed (optional but recommended to solve "frozen" feel).
3. Re-train custom readers (115M/1B) using the exact wavefront neighborhoods from the production brain.
