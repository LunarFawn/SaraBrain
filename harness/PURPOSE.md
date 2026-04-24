# Purpose — Why This Directory Exists

## The short version

`sara_test/` is an isolated harness for measuring whether Sara Brain has successfully transferred knowledge to an LLM. It exists separately from `sara_brain/` because opening a fresh Claude Code session inside the main repo gives the model filesystem access to the source documents, teach scripts, and draft papers — which silently defeats the measurement. A session opened in `sara_test/` has nothing on disk except the MCP config and the committed brain substrates, so the LLM must actually use Sara for retrieval.

## The longer version

### Sara as a measurement instrument

See `sara_brain/papers/sara_as_instrument_draft_v1.md` (Pearl 2026f) for the full argument. The short form:

Large language model research has lacked a fine-grained measurement instrument for transformer behavior — an apparatus that distinguishes retrieval from synthesis from training recall from context leakage at the token level. Benchmarks measure outputs against targets; they don't expose the internal dynamics of a model reading from a substrate.

Sara Brain is a path-of-thought knowledge graph: persistent, structured, and inspectable per-triple. When you teach Sara a specific set of triples and then ask an LLM a question answerable from those triples, you can grade the LLM's answer against the graph's actual contents. That grading — which triples were retrieved, which were paraphrased, which were invented — is *diagnostic*. It tells you where in the pipeline the LLM went right or wrong, not just whether it got the right answer.

For that measurement to work, four properties must hold of the substrate:

1. **Finite** — the complete contents must be enumerable by a human investigator.
2. **Structured** — each fact must be addressable as a (subject, relation, object) triple.
3. **Large enough** — the substrate must support nontrivial multi-hop questions.
4. **Training-orthogonal** — the substrate's contents must not be in the LLM's training data at test time.

Sara satisfies all four by construction. A Sara substrate is built at runtime by teaching, so the human teacher knows every triple. Triples are the storage primitive. Substrates scale to thousands of triples without losing inspectability. And the content comes from newly-written or newly-published source material that postdates the LLM's training cutoff.

### Why an isolated directory

Property 4 (training-orthogonality) is necessary but not sufficient for a clean measurement. Even if the substrate content isn't in the LLM's training weights, it might still be available to the LLM via paths *other than Sara*:

- **Filesystem access.** Claude Code's `Read` / `Grep` / `Glob` tools are scoped to the working directory. If the working directory is `sara_brain/`, the model can read `papers/aptamer_rev1/aptamer_paper_full.txt` directly and answer from that, never calling Sara. Observed empirically on 2026-04-23: Haiku gave a detailed §8-level answer by running `grep` against the paper file instead of querying Sara.

- **Auto-memory.** Claude Code maintains per-project memory scoped by working directory path. The memory for the `sara_brain/` project contains notes written during development — including aptamer-related feedback and project memories from today's sessions. Those auto-load when a Claude Code session opens in that directory, contaminating the context before the first question is asked.

- **Conversation context.** Any session that has previously discussed the substrate has those tokens in context. A session that teaches Sara and a session that tests Sara cannot be the same session — the teaching text poisons the testing.

Each of these is a distinct contamination vector. See `sara_brain/papers/model_infections_draft_v1.md` for the full taxonomy (keyword, narrative, tool-use, persona, user-input, cross-session memory, session-context, and filesystem contamination — seven types catalogued so far).

`sara_test/` is the operational response to the filesystem and auto-memory vectors. Opening a Claude Code session here gives the model access only to the committed brain in `brains/`, served through MCP. Nothing readable from the filesystem reveals the substrate content — only Sara does.

### What this directory is NOT

- **Not a replacement for grading.** The harness controls what the LLM has access to; it does not grade the LLM's answer. Grading is a separate step done by the human evaluator (or by another LLM with its own isolation).
- **Not a defense against all contamination.** Training-weight bias and user-input contamination persist regardless of the harness. Interpretation-layer and rendering-layer biases (e.g., the "SNARE → vesicle protein" auto-disambiguation observed with Haiku) can still occur. See `PROTOCOL.md` for mitigations.
- **Not a permanent arrangement.** A substrate used here today may be in training data in six months. The harness produces measurements valid *at time of test*; the instrument paper will argue that continuous generation of fresh substrates is part of the method, not a flaw in it.

### Relationship to the main repo

| | `sara_brain/` | `sara_test/` |
|---|---|---|
| **Role** | Development: building Sara, writing code, teaching new substrates, drafting papers | Measurement: isolated harness for testing whether substrates transfer via Sara |
| **What's on disk** | Full code, papers, drafts, teach scripts, source documents | MCP config, loader script, committed brain substrates, documentation |
| **Auto-memory scope** | All feedback/project memories from cumulative development sessions | Empty at first session; accumulates only test-related memory |
| **Who works here** | The author (developing) and the author's assistants | Fresh LLM sessions running tests, grading assistants |
| **Public visibility** | Private during development, public upon paper release | Private during development, public upon instrument paper release alongside the substrates it hosts |

The two directories are **siblings**. `sara_test/` does not contain source code; it imports the MCP server from `sara_brain/` via absolute paths. The `load_brain.sh` script reads brain DBs from `sara_brain/` and installs copies in `sara_test/brains/` where they are then tracked in git. Once loaded, a test session in `sara_test/` runs entirely without needing `sara_brain/` to be reachable — the MCP server is a subprocess that opens the committed brain via absolute path.

### Reading list — the context you need to understand this harness

1. **`PROTOCOL.md`** (this directory) — day-to-day workflow and measurement protocol.
2. **`sara_brain/papers/sara_as_instrument_draft_v1.md`** — the full methodological argument for Sara as instrument.
3. **`sara_brain/papers/model_infections_draft_v1.md`** — the catalog of contamination vectors this harness defends against.
4. **`sara_brain/papers/training_corrupts_reading_draft_v1.md`** — the empirical finding that motivated the harness (Haiku vs. Opus comparison).
5. **`sara_brain/papers/teaching_vs_training_draft_v2.md`** — the prior paper (Pearl 2026b) establishing the teaching-side analog of the reading-side failure mode.
