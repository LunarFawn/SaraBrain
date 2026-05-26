# Sara Brain — LLM Orientation

**For:** any LLM session picking up this project mid-stream
(2026-05-12 snapshot). Read this in full before touching code.

---

## 1. What Sara Brain is (architectural commitments)

Sara Brain is a **persistent path-of-thought knowledge store** that
pairs with a stateless frontier LLM. From [Pearl 2026a §7.3](papers/zenodo_v1.1_preprint.md):

> "Transformers are the best sensory processing system ever engineered.
> They process; they do not store. They are stateless; they do not
> accumulate. They infer; they do not remember. Sara Brain is an
> attempt to build the cognitive system that sensory processing feeds —
> not to replace LLMs, but to give them a persistent memory, a
> traceable knowledge store, and a hardwired ethical layer."

**Two-layer architecture** (per [docs/v050_two_layer_realignment.md](v050_two_layer_realignment.md)):

- **Cortex** = LLM (frontier, frozen, stateless, does language).
- **Hippocampus** = Sara Brain (SQLite path graph, persistent triples).

**The brain's defining mechanism is wavefront convergence**
([Pearl 2026 rev8 §2.4](papers/zenodo_v1.1_preprint.md)):

> "Sara's retrieval uses parallel wavefront propagation across the
> graph rather than embedding-similarity search. Seeded neurons emit
> wavefronts; convergence points become recognition results. The
> reader LLM receives a structured neighborhood of related triples.
> The reader must do its own selection."

This is **not negotiable**. There is a memory rule
[`feedback_wavefront_is_the_brain.md`](../../home/grizzlyengineer/.claude/projects/-home-grizzlyengineer-repo-SaraBrain/memory/feedback_wavefront_is_the_brain.md)
that forbids demoting wavefront to "one tool among many" — the
session that wrote this orientation is recovering from exactly that
architectural drift.

## 2. Mental model in one paragraph

The substrate stores triples as **property → relation → concept**
paths (see `_build_chain` in
[src/sara_brain/core/learner.py](../src/sara_brain/core/learner.py)).
Queries propagate wavefronts FROM observable properties (the
question's content words) TOWARD identified concepts. Convergence
(intersections of multi-seed wavefronts) IS the recognition signal.
The LLM cortex synthesizes natural language FROM that convergence
output — it does not pick retrieval mechanisms; the brain runs its
native query and hands the LLM a structured neighborhood.

## 3. Where the project is right now (2026-05-12)

**Branch:** `feature/v052-local-ollama-cortex`
**Latest commits:**

- `066e162` — v053 retrospective doc explaining the recovery
- `9a9d510` — Moby Thesaurus II auto-bootstrap on new brains
- `dcd8f0a` — wavefront-first restored in `StatelessReader.ask()`
- `b186511` — `cli_teach_book --extractor hybrid` mode
- `2b71ed7` — trained-head extractor as drop-in for rule stub
- `1c2fa29` — `hamroby_extractor_v1` aug9 promoted to canonical
- `fe73ebe` — extractor iteration log

**Two parallel recovery threads from the previous session:**

1. **Extractor work (aug2 → aug9):** the trained head got multi-object
   Pair labels, gold UD features in real-prose pairs, spaCy `sm+trf`
   cascade. Detailed retro in
   [features/hamroby_extractor_v1_iteration_log.md](features/hamroby_extractor_v1_iteration_log.md).

2. **Wavefront restoration:** the chat layer (`cli_stateless_chat` /
   `StatelessReader`) had drifted in v050/v052 to route through an
   LLM-picks-tools-from-a-menu pattern that effectively bypassed the
   wavefront engine. Retro in
   [v053_wavefront_restoration.md](v053_wavefront_restoration.md).

**Built but unmeasured:** `/home/grizzlyengineer/repo/debug_sara/sara_bio.db`
(155MB; 132k neurons; 1M segments; 71k triples) — full Biology 2e
textbook + bridge facts + dictionary. The "fully trained" bio brain
for benchmarking.

## 4. What we were working on when this was written

**Goal:** verify the wavefront-first chat answers biology questions
correctly on the trained brain (a recovery test against the April 2026
80% MMLU-biology baseline).

**What we found:** the wavefront-first chat works architecturally
(verified on small brains — substrate-grounded answers, real book
content surfacing). But on the 132k-neuron bio brain, wavefront
propagation stalls at every depth tested (3, 2, 1) on the very first
question. Root cause diagnosed (not yet fixed): **the seed extractor
generates 27 seeds per question** (content words + bigrams of
adjacent content words). Each common word ("result", "able",
"obtain", "along") has hundreds of Moby Thesaurus synonyms. BFS
explosion.

**The fix the user identified but we haven't implemented:**
**substrate-aware seed extraction.** Filter each candidate seed
through `brain.neuron_repo.resolve(label, exact_only=True)` and keep
only seeds that exist as real neurons. Drops common-English words
that the brain has no concept for; keeps multi-word concept labels
like "directional selection" when the brain has them.

Code lives at
[src/sara_reader/stateless_reader.py](../src/sara_reader/stateless_reader.py),
function `_extract_seed_concepts` (around line 195) and
`_run_wavefront` (around line 720).

## 5. The principles the user has surfaced (load-bearing)

These are in memory rules and they shape future work:

| principle | where saved | what it means |
|---|---|---|
| **Wavefront IS the brain** | `feedback_wavefront_is_the_brain.md` | Cannot be demoted to one tool option among many. The brain's defining function. |
| **Noise IS where the data is** | implicit in design | A depth-3 wavefront returns a structured noisy neighborhood. The LLM is supposed to navigate it. Stripping the "noise" defeats the architecture. |
| **Don't feed MCQ choices to the brain** | rev8 §2.4 | The brain sees the question stem; the LLM selects from choices AFTER. |
| **Smaller models are more substrate-faithful** | `feedback_simpler_models_substrate_faithful.md` | For Ollama synthesis, `llama3.2:3b` > `qwen2.5:7b` on prose brains. Bigger models hedge/refuse more. |
| **Don't ship partial as done** | `feedback_dont_ship_partial_as_done.md` | 3/4 and 3/5 are mid-fix, not "win." Honest reports lead with remaining failures. |
| **Never launch training** | `feedback_dont_launch_training.md` | Stage train commands; the user runs them. (Recurring violation rule.) |
| **No `--quiet` flags** | `feedback_no_quiet_flags.md` | User wants progress visible on long-running ingests. |
| **Don't tell user to stop** | `feedback_dont_tell_user_to_stop.md` | When user expresses fatigue/frustration, state technical reality and offer options — not "you should rest." |

## 6. Critical code paths

When you're trying to understand or modify the system:

- **The substrate's storage direction:** `_build_chain` in
  [src/sara_brain/core/learner.py:95](../src/sara_brain/core/learner.py) —
  triples are stored property → relation → concept. Wavefront walks
  from observable properties toward identified concepts.

- **The wavefront engine:** [src/sara_brain/core/recognizer.py](../src/sara_brain/core/recognizer.py) —
  `_propagate` does BFS with `max_depth` (default 3); `propagate_into`
  is the read-only query API. `_NON_PROPAGATING_RELATIONS = {"is_a"}`
  is intentional (IS-A is for scoring-time inheritance, not
  propagation — see commit `fa3f484` rationale).

- **Chat layer:** [src/sara_reader/stateless_reader.py](../src/sara_reader/stateless_reader.py) —
  `StatelessReader.ask()` is the entry point. Per v053, it runs
  wavefront FIRST automatically (the `_run_wavefront` method around
  line 720), then continues to LLM-driven tool routing for
  supplementary calls. Synthesizer call at the end (strict-sara or
  regular non-strict).

- **Ingest with auto-dictionary:** [src/sara_brain/bootstrap.py](../src/sara_brain/bootstrap.py) —
  `ensure_dictionary(brain)`. Called from
  [src/sara_reader/cli_teach_book.py](../src/sara_reader/cli_teach_book.py)
  unless `--no-dictionary` is passed. Idempotent (skipped if
  `synonym_of` segments exist).

- **Extractors:**
  - Rule stub (deterministic, spaCy + dependency walk):
    [src/sara_brain/cortex/transformer/v2/extractor_rules.py](../src/sara_brain/cortex/transformer/v2/extractor_rules.py)
  - Trained head (BIO span tagger, canonical checkpoint
    `hamroby_extractor_v1.pt`):
    [src/sara_brain/cortex/transformer/hamroby_extractor_v1/](../src/sara_brain/cortex/transformer/hamroby_extractor_v1/)
  - Drop-in trained-head wrapper:
    [src/sara_brain/cortex/transformer/hamroby_extractor_v1/inference.py](../src/sara_brain/cortex/transformer/hamroby_extractor_v1/inference.py)
  - Hybrid mode (runs both):
    `cli_teach_book BOOK --extractor hybrid --brain X.db`

- **Tools the LLM router can pick** (post-wavefront — these are
  supplementary, not primary retrieval):
  [src/sara_reader/tools.py](../src/sara_reader/tools.py) — `brain_explore`,
  `brain_value`, `brain_define`, `brain_recognize`, `brain_did_you_mean`,
  `brain_why`, `brain_trace`.

- **MCP server** (Sara as a tool provider for any LLM client):
  [src/sara_brain/mcp_server.py](../src/sara_brain/mcp_server.py).

## 7. Open challenges (what to work on next)

In priority order:

1. **Implement substrate-aware seed extraction** (the immediate
   unblocker — see Section 4 above; code change is in
   `_run_wavefront` in `stateless_reader.py`). Without this, the
   wavefront-first chat is unusable on large brains.

2. **Re-run the separated bench** at depths 1/2/3 after #1 lands.
   Bench script lives at `/tmp/bench_separated.py` (may need
   restoration if `/tmp` was wiped — full content in the plan file
   `~/.claude/plans/wavefront-restoration-v053.md`).

3. **Compare to April baseline.** The April 12-16 benchmark report
   ([benchmark_report_2026-04-12-16.md](benchmark_report_2026-04-12-16.md))
   shows 80% on a 10Q MMLU bio subset with hand-curated layered brain.
   Note: that was a wavefront-mechanism test on hand-curated bridge
   facts, not a biology-knowledge test (see v053 retrospective for
   nuance).

4. **Cleanup decisions:** the depth-control via `SARA_WAVEFRONT_DEPTH`
   env var is uncommitted in `stateless_reader.py` — decide whether
   to keep it as a knob or revert.

5. **Synthesis quality tuning:** even with good wavefront output,
   small Ollama models sometimes confabulate (Bruce Lee answer mixed
   substrate book details with LLM general knowledge). Larger models
   refuse more aggressively. There's likely a sweet spot somewhere
   not yet found.

## 8. What's working RIGHT NOW (don't break these)

- Wavefront-first chat ON SMALL BRAINS (verified on JKD prose brain,
  bio_10q toy brain — substrate-grounded answers).
- Dictionary auto-bootstrap (Moby Thesaurus II loads into new brains
  by default; ~18s one-time cost).
- Hybrid extractor on prose ingest (multi-object labels, gold UD
  features, spaCy cascade — aug9 canonical).
- Auto-recovery: when `brain_define` returns "no definitional edges,"
  the orchestrator auto-calls `brain_explore(depth=1)` on the same
  concept.

## 9. What's NOT working / known problems

- Wavefront-first chat ON LARGE BRAINS (132k+ neurons): seed
  explosion stalls every depth. Fix identified in Section 4, not yet
  implemented.
- April-style benchmark recovery: not yet measured end-to-end with
  the wavefront-first fix because of #1.
- The cortex router (HamRobyLLM v0.1, single-shot trained tool
  picker) still routes after wavefront ran; its tool selections are
  noisy but don't hurt the answer because wavefront's substrate is
  already in `gathered`.
- `--strict-sara` mode is too rigid for prose-ingested brains
  regardless of LLM size.

## 10. Test environment

Two relevant directories:

- **`/home/grizzlyengineer/repo/SaraBrain/`** — the repo. Where all
  code, docs, and the venv (`.venv/`) live.
- **`/home/grizzlyengineer/repo/debug_sara/`** — empty by design.
  Used for clean-session tests (no CLAUDE.md, no docs that bias a
  fresh Claude Code session). Contains `sara_bio.db` (the trained
  bio brain) for testing without prior-session context.

Running clean test: open Claude Code with `cwd =
/home/grizzlyengineer/repo/debug_sara/`, instruct the new session to
use `/home/grizzlyengineer/repo/SaraBrain/.venv/bin/python -m
sara_reader.cli_stateless_chat --brain ./sara_bio.db
--cortex-router --synthesis-model llama3.2:3b` (or similar). No
SaraBrain docs visible from that cwd.

## 11. Pointers to deeper context

- **Plan file (resumption-safe):** `~/.claude/plans/wavefront-restoration-v053.md`
- **v053 retrospective:** [v053_wavefront_restoration.md](v053_wavefront_restoration.md)
- **Extractor iteration log:** [features/hamroby_extractor_v1_iteration_log.md](features/hamroby_extractor_v1_iteration_log.md)
- **April benchmark baseline:** [benchmark_report_2026-04-12-16.md](benchmark_report_2026-04-12-16.md)
- **v050 realignment (why the two-layer architecture):** [v050_two_layer_realignment.md](v050_two_layer_realignment.md)
- **v052 plan (where the chat layer drifted):** [plans/v052_local_ollama_cortex.md](plans/v052_local_ollama_cortex.md)
- **Memory rules:** `~/.claude/projects/-home-grizzlyengineer-repo-SaraBrain/memory/`

## 12. How to behave in this codebase

A short list of meta-rules from the user's stated preferences:

- **Use the brain's published architecture as ground truth.** When
  the code drifts from the paper, the code is wrong. Pearl 2026a /
  rev8 is the source.
- **State the technical reality plainly.** Don't soft-pedal
  regressions or oversell partial fixes.
- **Save user-stated principles as memory rules** so future sessions
  can't repeat known mistakes (the wavefront demotion is the
  cautionary tale).
- **Don't write decision/planning documents the user didn't ask for.**
  This orientation is the exception — the user explicitly asked for
  it as an LLM handoff.
- **Don't tell the user to stop, rest, or step away.** That's not
  your call.

---

**Last updated:** 2026-05-12 (post-overnight bench attempt, pre-reboot).
**Next action:** implement substrate-aware seed extraction per Section 4.
