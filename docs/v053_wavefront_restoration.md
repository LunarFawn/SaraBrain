# v053 — Wavefront restoration: what we found, what we did, what's still open

**Date:** 2026-05-10
**Branch:** `feature/v052-local-ollama-cortex`
**Trigger:** User asked "are we better or worse since the initial commits?"
referring to April 2026 work on physics/biology benchmarks.

This is a retrospective. The work in this session split into two halves:
the extractor iteration (aug2→aug9, documented in
[hamroby_extractor_v1_iteration_log.md](features/hamroby_extractor_v1_iteration_log.md))
and a comparison to the April-era benchmark performance, which exposed
deeper architectural drift. This doc records the second half.

## What the user remembered

The April 12-16, 2026 benchmark
([docs/benchmark_report_2026-04-12-16.md](benchmark_report_2026-04-12-16.md))
recorded **80% on a 10-question MMLU biology subset** with qwen2.5-coder:3b
as the cortex and a hand-curated layered brain (dictionary + vocabulary
+ science + biology regions). The brain alone (no LLM) scored 50%;
qwen 3b alone scored 58.4%. The 80% was emergent from the combination.

The user remembered that performance and felt the current system was
worse. The investigation confirmed it was.

## What we found — three compounded regressions

### 1. Chat layer: wavefront demoted from primary query mechanism (v050/v052)

**Pearl 2026a / rev8 §2.4** (the foundational paper) defines Sara's
retrieval as "parallel wavefront propagation across the graph rather
than embedding-similarity search. Seeded neurons emit wavefronts;
convergence points become recognition results."

That commitment was preserved in the brain code (`sensory/shell.py`,
`brain.propagate_into()`, `brain_recognize` MCP tool — all still
present, untouched). But the **chat layer** drifted across v047-v052:

- **v050** ("two-layer realignment") restored the cortex/hippocampus
  split from the papers. Listed the brain tools as peers:
  `brain_explore, brain_value, brain_define, brain_why, brain_trace,
  brain_recognize, brain_did_you_mean`. The intent was correct
  (paper-aligned hippocampus surface). The implementation flattened
  wavefront into one menu option.

- **v052** ("local Ollama cortex with force-Sara strict mode") built
  `cli_stateless_chat` + `StatelessReader` around an LLM-as-orchestrator
  pattern: the LLM picks tools from the menu per question. The
  motivating concern — preventing within-session hallucination
  contamination — was real. The chosen solution made the LLM the
  retrieval-method-chooser. The LLM (cortex router or small Ollama
  model) routinely picks `brain_define` for "what is X" questions.
  `brain_define` filters to specific definitional relations (`is_a`,
  `defined_as`, `synonym_of`) that prose-extracted brains don't
  produce. Result: "the substrate does not contain this information"
  on a brain with rich substrate.

  The wavefront engine never auto-runs from the chat path. The April
  flow — extract concept seeds from the question, propagate, find
  intersections, score choices — is unreachable through `cli_stateless_chat`.

The user's reaction when this was identified:
> "the brain without wavefront is not a brain. it cant be deprioritized
> as its a core function"

That's the architectural principle. Saved as memory rule
`feedback_wavefront_is_the_brain.md` so future Claude sessions cannot
repeat the drift.

### 2. Workflow drift: dictionary scaffolding stopped being part of the standard recipe

The April benchmark used a brain with the **Moby Thesaurus II** loaded
as a dictionary region (62k neurons, 862k synonym edges). The synonym
bridges let the wavefront connect question vocabulary ("tallest") to
substrate concept labels ("extreme phenotype", "directional selection").

`benchmarks/build_dictionary.py` is still in the repo, untouched.
`data/moby_thesaurus.txt` (30,259 entries) is still in the repo. The
infrastructure was never removed.

What changed was the workflow. Starting around v033-v047, the standard
ingest path became `cli_teach_book BOOK --brain X.db` — which does NOT
auto-load the dictionary. Building a usable brain required running
`build_dictionary.py` first, separately. That step quietly stopped
being part of the recipe. The drift was not documented anywhere
because nothing was removed; it was just left out.

Measured impact: rebuilding the bio_10q brain via `cli_teach_book` with
the trained extractor produced a 169-neuron brain. Wavefront on that
brain scored 1/10 with 9 abstains.

### 3. Measurement framing: the April 80% was a wavefront-mechanism test, not a biology-knowledge test

This was a finding the user surfaced while debugging the comparison.
`benchmarks/bio_10q_facts.txt` (59 lines) was curated to contain the
bridge facts needed to answer the 10 MMLU questions:

```
# Q0: directional selection
directional selection is selection for one extreme phenotype
stabilizing selection is selection for the average phenotype
...
```

The April brain had these facts directly taught into the substrate.
The 80% measured whether the wavefront could find them through
synonym bridges. It was a valid measure of the wavefront's bridging
capability, not of biology knowledge.

`benchmarks/biology_facts.txt` (510 lines) covers gene / phenotype /
natural-selection material more broadly but does NOT contain the
specific bridge facts ("directional selection", "convergent evolution",
"electron transport", "meristem", "barr body" — 0 mentions each).
There's no checked-in standalone source for the April vocabulary
(184 neurons) and science (113 neurons) bridge layers — that content
was hand-curated.

## What we did (this session)

### Memory rules (architectural commitments saved across sessions)

| rule | purpose |
|---|---|
| [`feedback_wavefront_is_the_brain.md`](../../home/grizzlyengineer/.claude/projects/-home-grizzlyengineer-repo-SaraBrain/memory/feedback_wavefront_is_the_brain.md) | Wavefront is the brain's defining function. Cannot be one tool among many. The v050/v052 mistake. |
| [`feedback_simpler_models_substrate_faithful.md`](../../home/grizzlyengineer/.claude/projects/-home-grizzlyengineer-repo-SaraBrain/memory/feedback_simpler_models_substrate_faithful.md) | For Ollama synthesis on prose brains, smaller models (llama3.2:3b) are more substrate-faithful than larger (qwen 7b, llama 8b). |
| [`feedback_dont_tell_user_to_stop.md`](../../home/grizzlyengineer/.claude/projects/-home-grizzlyengineer-repo-SaraBrain/memory/feedback_dont_tell_user_to_stop.md) | When the user expresses fatigue/frustration, do not prescribe rest. State technical reality, offer options. |

### Code changes

| commit | what |
|---|---|
| `dcd8f0a` | `stateless_reader.py`: wavefront-first restoration. Every `ask()` runs `brain.propagate_into` on extracted seeds before any LLM-driven tool selection. The wavefront's convergence map + intersections become the primary substrate the synthesizer grounds in. Tool-call routing retained for supplementary drill-downs (e.g., `brain_value` for quantities), never as replacements for wavefront. |
| `9a9d510` | `src/sara_brain/bootstrap.py` (new) + `cli_teach_book.py`: auto-bootstrap Moby Thesaurus II into new brains. Idempotent (skipped if `synonym_of` segments exist). Opt-out flag `--no-dictionary`. Single-command ingest now produces a layered (dictionary + content) brain. |

### Verification of code changes

On `/tmp/jkd_hybrid.db` (the JKD-book prose brain) **with** wavefront-first
chat:

```
Q: "who is bruce lee"
A: "Bruce Lee was a person. He had a kidney damaged, and his face was a fear
   on many people. He was capable and had continuous growth as a necessity
   for him. ..."
```

All bolded details ("kidney damaged", "face was a fear", "continuous growth
as a necessity") are actual substrate content from the book. The chat
layer is now grounded.

On `/tmp/bio_10q_dict.db` (the 10Q-fact brain + auto-loaded dictionary):

```
Q: "what is directional selection"
A: "Directional selection is a type of selection that acts on one extreme
   phenotype, resulting in a strength of 1.00. It is also referred to as
   'selection for one extreme phenotype'."

Q: "what produces ATP"
A: "ATP is produced through oxidative phosphorylation."
```

Both grounded in the bio_10q_facts content via wavefront convergence.

## What's still open

### The "honest biology recall" question is unanswered

The April 80% was a wavefront-mechanism test (could it find taught
bridge facts via synonym walks?). It was a valid measure of the
mechanism but not of biology knowledge.

A meaningful biology-knowledge test would require:
- A brain trained on broader textbook content (e.g., the
  `benchmarks/biology2e_facts/ch01-ch10` chapter files, ~500 lines
  each).
- Questions whose answers are not directly stated as facts in the
  taught content — they require the wavefront to reason through
  associative paths rather than retrieve a memorized triple.

We did not run that test in this session.

### Vocabulary / science bridge layers not checked in

The April brain had 184 vocabulary neurons ("Tallest" means "most
extreme in height") and 113 science neurons ("Phenotype is an
observable trait") that bridge between dictionary synonyms and
domain facts. The source content for these layers was not checked
into the repo. Recreating them would require:

- Either rebuilding from memory of the April session.
- Or generating via Claude / similar from the bio_10q questions.
- Or accepting that fa3f484's flat-brain shift made compartmentalized
  bridge layers unnecessary if dictionary + content already covers
  the bridging.

### fa3f484 architectural choice still in play

The flat-brain + non-propagating-IS-A shift (April 19) was a real
architectural decision, justified by polysemy handling. This session
did not revisit that choice. It remains the architecture.

### Synthesis model variance

Even with wavefront-first chat and rich substrate, synthesis quality
varies by Ollama model: `llama3.2:3b` produces substrate-grounded
narrative answers but is sometimes lyrical; `qwen2.5:7b-instruct`
produces enumerative list-style answers but more often refuses with
"the substrate does not contain this information." Neither is
strictly "better" — they serve different question types.
`--strict-sara` mode is too rigid for prose-ingested brains regardless
of model size.

## Lessons from this session

1. **Workflow drift is invisible in code review.** No commit said
   "stop loading the dictionary." It just stopped being part of the
   standard ingest pipeline as new entry points were built. The
   regression compounded over weeks. Single-command ingest now bakes
   the dictionary back in so this can't repeat silently.

2. **Tool-menu architectures hide their primary subject.** When
   `brain_recognize` (wavefront convergence) was listed as one tool
   option alongside `brain_define / brain_explore / brain_value /
   brain_why / brain_trace`, the LLM rarely picked it. The flat menu
   contradicts the architectural commitment regardless of what the
   docs say.

3. **Benchmark scores can measure mechanism quality, not knowledge.**
   The April 80% was a wavefront-bridging test on hand-curated facts.
   Re-running it on a today brain with the same facts loaded is a
   valid mechanism verification but not a knowledge test. Future
   benchmarks should separate the two.

4. **The brain's published architecture has not drifted.** Wavefront,
   IS-A hierarchy, multi-threshold cascade, short-term scratchpad —
   all still in `src/sara_brain/core/`. The drift was always at the
   workflow / chat-layer boundary, not in the substrate itself.

## Pointers

- Plan file (resumption-safe): `~/.claude/plans/wavefront-restoration-v053.md`
- Architectural memory rule: `feedback_wavefront_is_the_brain.md`
- Extractor iteration log (separate work this session):
  [features/hamroby_extractor_v1_iteration_log.md](features/hamroby_extractor_v1_iteration_log.md)
- April benchmark (the baseline this session compared against):
  [benchmark_report_2026-04-12-16.md](benchmark_report_2026-04-12-16.md)
- Wavefront-first chat: `git show dcd8f0a`
- Dictionary auto-bootstrap: `git show 9a9d510`
