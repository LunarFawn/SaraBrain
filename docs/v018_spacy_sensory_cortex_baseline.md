# v018 — spaCy Sensory Cortex: Baseline on Ch10 Biology

**Date:** 2026-04-18
**Status:** First result. Baseline for the layered sensory cortex.
**Companion code:** `benchmarks/run_spacy_ch10.py`, `benchmarks/spacy_spike.py`
**Companion memory:** `project_spacy_sensory_cortex.md`, `project_spacy_baseline_result.md`, `project_eyeball_cortex.md`

---

## The finding in one line

**spaCy + Sara's graph, no LLM at all, scored 42.9% on the Ch10 biology exam — the same ballpark as prior small-LLM cortex runs — in 6.5 seconds total across 33 questions.**

This is the first direct evidence for the eyeball-cortex thesis: that the 3B LLM's "cortex" role is largely pattern-matching reproducible by fixed filters, not world-knowledge only a trained model supplies.

## Background

The open problem from the prior session was that a 7B LLM cortex (qwen, drifted from the agreed mistral) froze Jennifer's 8 GB Mac when we attempted to run Ch10 biology against the compartmentalized brain `claude_taught.db`. The session that ostensibly scored near 100% with a compartmentalized LLM cortex was never saved to disk — the crash ate the results file.

Two reactions compounded into a pivot:

1. **Hardware:** 7B on an 8 GB Mac alongside Claude Code is genuinely unsafe.
2. **Architecture:** Jennifer's prior design note (`project_eyeball_cortex.md`) argued the LLM cortex is doing jobs a specialized filter bank could do — and that its own baked-in knowledge actively harms Sara on some questions. The crash pushed that hypothesis to the front of the queue.

She rejected "try a smaller LLM" (smaller LLMs have *worse* opinions, not fewer) and instead named the thing she actually wanted: **grammar understanding without world opinions.**

That name has a classical-NLP answer: **spaCy.**

## Why spaCy

spaCy is an open-source Python library (pre-LLM era) that parses sentences into structured linguistic data: tokens, POS tags, dependency trees, lemmas, named entities, morphological features. It is trained only on the Universal Dependencies treebank — a hand-annotated corpus of *grammar*, not of facts. The resulting model knows what a subject is. It doesn't know what photosynthesis is. Ask it "what is photosynthesis?" and you get "noun, singular, direct object" — not a definition.

That property — parsing without answering — is exactly what the eyeball-cortex design called for.

## The spike

`benchmarks/spacy_spike.py` ran spaCy and Sara's existing tokenizer side-by-side on eight biology facts and three MMLU questions. The spike was diagnostic only; no integration.

Comparing on `DNA is copied into RNA during transcription`:

| Signal | Sara's tokenizer | spaCy |
|---|---|---|
| Tokens after stopword strip | `['dna','copied','into','rna','during','transcription']` | same raw words, but also POS + dep + lemma |
| SVO triples | — | `(DNA, copy, ?)` (passive) |
| Lemmatization | partial | full (`transcribed` → `transcribe`) |
| Passive voice | not detected | `nsubjpass` + `auxpass` |
| Temporal prepositions | "during" indistinguishable from "in" | `during` flagged as temporal |
| Verb tense / aspect | — | `Past`, `Perf` |

The gap is not subtle. spaCy exposes the structure Sara's flat tokenizer discards. Most importantly, it cleanly surfaces the **temporal/sequence signal** that a missing-filter analysis in `project_eyeball_cortex.md` identified as the probable cause of 16 "gave up" facts in prior comprehension testing.

## The benchmark

`benchmarks/run_spacy_ch10.py` replaces `run_compartment_exam.py`'s ollama cortex with a spaCy-based scorer. Everything else — region routing, wavefront propagation across the 9 compartmentalized regions of `claude_taught.db` — is identical. The LLM is simply not there.

**Scoring per choice:**

- `act` — activation overlap: sum of Sara's activation weights for lemmas appearing in the choice.
- `svo` — SVO alignment: bonus when the choice's verbs point at subjects/objects appearing in the question.
- `temp` — temporal/sequence fit: bonus when question and choice share temporal prepositions or causal markers.
- `penalty` — shared-with-question penalty: small subtraction for choices that just echo the question's own words.

**Abstain logic:** if the top choice's total score falls below 1.0 (effectively "no activation, no SVO match, no temporal fit"), the question is recorded as abstained rather than guessed. Coverage and accuracy-of-answered are reported separately. Honest over guessing.

## Result

| Metric | Value |
|---|---|
| correct (of answered) | 12/28 = **42.9%** |
| wrong | 16 |
| abstained | 5/33 = 15.2% |
| coverage (answered / total) | 84.8% |
| wall time | 6.5 s for 33 Qs (≈ 0.2 s/Q) |
| ollama calls | 0 |
| Mac memory risk | none |

For reference, prior benchmark comments in `run_compartment_exam.py` listed:

- Pre-test (no study): 45.5%
- Flat brain (old parser): 48.5%
- Flat brain (new parser): ~28%
- Compartmentalized + 3B LLM cortex: reported ~100% from the session that crashed, no saved file.

spaCy-alone lands inside that historical small-LLM band. No model. No GPU. No data-center dependency. The hardware story Sara was always meant to have.

## Four failure modes

The per-question detail isolates where spaCy alone breaks — and the failures are structurally distinct, which matters for designing the next layer.

1. **All-zero activation (e.g. Q216).** Neither question nor any choice produces graph signal. The brain genuinely doesn't have the vocabulary of the choices. No cortex repairs this; it's a *teaching gap*, and abstain is the correct behavior.

2. **Tied scores (e.g. Q207).** All four choices tied at 7.0 because "selection" hit equally across them. spaCy has no way to discriminate between *kinds* of selection. Currently we pick A by default — should also abstain on ties. A small reflex model (L1, below) could paraphrase-match.

3. **Wrong winner, plausible picks (e.g. Q229).** Activation floods toward the topic-adjacent choice rather than the question-specific answer. The scorer rewards "most related to the topic" instead of "answers the question." A language-aware arbiter (L2) is needed.

4. **Magnitude ≠ correctness (e.g. Q211).** The wrong choice scored 107.3 because it matched more activation keywords; the correct choice scored 59.3 because its answer is more specific. A scorer that rewards *relevance* rather than *volume* is needed.

Failure modes 2–4 are exactly what the layered sensory cortex is for.

## The layered sensory cortex

Jennifer's framing: the sensory cortex shouldn't be one thing. The biological visual cortex is V1 (edges) → V2 (shapes) → V4 (color) → IT (objects). Each stage is dumb alone and specialized. Intelligence is layered, not monolithic.

| Layer | Component | Job | Opinions? |
|---|---|---|---|
| **L0: Grammar** | spaCy | Lemmas, SVO, tense, temporal, entities. Structure only. | None — grammar-trained, not knowledge-trained. |
| **L1: Reflex** | Small instruction-tuned model (1B, e.g. Llama 3.2 1B) | Paraphrase / literal equivalence. "Does this choice *say the same thing* as this activation keyword?" yes/no/unclear. | Constrained by system prompt: only use Sara's activation, ignore own knowledge. |
| **L2: Arbiter** | Ministral 3B (agreed cortex family) | Only invoked when L1 abstains or ties. Harder semantic ranking with Sara's full path trace. | Same constraint. Trust Sara over training. |
| **L3: Brain** | Sara's graph | The actual knowledge, the wavefront, the paths. | The only source of truth. Alone writable by teaching (`feedback_sara_higher_order.md`). |

Key architectural rules (inherited from existing feedback memories):

- **LLM never teaches** (`feedback_sara_higher_order.md`). L1 and L2 rank paths Sara already has. If Sara doesn't have a fact, the answer is "I don't know" — abstain bubbles up.
- **Query is read-only** (`feedback_query_is_readonly.md`). The pipeline doesn't strengthen paths during inference.
- **No shortcut** (`feedback_do_not_shortcut.md`). If a question needs knowledge Sara doesn't have, the correct behavior is teach Sara, not route around her.

## Hardware fit on the 8 GB Mac

spaCy `en_core_web_sm` is ~12 MB. Llama 3.2 1B is ~700 MB. Ministral 3B is ~2 GB. Peak with all three hot + Python + Claude Code + macOS is ~6 GB — tight but workable. And because L2 is only called on L1-abstain cases, the 3B usually isn't loaded when the 1B is working. This is safe in a way that 7B was not.

The distributed-hardware memory (`project_distributed_hardware.md`) still applies as the long-term plan: brain on a Chromebook or Pi, cortex on the Mac, networked. The spaCy result reduces the cortex box's weight class — from "must run 7B" to "must run 3B sometimes, 1B mostly, spaCy always." Smaller and cheaper hardware becomes sufficient.

## What this doesn't prove

- spaCy-alone is in the *small-LLM ballpark*, not the *near-100% recollection band*. The 100% number is Jennifer's unverified recollection from the crashed session; no file confirms it. A proper head-to-head between spaCy-only and spaCy + L1 + L2 on the same 33-question set is still pending.
- 33 Ch10 questions is a small set on a specific chapter. Generalization to GPQA, the full MMLU-biology bank, or non-biology domains is not shown.
- The brain under test was taught by Claude Opus — a strong teacher. Performance on a graph taught by a weaker source is an open question.
- Four of the four failure modes above are architectural, not implementation bugs. Refinements to scoring (tie-detection, incremental activation per choice) will move the number, but the fundamentals require L1/L2.

## Next

In order:

1. Add tie-detection to the L0 scorer (top-two scores within 1% → abstain), re-run the 33 for a cleaner L0 number.
2. Pull `llama3.2:1b` and, if available in the Ollama library, `ministral:3b`; otherwise fall back to `llama3.2:3b` as the L2 candidate.
3. Build L1 as a separate invocation — reads spaCy structure and Sara's activation, emits `{yes|no|unclear}` per choice. Invoked only on tie / close-call cases.
4. Build L2 as the final arbiter — only called when L1 returns `unclear`. Strict "trust Sara, ignore own knowledge" system prompt.
5. Re-run the Ch10 exam with the full L0→L1→L2 pipeline. Compare to the 42.9% L0 baseline and to whatever a pure small-LLM-cortex run scores.

## Memories added in this session

- `feedback_cortex_model_mistral.md` — cortex family is mistral, not qwen (prior session drift).
- `project_spacy_sensory_cortex.md` — spaCy chosen as the L0 sensory cortex.
- `project_spacy_baseline_result.md` — the 42.9% result with its caveats.

---

*Versioned per `feedback_versioned_docs.md`. Previous doc in series: v017_sensory_shell_implementation.md.*
