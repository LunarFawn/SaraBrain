# v025 — HamlinLLM-v0.1 Status & Handoff

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex` (pushed to origin)
**Implements:** Phase 1 of [v024_organ_architecture_plan.md](v024_organ_architecture_plan.md)
**Model name:** HamlinLLM-v0.1 — named after Hamlin Robinson School (the Seattle dyslexia school)

This doc is a self-contained handoff so any future session can pick up
the work without re-reading the whole conversation history.

---

## What HamlinLLM is

A from-scratch grammar transformer (125 M params) plus small task heads
that together replace `llama3.2:3b` (router) and Claude Haiku
(synthesizer) inside Sara's `sara_reader` loop. No LLM, no API key, no
external service. The model never holds world knowledge — Sara's
substrate (`brain.db`) is the source of truth.

---

## Quick start (the only commands you need)

```bash
# one-time env (already done if .venv exists)
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install datasets sentencepiece tqdm spacy
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/pip install -e .

# interactive chat (the main UI)
.venv/bin/python -m sara_brain.cortex.transformer.chat --brain /tmp/sara_demo.db

# or through Sara's existing CLI
.venv/bin/python -m sara_reader.cli_stateless \
  "what is newton's first law" \
  --brain /tmp/sara_demo.db \
  --cortex-router --cortex-synthesizer

# train the grammar LM from scratch (~50 min on a 3070)
PRESET=base STEPS=20000 ./scripts/train_grammar.sh

# train the router head on a Sara brain (~4 min)
.venv/bin/python -m sara_brain.cortex.transformer.train_router \
  --grammar-ckpt src/sara_brain/cortex/checkpoints/grammar_base_015000.pt \
  --brain /tmp/sara_demo.db
```

The `/tmp/sara_demo.db` is a copy of `aptamer_full.db.bak` with a `.db`
extension (Sara's loader requires `.db` or directory).

---

## What's done

**Trained models** (in `src/sara_brain/cortex/checkpoints/` — gitignored):
- `grammar_base_015000.pt` — 125 M, dev perplexity **2.806**
- `router_head.pt` — 4-class, dev accuracy **92.2 %** (was 90.0 % before "how does X work" template retraining)
- `grammar_prod_*.pt` — 300 M scaling probe, best dev ppl 2.915 (worse than 125 M; see scaling note in cortex README)

**Code** (all in `src/sara_brain/cortex/transformer/`):

| File | Purpose |
|---|---|
| `vocab.py` | 76-token structural vocab (UPOS + UD deps + slots) |
| `model.py` | TransformerBlock + GrammarModel + tiny/base/prod presets + `encode_hidden()` for downstream heads |
| `ud.py` | UD treebank ingestion (6 English treebanks) |
| `synthetic.py` | Grammar-LM dataset + LM batch generator |
| `train.py` | Grammar-LM trainer (bf16, cosine LR, dev-ppl eval, resume) |
| `inference.py` | Sample / score the trained grammar LM |
| `router_data.py` | Substrate-driven labeler for the router head |
| `router_head.py` | Frozen-encoder + small classifier |
| `train_router.py` | Router-head trainer |
| `router_args.py` | Rule-based arg extractor (concept/type/label/term) |
| `router.py` | `CortexRouter` — full router pipeline |
| `synthesizer.py` | Template renderer (substrate edges → prose) |
| `synth_data.py` | Labeler: emits (edges, prose) pairs for the future neural synthesizer |
| `clarify.py` | Wh-typo + concept-typo did-you-mean state machine |
| `dig.py` | Sibling expansion + comprehensive-intent detection |
| `chat.py` | Interactive REPL (the main user-facing UI) |
| `ask.py` | Standalone one-shot CLI (router + substrate, no chat) |
| `README.md` | End-user docs |

**Reader integration** (`src/sara_reader/`):
- `stateless_reader.py` — `--cortex-router-ckpts`, `skip_synthesis`, `cortex_synthesizer` kwargs
- `cli_stateless.py` — `--cortex-router`, `--cortex-synthesizer`, `--no-synthesis`, `--grammar-ckpt`, `--head-ckpt` flags
- `tools.py` — `_DEFINITIONAL_RELATIONS` expanded with `states`, `also_known_as`, etc.; output preserves `_attribute` suffix

---

## Chat REPL commands (cheat sheet)

```
/help              show this
/teach STATEMENT   teach a fact (brain.teach)
/refute STATEMENT  refute a fact
/dig               expand last query — pull substrate siblings
/dig CONCEPT       drill into a named concept
/depth N           re-run last query at hop distance N
/trace             toggle routing decision + classifier confidence
/verbose           toggle raw substrate output (skip synthesis)
/brain PATH        switch brain.db without restarting
/model             show loaded checkpoints
/quit /exit Ctrl-D leave
```

Natural-language phrases ("tell me everything about X", "complete picture
of X") auto-trigger the same expansion as `/dig`.

---

## Open issues / next steps

1. ~~**Template gaps**~~ — fixed 2026-05-03. `is` and `have` added to
   both `_TEMPLATES` and `_ATTR_TEMPLATES`. Stop-word subjects (e.g.
   `"in"` from `is_part_of` over multi-word labels) are now dropped at
   `render_edges` time using `dig._STOP_WORDS`. Cluster
   capitalization in `render_edges` also fixed — every sentence
   capitalizes its leading letter, not just the first per source cluster.
2. **Neural synthesizer head** — labeler exists (`synth_data.py`,
   produces ~634 pairs from the aptamer brain), but no training loop
   yet. Same architecture pattern as `router_head.py`: frozen grammar
   LM + small generative head. Would replace template-renderer for
   variety; templates stay as the labeler.
3. **Define / explore confusion** still ~9 % on dev; some "what is X"
   templates routed to explore. Adding more disambiguating templates
   to `router_data.py` and retraining would tighten this.
4. **Multi-concept questions** ("explain X in the context of Y")
   currently pick one concept. `/dig` + comprehensive-intent partly
   addresses this; a real fix is multi-hop reasoning that picks 2+
   concepts up front.
5. **Eyes / Ears / Hands organs** (Phases 2–4 of v024) — not started.

---

## User preferences (read these before assisting)

Two memory rules under `~/.claude/projects/.../memory/`:

- **Always ask before saving memory.** Never persist memory
  proactively. Propose, wait for explicit yes.
- **Do what the user asks, nothing more.** No caretaker framing, no
  pacing/wellbeing editorializing, no "good stopping point" / "sleep on
  it" / time-of-day comments. List options, ask, do what they say.
  User has autism + dyslexia; do not pattern-match neurotypical
  signals onto them.

---

## Git state

Branch `feature/grammar-cortex` based on `feature/reader-sdk` (which
holds the v024 plan).

Recent commits:
```
8388ea1 dig: filter substrate-wide words from sibling matches
867ee2d chat: add /dig, /depth, and comprehensive-intent auto-expansion
d7cd639 end-to-end fixes for "what is newton's first law" style queries
2d25311 router: handle "how does X work" question shape
b0aa5eb chat: arrow-key history via readline + persistent ~/.hamlinllm_history
5c02cd9 synthesizer: invert templates for _attribute-targeted edges
5cdcf40 chat: add /teach and /refute slash commands
32c99f1 chat: clarification flow — fuzzy typos without assumption
bbf0e12 cortex: interactive HamlinLLM chat REPL
b84da21 cortex: name the model — HamlinLLM (named after Hamlin Robinson School)
```

Push from a terminal with creds:
```bash
git push origin feature/grammar-cortex
```

---

## The scaling result (worth keeping)

Trained two sizes on the same 1.3 M-token corpus (6 English UD treebanks):

| Model | Best dev ppl | Train ppl at end |
|---|---|---|
| 125 M base | **2.806** | ~1.6 |
| 300 M prod | 2.915 | ~2.7 |

Bigger model, same data → worse generalization, more memorization. The
grammar task has a low entropy ceiling 125 M already approaches.
Consistent with the v024 small-organ thesis.
