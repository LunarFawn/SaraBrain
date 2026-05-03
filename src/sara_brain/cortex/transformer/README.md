# Sara Grammar Cortex

A from-scratch transformer that learns English grammar structure (no
world knowledge) and serves as the foundation for Sara's small organ
heads. Implements Phase 1 of the v024 architecture
([../../../../docs/v024_organ_architecture_plan.md](../../../../docs/v024_organ_architecture_plan.md)).

The premise: Sara already holds all knowledge in her substrate. Each
organ is a tiny transducer that maps between language structure and
substrate operations — it cannot hallucinate facts because facts were
never in its training data.

## What this gives you

Two trained models you can run on a single 8 GB GPU:

1. **Grammar LM (125 M params)** — predicts the next UD grammar tag in
   a sentence. Trained from scratch on six English Universal Dependencies
   treebanks. Use it as a perplexity scorer, a structure sampler, or a
   frozen encoder for downstream heads.
2. **Router head (≈600 K trainable params)** — a 4-way classifier on top
   of the frozen grammar LM that picks one of Sara's substrate tools
   (`brain_value`, `brain_define`, `brain_explore`, `brain_did_you_mean`)
   from the question's structure. Replaces the `llama3.2:3b` routing call
   in `sara_reader/stateless_reader.py`.

A rule-based argument extractor handles concept / type / label / term
extraction from the question, using the spaCy parse and (optionally) the
substrate's own label index.

## Hardware

- Trains comfortably on a 3070 (8 GB). 125 M base preset peaks at
  ~3.7 GB during training (batch 32, seq 96).
- Inference runs on CPU; no GPU needed at serve time.

## Quick start

```bash
# 0. one-time environment
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install datasets sentencepiece tqdm spacy
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/pip install -e .

# 1. train the grammar LM (~50 min on a 3070, all English UD treebanks
#    download automatically on first run)
PRESET=base STEPS=20000 ./scripts/train_grammar.sh
tmux attach -t sara-train         # detach with Ctrl-b d

# 2. train the router head on top of a Sara brain (~4 min)
.venv/bin/python -m sara_brain.cortex.transformer.train_router \
  --grammar-ckpt src/sara_brain/cortex/checkpoints/grammar_base_015000.pt \
  --brain path/to/your/sara.db

# 3. ask Sara a question — no LLM in the loop
.venv/bin/python -m sara_brain.cortex.transformer.ask \
  --question "what do you know about serena rna analysis tool" \
  --brain path/to/your/sara.db \
  --device cpu
```

You can also use the cortex through Sara's existing CLI:

```bash
.venv/bin/python -m sara_reader.cli_stateless \
  "what is the kdoff of super-performing mode" \
  --brain path/to/your/sara.db \
  --cortex-router --no-synthesis
```

## Try the trained grammar LM directly

```bash
# generate plausible English grammar sequences
.venv/bin/python -m sara_brain.cortex.transformer.inference \
  --ckpt src/sara_brain/cortex/checkpoints/grammar_base_015000.pt \
  --sample 5

# score 10 dev sentences (sorted by perplexity, low = grammatical)
.venv/bin/python -m sara_brain.cortex.transformer.inference \
  --ckpt src/sara_brain/cortex/checkpoints/grammar_base_015000.pt \
  --score-dev 10
```

## Results

Grammar-LM, 125 M params, 20 000 steps, batch 32, seq 96 on six English
UD treebanks (36 404 train sentences, 5 922 dev):

| Configuration | Best dev perplexity | Notes |
|---|---|---|
| EWT only (12 544 sentences) | 3.01 (overfit past step 7 K) | First run, data-starved |
| All English UD (36 404 sentences) | **2.81** at step 15 K | The shipped checkpoint |
| Random baseline (uniform over 76-token vocab) | 76 | |

Router head, trained on 6 000 substrate-templated questions
(1 500 per class) from a Sara brain with 238 concepts:

| Class | Dev accuracy |
|---|---|
| `brain_did_you_mean` | 100.0% |
| `brain_value` | 93.3% |
| `brain_explore` | 85.9% |
| `brain_define` | 81.8% |
| **Overall** | **90.0%** |

## Architecture

```
     question (English)
            ↓
       spaCy parser
            ↓
    UD tag stream (DEPREL, UPOS pairs)
            ↓
   Grammar LM encoder (frozen, 125 M)
            ↓
     mean-pooled hidden state
            ↓
   Router head (≈600 K, trainable)
            ↓
       tool selection
            ↓
   Rule-based argument extractor
   (uses parse + substrate label index)
            ↓
       {tool, args}  — one shot, no retry loop
            ↓
        Brain executor
            ↓
        substrate result
```

The split is deliberate: the neural part learns *question shape*, the
rule part owns substrate-aware substring extraction (concept selection,
compound labels, normalization). This keeps the grammar LM purely
structural per the v024 thesis — substrate knowledge stays out of the
weights.

## Files

| File | What it is |
|---|---|
| [`vocab.py`](vocab.py) | 76-token structural vocabulary (UPOS + UD deps + slots) |
| [`model.py`](model.py) | `TransformerBlock` + `GrammarModel` with tiny / base / prod presets |
| [`ud.py`](ud.py) | UD treebank ingestion (downloads CoNLL-U, parses to (UPOS, DEPREL) streams) |
| [`synthetic.py`](synthetic.py) | `UDStreamDataset` and LM-batch generator |
| [`train.py`](train.py) | Grammar-LM training loop (bf16 AMP, cosine LR, dev-perplexity eval, resume) |
| [`inference.py`](inference.py) | Sample / score utility for the trained grammar LM |
| [`router_data.py`](router_data.py) | Substrate-driven labeler: `brain.db` → templated (question, tool) pairs |
| [`router_head.py`](router_head.py) | Frozen-encoder + classifier head |
| [`train_router.py`](train_router.py) | Router-head training loop |
| [`router_args.py`](router_args.py) | Rule-based argument extractor |
| [`router.py`](router.py) | `CortexRouter` end-to-end pipeline |
| [`ask.py`](ask.py) | Standalone CLI: question → cortex → substrate result |

## Limitations and next steps

- **English only.** Same UD-tag vocabulary works across all UD languages,
  so adding multilingual data is mostly a config change in `ud.py` —
  but the router head is currently trained on English question templates.
- **No synthesizer organ yet.** Sara's prose-writing half still goes to
  Claude Haiku via `sara_reader`. A v024 synthesizer organ (substrate
  edges → templated prose, same architecture, separate training) would
  remove that dependency. Use `--no-synthesis` to bypass it for now.
- **Define / explore confusion.** "What is X" and "Tell me about X" are
  genuinely interchangeable in casual usage — the head bottoms out at
  ~82% on `brain_define` because of this. Adding more disambiguating
  templates and possibly fine-tuning the encoder would push higher.
- **Scaling note.** A 300 M variant on the same 1.3 M-token corpus
  underperformed the 125 M base — the grammar task has a low entropy
  ceiling that 125 M already approaches. Bigger models need bigger data,
  and bigger data here means more languages (which dilutes the
  English-router signal). The structural-task / small-organ thesis is
  consistent with what we measured.
