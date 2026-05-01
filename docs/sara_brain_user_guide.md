# Sara Brain User Guide — From Install to Experiment

Sara Brain is a knowledge substrate for measuring how faithfully a language model retrieves what you taught it versus confabulating from its training weights. This guide walks you through installation, teaching Sara, asking questions, and running controlled experiments that produce publishable data.

---

## What Sara Brain Is

When you teach a language model something in a conversation, that knowledge lives in the session context. The moment the session ends, it is gone. The model's weights — what it actually learned during pre-training — are unchanged. The next conversation starts with the same biases and the same gaps.

Sara Brain changes this. When you teach Sara a fact, Sara stores it permanently in a SQLite graph. When you ask a question, a routing model searches Sara's graph and injects what it finds into the synthesis model's context. The synthesis model then answers from what Sara retrieved, not from its training weights.

This creates a measurable difference: the same question, asked in a fresh session with Sara versus without Sara, should produce different answers. The gap between those two answers is the measurement.

---

## Install

```bash
pip install sara-brain
```

This gives you the brain core and all CLI tools. If you want to use the Anthropic provider (Claude models) for synthesis:

```bash
pip install sara-brain[reader]
```

Requires Python 3.11+.

---

## Step 1 — Create a Brain

```bash
sara-new --out my_brain.db
```

This creates a fresh empty Sara brain at the path you specify. The brain is a SQLite file. You can create as many as you need — one per project, one per experiment, or one per topic domain.

---

## Step 2 — Teach Sara

```bash
sara-teach \
  --subject "mitochondria" \
  --relation "is_a" \
  --object "organelle" \
  --brain my_brain.db

sara-teach \
  --subject "mitochondria" \
  --relation "produces" \
  --object "ATP" \
  --brain my_brain.db
```

Each call stores a triple: (subject, relation, object). The labels are stored verbatim — Sara does not parse, embed, or interpret them. "Mitochondria" in Sara is the string "mitochondria" connected to "is_a" and "organelle" by directed edges.

You can teach as many triples as you want. Multi-word compound terms work exactly the same way:

```bash
sara-teach \
  --subject "sodium potassium pump" \
  --relation "regulates" \
  --object "membrane potential" \
  --brain my_brain.db
```

---

## Step 3 — Ask a Question

```bash
sara-ask-stateless "what does mitochondria produce?" \
  --brain my_brain.db \
  --router-model llama3.2:3b \
  --synthesis-provider ollama \
  --synthesis-model llama3.2:3b
```

`sara-ask-stateless` is the command that implements the measurement protocol. It works in two phases:

**Phase 1 — Routing.** A small local model (the router) receives your question and a list of tools that search Sara's graph. The router decides which brain tools to call and in what order. It calls `brain_explore`, `brain_trace`, `brain_why`, and similar tools until it has gathered enough information or reaches the step limit.

**Phase 2 — Synthesis.** The gathered results from Sara's graph are passed to the synthesis model along with your question. The synthesis model writes the final answer using what Sara retrieved.

The synthesis model never sees the brain directly — it only sees the text that the router extracted. If Sara has no relevant triples, the router finds nothing, and the synthesis model has nothing to work with.

---

## Step 4 — Run a Controlled Experiment

Single observations are anecdotes. To produce data, you need to ask the same question many times and record the results.

`sara-experiment` does this automatically. It runs two sessions in parallel:

**Session B** — Sara is present. The router searches the brain, injects the retrieved triples, and the synthesis model answers.

**Session C** — Sara is not present. The synthesis model receives the same question directly, with no brain search and no retrieved context. It can only answer from its training weights.

```bash
sara-experiment \
  --brain my_brain.db \
  --manifest my_brain.db.manifest.json \
  --question "what does mitochondria produce?" \
  --trials 50 \
  --synthesis-provider ollama \
  --synthesis-model llama3.2:3b \
  --out results.json
```

Console output:

```
Session B (substrate present):   43 / 50 hits  (86.0%)
Session C (bare model):           2 / 50 hits   (4.0%)
Delta:                          +82.0 percentage points
```

The JSON file contains every trial's raw answer, so you can read what the model actually said in each case.

---

## Using a Synthetic Substrate for Clean Measurement

For the cleanest possible experiment, use a substrate whose contents cannot possibly be in any model's training data. `sara-synth` generates one:

```bash
sara-synth --out synth.db --seed 42 --concepts 30 --triples 80
```

This creates a brain populated with randomly-generated nonsense words — "zilkrap", "bortle", "fenduv" — connected by real relation labels. These words did not exist before this command ran. No language model has ever been trained on them.

```bash
sara-experiment \
  --brain synth.db \
  --manifest synth.db.manifest.json \
  --question "what is zilkrap?" \
  --trials 100 \
  --out synth_results.json
```

With a synthetic substrate, the scoring is unambiguous:

- A hit in **Session B** means the router found "zilkrap" in Sara's graph and the synthesis model repeated it in the answer. The model could not have known this word from training — it came entirely from Sara.
- A hit in **Session C** is not possible. The model has no source for a word that was invented at runtime. A Session C answer will say "I don't know" or confabulate something unrelated.

The `--seed` flag makes the substrate reproducible. Pass the same seed to regenerate the identical concepts and triples, which lets you repeat the experiment on a fresh brain and get comparable results.

---

## The A/B/C Protocol

The full measurement protocol has three sessions:

| Session | Command | What it measures |
|---------|---------|-----------------|
| A — Teach | `sara-teach` | Load knowledge into Sara |
| B — Test | `sara-ask-stateless` (fresh session) | Retrieval with substrate |
| C — Control | `sara-experiment` bare model path | Training weights only |

Session A is just teaching — it does not produce a measurement. Sessions B and C are the measurement. Run them in the same experiment call and compare the delta.

The critical requirement: Session B must be a **fresh session**. If you teach Sara in one conversation and then ask in the same conversation, the session context itself contains the teaching history. The model may answer correctly not because Sara retrieved anything, but because the teaching is sitting in the context window. `sara-ask-stateless` enforces fresh sessions by design — each call to the router is a new stateless request with no memory of prior calls.

---

## Running Overnight

50 to 100 trials per session is enough for a publishable result. On an M2 Mac with 8 GB RAM using local Llama models, this takes roughly 2–4 hours depending on the model and step count. You can start an experiment before bed and read the results in the morning:

```bash
sara-synth --out /tmp/overnight.db --seed 1234 --concepts 30 --triples 80

sara-experiment \
  --brain /tmp/overnight.db \
  --manifest /tmp/overnight.db.manifest.json \
  --question "what is zilkrap?" \
  --trials 100 \
  --router-model llama3.2:3b \
  --synthesis-provider ollama \
  --synthesis-model llama3.2:3b \
  --out /tmp/overnight_results.json
```

The results file will be ready when you wake up.

---

## MCP Server (Teaching from Claude)

If you use Claude Desktop or any MCP-compatible client, you can teach Sara interactively without writing CLI commands:

```bash
sara-mcp --brain my_brain.db
```

Configure your MCP client to point to this server. You can then have a conversation with Claude where you say "teach Sara that X is Y" and it will call the `brain_teach_triple` tool directly. This is useful for building up a substrate from a conversation or from reading a document.

---

## All CLI Commands

| Command | What it does |
|---------|-------------|
| `sara-new --out path.db` | Create a fresh empty brain |
| `sara-synth --out path.db --seed N` | Generate a synthetic training-orthogonal substrate |
| `sara-teach --subject X --relation R --object Y --brain path.db` | Teach one triple |
| `sara-ask-stateless "question" --brain path.db` | Ask a question (stateless, fresh session) |
| `sara-experiment --brain path.db --manifest path.manifest.json --question "..." --trials N` | Run paired B/C trials |
| `sara-mcp --brain path.db` | Start MCP server for interactive teaching |
| `sara` | Launch the interactive REPL |

---

## Paper

The measurement instrument design and the session-context infection findings are documented in:

**Sara Brain: A Measurement Instrument for Transformer Behavior**
[https://doi.org/10.5281/zenodo.19436522](https://doi.org/10.5281/zenodo.19436522)
