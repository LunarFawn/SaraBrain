# Local Sara cortex — setup and usage

A working tool for running Sara Brain with a frozen local LLM cortex,
no API keys, no conversation memory, no training-data fallback.

You teach Sara cheaply (one command per fact). You query through a
chat REPL whose cortex is whichever Ollama model your hardware fits.
The cortex stays stateless across questions; Sara is the persistent
memory.

This document is the practical setup + usage reference. The
architectural reasoning is in
[v050_two_layer_realignment.md](v050_two_layer_realignment.md).

## What this gives you

A chat REPL that:

```
                                you (typed input)
                                       │
                                       ▼
                          input parsing  +  routing
                          (your cortex transformer
                           — grammar LM + router head,
                           or an Ollama model)
                                       │
                                       ▼
                            substrate retrieval
                          (Sara — SQLite path graph
                           on disk, your facts)
                                       │
                                       ▼
                              prose synthesis
                       (Ollama-served local LLM,
                        substrate-only via strict-Sara)
                                       │
                                       ▼
                                  answer
```

Every claim in the answer traces to a substrate edge. Each question
is structurally isolated from every other (no conversation history).
Adding a fact via `/teach` makes it immediately available to the
next question; no model retraining.

## Hardware sizing

The wrapper is model-agnostic; pick the synth model that fits your
GPU VRAM. Identical code path across device classes.

| device class       | hardware                  | router            | synthesis                                 |
|--------------------|---------------------------|-------------------|-------------------------------------------|
| mobile             | phones, ≤4 GB RAM         | `llama3.2:1b`     | `llama3.2:1b` (compliance is fragile)     |
| low-end laptop     | 8 GB RAM, integrated GPU  | `llama3.2:3b`     | `llama3.2:3b`                             |
| mid-range desktop  | RTX 3070 / 8 GB VRAM      | cortex-router     | `llama3.2:3b` (recommended)               |
| high-end desktop   | RTX 4090 / 24 GB VRAM     | cortex-router     | `qwen2.5:7b-instruct-q4_K_M` or `llama3.1:8b` |
| workstation        | A100 / H100 / dual-GPU    | cortex-router     | `llama3.3:70b`                            |

Below 3B as the synth model, the strict-Sara prompt's compliance
breaks (the model receives substrate context but ignores it). 3B is
the practical floor.

## Prerequisites

You need:

```
Linux or WSL2 (tested), macOS likely works
Python 3.11+
~10 GB free disk for one synth model
NVIDIA GPU recommended (CPU works but is slow)
```

Plus the SaraBrain repo cloned and its `.venv` set up — the existing
project virtualenv with sara_brain and sara_reader installed.

## Install Ollama

If you have an old snap-installed Ollama, remove it first (snap
confinement blocks GPU access):

```
sudo snap remove --purge ollama
```

Install the official Linux build (auto-detects CUDA, no snap):

```
curl -fsSL https://ollama.com/install.sh | sh
```

Verify the install path:

```
which ollama
```

Expected output: `/usr/local/bin/ollama`. If you see `/snap/bin/...`,
run `hash -r` to clear bash's command cache and try again.

## Pull a synthesis model

Pick one that matches your hardware tier from the table above.
Recommended starting point for an RTX 3070 / 8 GB VRAM:

```
ollama pull llama3.2:3b
```

For richer answers if you have headroom (4090 / 24 GB):

```
ollama pull qwen2.5:7b-instruct-q4_K_M
```

For paper-grade fidelity testing (8B Llama):

```
ollama pull llama3.1:8b
```

Verify it loads on GPU after first call:

```
ollama run llama3.2:3b "hi" &
sleep 3
ollama ps
```

The PROCESSOR column should read `100% GPU`. If it shows `100% CPU`
or partial offload, the GPU isn't being used — see Troubleshooting.

## Build a brain

You need a Sara brain.db file with substrate content. Either use an
existing one (e.g. the RNA aptamer brain at `/tmp/sara_demo.db`) or
build from teaching scripts:

```
.venv/bin/python papers/aptamer_rev1/teach_full_paper.py /tmp/your_brain.db
```

Or start with an empty brain and teach interactively (next section).

## Run the chat REPL

The canonical command for a 3070-class machine, fastest path,
substrate-grounded:

```
SARA_AUDIT_LOG=/tmp/sara_audit.log \
.venv/bin/python -m sara_reader.cli_stateless_chat \
  --brain /tmp/sara_demo.db \
  --cortex-router \
  --synthesis-model llama3.2:3b \
  --strict-sara \
  --explore-first
```

For higher-quality answers (slower, Qwen 7B as synth):

```
SARA_AUDIT_LOG=/tmp/sara_audit.log \
.venv/bin/python -m sara_reader.cli_stateless_chat \
  --brain /tmp/sara_demo.db \
  --router-model qwen2.5:7b-instruct-q4_K_M \
  --synthesis-model qwen2.5:7b-instruct-q4_K_M \
  --max-routing-steps 2 \
  --strict-sara \
  --explore-first
```

Note: the 7B + cortex-router combo doesn't fit cleanly in 8 GB
VRAM; use single-model 7B on a 3070, or pair cortex-router with
7B+ only on 12 GB+ VRAM cards.

## Inside the REPL

Type a question, get an answer. Each question is a fresh routing
loop — no conversation history.

```
> what is the molecular snare?
```

Slash commands:

```
/help                         show all commands
/teach SUBJ REL OBJ           teach a flat triple
/teach STATEMENT              teach a parsed natural-language fact
/refute STATEMENT             negate a fact
/teach-event SUBJ ACTION [object=O] [location=L] [from=ISO] [to=ISO] [modifier=M]
                              create a v047 reified event node
/where-is SUBJECT [at=ISO]    point-in-time location query
/list-events SUBJECT          chronological event list for a subject
/find-function NAME           v049 function info (signature, returns, params, calls, docstring)
/callers NAME                 who calls NAME
/callees NAME                 what NAME calls
/returns-type TYPE            functions returning TYPE
/takes-type TYPE              functions taking TYPE as parameter
/trace                        toggle trace output (shows tool calls)
/audit                        print SARA_AUDIT_LOG path
/quit                         exit
```

Up-arrow recalls history. Ctrl-D and Ctrl-C exit cleanly. History
persists across sessions in `~/.sara_chat_history`.

## Teach Sara new facts

Direct triple teaching (3-token form):

```
> /teach fulcrum is_a support point
> /teach fulcrum part_of lever
> /teach kdon stands_for kd of the on state
```

Multi-word object after the first two tokens:

```
> /teach helix manufactured_by alpha corporation
```

Reified event with optional bindings:

```
> /teach-event alice walked_to cafe location=downtown from=2026-05-08T14:00 to=2026-05-08T16:00
```

The next question you ask immediately sees the new triples — no
model reload, no rebuild.

## Verification

Confirm Sara is being queried (not just the model fabricating):

```
cat /tmp/sara_audit.log
```

Each tool call appends one row in TSV format:

```
ISO_TIMESTAMP    tool_name    args_json    result_bytes
```

A row per question with non-zero `result_bytes` means substrate was
queried and returned content. If you see zero rows after asking a
question, the cortex didn't query Sara at all — that's a setup
issue.

## Configuration knobs

`--cortex-router` uses your trained classifier head for tool
selection (~10 ms per call, deterministic, doesn't paraphrase
labels). Without it, the configured `--router-model` (an Ollama
LLM) does routing — slower, sometimes substitutes labels for
training-baked terms (rev8 §5.3).

`--strict-sara` activates the force-Sara synthesis prompt: substrate
is wrapped in `<substrate>` tags, strict rules in the system_prompt
channel, training-derived inference forbidden. Compliance scales
with model size — 3B is the practical floor.

`--explore-first` always prepends a `brain_explore depth=3` call
with a heuristic-extracted topic from the question. Captures the
associative neighborhood per Pearl 2026a §2.4 even when downstream
routing picks narrower tools. Recommended pairing with `--strict-sara`.

`--max-routing-steps N` caps the routing loop. Lower = faster but
the router has fewer chances to refine. Default 6, recommended 2-3
for `--cortex-router` workflows.

`--no-synthesis` skips the LLM synthesis step entirely. The REPL
prints raw substrate triples instead of prose. Sub-second per
question; the cortex is bypassed and you read the substrate
directly.

`SARA_AUDIT_LOG=/path/to/log` (env var) appends one TSV row per
substrate tool call. Set whenever you want to verify what was
queried.

`OLLAMA_KEEP_ALIVE=24h` (env var) keeps Ollama models resident
between calls. Without this, Ollama unloads after 5 minutes idle
and the next call pays a model-load cost (~3-10 seconds).

## Switching brains

The `--brain` flag accepts any Sara brain.db path. You can run the
REPL against multiple brains by changing the flag:

```
SARA_AUDIT_LOG=/tmp/sara_audit.log \
.venv/bin/python -m sara_reader.cli_stateless_chat \
  --brain /path/to/your_other_brain.db \
  --cortex-router \
  --synthesis-model llama3.2:3b \
  --strict-sara \
  --explore-first
```

Each brain is its own SQLite file. Teaching one doesn't affect
others.

## Save a launcher script

For daily use, save the canonical command as `~/bin/sara-chat`:

```
#!/usr/bin/env bash
SARA_AUDIT_LOG="${SARA_AUDIT_LOG:-/tmp/sara_audit.log}" \
exec /home/YOUR_USERNAME/repo/SaraBrain/.venv/bin/python \
  -m sara_reader.cli_stateless_chat \
  --brain "${SARA_BRAIN:-/tmp/sara_demo.db}" \
  --cortex-router \
  --synthesis-model llama3.2:3b \
  --strict-sara \
  --explore-first
```

Make it executable:

```
chmod +x ~/bin/sara-chat
```

Then daily use:

```
sara-chat
```

Or against a different brain:

```
SARA_BRAIN=/tmp/your_other_brain.db sara-chat
```

## Troubleshooting

`bash: /snap/bin/ollama: No such file or directory`

Bash cached the old snap path. Run:

```
hash -r
```

`ollama ps` shows `100% CPU` instead of `100% GPU`

The model isn't using GPU. Most common causes on WSL2 / Linux:

```
nvidia-smi
```

If this errors, install NVIDIA's WSL CUDA driver from
https://developer.nvidia.com/cuda/wsl (Windows-side install).

If `nvidia-smi` works but Ollama still shows CPU, check whether
Ollama was installed via snap:

```
ps aux | grep ollama
```

If you see `/snap/ollama/...` paths, snap is sandboxing Ollama
away from `/dev/nvidia*`. Remove and reinstall:

```
sudo snap remove --purge ollama
curl -fsSL https://ollama.com/install.sh | sh
hash -r
```

`ollama ps` shows partial offload like `7%/93% CPU/GPU`

Model is too big for VRAM. Either use a smaller model:

```
ollama pull llama3.2:3b
```

Or unload other models to free VRAM:

```
ollama stop other-model-name
```

`The substrate does not contain this information` (when you know
the substrate has it)

The synth model decided substrate was insufficient and refused.
3B models do this on partial substrate. Either teach more facts
about the topic (`/teach SUBJECT is_a ...`), or fall back to a
larger synth model (7B Qwen handles partial substrate more
gracefully):

```
.venv/bin/python -m sara_reader.cli_stateless_chat \
  --brain /tmp/sara_demo.db \
  --router-model qwen2.5:7b-instruct-q4_K_M \
  --synthesis-model qwen2.5:7b-instruct-q4_K_M \
  --max-routing-steps 2 \
  --strict-sara \
  --explore-first
```

Confirm what was actually queried:

```
tail -10 /tmp/sara_audit.log
```

Up-arrow shows `^[[A` instead of recalling history

Your terminal isn't getting readline support. Already fixed in
the latest cli_stateless_chat (readline is imported automatically),
but if you're on a non-standard terminal it can still happen.
Try a different terminal or a fresh shell.

## Reference

For architectural background:

```
docs/v050_two_layer_realignment.md       — two-layer cortex/hippocampus rationale
docs/plans/v052_local_ollama_cortex.md   — this slice's plan + design notes
papers/sara_as_instrument_paper_rev8.md  — instrument paper (rev8)
papers/zenodo_preprint_rev3_2026-04-12.md — foundational path-of-thought paper
```

For the production paper-aligned MCP server (alternative to
cli_stateless_chat — for use with Claude Code / Claude Desktop):

```
docs/v047_reified_events_and_narrative_corpus.md
docs/v049_code_knowledge_substrate.md
src/sara_brain/mcp_server.py
```
