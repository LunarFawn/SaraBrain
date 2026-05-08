# Plan — v052: local Ollama cortex with force-Sara strict mode (model-agnostic)

## Context

User wants a local-Ollama cortex paired with Sara that runs on
**any** Ollama-supported model — from mobile-device-grade tiny
models to highest-end PC models — without the cortex falling back
to training knowledge or accumulating conversation memory. The
target is a working tool that scales across device classes,
not a single-model setup.

The strict requirement: when querying Sara, the cortex must only
use Sara's substrate. No memory.md / per-project memory file
equivalents; no conversation history accumulation; no fallback to
the model's training-baked knowledge.

The cortex model is a parameter, not a fixed choice. Same code path
runs against `llama3.2:1b` on a phone-class device, `llama3.1:8b`
on a mid-range desktop (3070-class), or `llama3.3:70b` (with CPU
offload) on a high-end workstation. The user picks the model that
fits their hardware; the wrapper doesn't care.

Two structural layers were named in the prior discussion:

- **Layer A** — strict system prompt + structured `<substrate>` tags
  in the user message. Tells the model unambiguously that the only
  facts it may use are inside the tags.
- **Layer B** — single-turn stateless inference. Each call is
  independent: no conversation history, no memory file, no state
  accumulation. Defeats rev8 §5.4 within-session infection by
  construction.

The good news: most of the infrastructure already exists. The
`cli_stateless.py` CLI + `StatelessReader` class
([src/sara_reader/cli_stateless.py](src/sara_reader/cli_stateless.py),
[src/sara_reader/stateless_reader.py](src/sara_reader/stateless_reader.py))
already implement two-tier routing (Ollama picks tools) +
synthesis (Ollama generates answer) with no conversation memory.
Layer B is structurally true today. The Ollama provider
([src/sara_reader/providers/ollama.py:38](src/sara_reader/providers/ollama.py#L38))
takes a fresh `messages` list per call and supports `system_prompt`
(currently passed as empty string).

What's missing is Layer A's strictness. The current synthesis prompt
([stateless_reader.py:68](src/sara_reader/stateless_reader.py#L68))
says "use ONLY substrate facts" but doesn't:
- forbid training-derived inference explicitly
- wrap substrate in `<substrate>` tags
- enforce per-claim grounding ("every fact must trace to a triple")
- use the OllamaProvider's `system_prompt` channel to separate rules
  from data

The plan is a small targeted addition: a `--strict-sara` flag that
swaps in the tighter prompt and uses `system_prompt` to deliver the
rules separately from the user message.

## Recommended approach

### Slice 1 — add the strict synthesis prompt

**File modified:** [src/sara_reader/stateless_reader.py](src/sara_reader/stateless_reader.py)

Add a sibling to `_SYNTHESIS_PROMPT_TEMPLATE`:

```python
_STRICT_SARA_SYSTEM_PROMPT = """\
You are a substrate-bound research assistant. You have access to facts
ONLY through <substrate> tags in the user message. The contents of
those tags are the COMPLETE set of facts you may use.

Rules — these are absolute, no exceptions:
1. Every factual claim in your answer MUST trace to a triple inside
   <substrate>. If a triple does not state it, you do not state it.
2. If <substrate> does not contain the answer, respond exactly:
   "The substrate does not contain this information."
3. Do NOT use any knowledge from your training, even if you "know"
   the topic. Your training is unverifiable; the substrate is verified.
4. Do NOT make inferences that go beyond what the triples directly
   state. No "this likely means" or "in general."
5. Do NOT add hedging connectives ("additionally", "furthermore",
   "moreover") that smuggle in training-derived content.
6. When in doubt, say less. A short substrate-true answer is correct;
   a long answer with even one training-derived claim is wrong.
"""

_STRICT_SARA_USER_TEMPLATE = """\
<substrate>
{gathered}
</substrate>

Question: {question}
"""
```

Add a method on `StatelessReader.__init__` that takes a `strict_sara:
bool = False` flag. In the synthesis call site
([stateless_reader.py:345](src/sara_reader/stateless_reader.py#L345)),
branch on the flag:

```python
if self.strict_sara:
    user_msg = _STRICT_SARA_USER_TEMPLATE.format(
        gathered=gathered, question=question,
    )
    response = self.synthesizer.chat(
        messages=[{"role": "user", "content": user_msg}],
        system_prompt=_STRICT_SARA_SYSTEM_PROMPT,
    )
else:
    # existing path unchanged
    synthesis_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(...)
    response = self.synthesizer.chat(
        messages=[{"role": "user", "content": synthesis_prompt}],
    )
```

The OllamaProvider already supports the `system_prompt` channel
([providers/ollama.py:113](src/sara_reader/providers/ollama.py#L113))
— it prepends a `{"role": "system", ...}` entry to the messages
list before calling Ollama's `/api/chat`.

### Slice 2 — wire the `--strict-sara` flag in the CLI

**File modified:** [src/sara_reader/cli_stateless.py](src/sara_reader/cli_stateless.py)

Add an argparse flag:

```python
ap.add_argument(
    "--strict-sara",
    action="store_true",
    help=(
        "Force-Sara mode: pass substrate as <substrate> tags, deliver "
        "strict rules via system prompt, forbid training-derived "
        "inference. The cortex (per Pearl 2026a §7.3) gets ONLY what "
        "Sara provides. Use this for paper-aligned measurement and "
        "for any workflow where the cortex must not fall back to "
        "training knowledge."
    ),
)
```

Pass it to `StatelessReader(strict_sara=args.strict_sara, ...)`.

### Slice 3 — document the local-Ollama cortex invocation (model-agnostic)

**File modified:** [docs/user_guide_v049.md](docs/user_guide_v049.md)

Add a section: "Local Ollama cortex (no API)". The invocation is
the same whatever model you pick — just swap the tag:

```bash
# one-time: install ollama and pull whichever model fits your device
ollama pull <model>           # see device-class table below

# every-question (each invocation is structurally stateless;
# no memory.md, no per-project memory, no conversation history)
.venv/bin/python -m sara_reader.cli_stateless \
  --brain /tmp/sara_demo.db \
  --router-model <model> \
  --synthesis-model <model> \
  --strict-sara \
  "what is the molecular snare?"
```

**Device-class sizing guide** (Ollama supports all of these via the
same `/api/chat` endpoint; the wrapper code path is identical):

| device class | typical hardware | router model | synthesis model |
|---|---|---|---|
| mobile / very-low-RAM | phones, small tablets, ≤4GB RAM | `llama3.2:1b` (Q4) | `llama3.2:1b` (Q4) |
| low-end desktop / laptop | 8GB RAM, integrated GPU | `llama3.2:3b` | `llama3.2:3b` |
| mid-range desktop | RTX 3070 / 8GB VRAM, 32GB RAM | `llama3.2:3b` | `llama3.1:8b` |
| high-end desktop | RTX 4090 / 24GB VRAM | `llama3.1:8b` | `qwen2.5:32b` or `llama3.3:70b` (offload) |
| workstation / multi-GPU | A100 / H100 / dual-GPU | `llama3.1:8b` | `llama3.3:70b` (full GPU) |

Smaller router model + larger synthesis model is often the right
trade — routing is mostly tool selection (a 1-3B model handles it
fine), synthesis benefits from a stronger model. But identical
router and synthesis (small=small or large=large) also works.

Document that:
- The wrapper has no hardcoded model assumption. Any model Ollama
  serves works.
- Each invocation = fresh process = zero state carried over.
- The `--strict-sara` flag enforces Layer A (system prompt rules).
- Layer B (single-turn isolation) is structural — no flag needed,
  no way to accidentally turn it off.
- A parallel audit log inside `StatelessReader` records each
  `execute_tool` call with timestamp/args/result-bytes (Slice 4)
  — paths shorter than the MCP audit log because no IPC.

### Slice 4 — substrate-call audit log inside StatelessReader

**File modified:** [src/sara_reader/stateless_reader.py](src/sara_reader/stateless_reader.py)

Mirror the v050 MCP audit log so the `cli_stateless.py` path produces
the same paper-grade per-call diagnostic. Behind a `SARA_AUDIT_LOG`
env var:

```python
def _audit(tool_name: str, args: dict, result: str) -> None:
    path = os.environ.get("SARA_AUDIT_LOG", "")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(f"{ts}\t{tool_name}\t{json.dumps(args)}\t{len(result)}\n")
    except Exception:
        pass
```

Call it from each `execute_tool(self.brain, tool, args)` invocation
inside the routing loop. Same log format as the MCP audit log so a
single grep can compare across paths.

## Files

**Modified:**
- `src/sara_reader/stateless_reader.py` — strict prompt + audit log.
- `src/sara_reader/cli_stateless.py` — `--strict-sara` flag.
- `docs/user_guide_v049.md` — canonical Llama 3.1 8B invocation
  section.

**New:**
- `docs/plans/v052_local_llama_cortex.md` — companion plan doc
  capturing this slice for the historical record (so the plan exits
  Claude's working plan file and lives in repo with the rest).

**Not touched:**
- `src/sara_reader/providers/ollama.py` — already supports
  `system_prompt`, no change needed.
- `src/sara_brain/agent/ollama.py` — separate teaching-side
  Ollama integration, irrelevant to this slice.
- The MCP server — paper-aligned and unchanged; the cli_stateless
  path is a parallel client, not a replacement.

## Reused functions / utilities

- `StatelessReader.__init__`, `.ask()`, `.synthesize()` —
  [stateless_reader.py](src/sara_reader/stateless_reader.py).
  All keep working; we add a flag, not a new code path.
- `OllamaProvider.chat(messages=, system_prompt=)` —
  [providers/ollama.py:38](src/sara_reader/providers/ollama.py#L38).
  The `system_prompt` channel is currently unused by the synthesis
  call; this slice activates it for strict mode.
- `execute_tool(brain, tool_name, arguments)` —
  [tools.py:479](src/sara_reader/tools.py#L479). Called from the
  routing loop inside StatelessReader; we add audit-log
  instrumentation around the call.

## Order of operations

0. **Create new branch** `feature/v052-local-ollama-cortex` off the
   current `feature/grammar-cortex`. Keeps this work isolated until
   the slice proves out; merges back when ready.
1. **Save this plan + commit** as
   `docs/plans/v052_local_ollama_cortex.md` (mirror the working
   plan file into the repo on the new branch).
2. **Slice 1** (~50 lines): add `_STRICT_SARA_SYSTEM_PROMPT` +
   `_STRICT_SARA_USER_TEMPLATE` + `strict_sara` constructor flag in
   stateless_reader.py. Single commit.
3. **Slice 2** (~10 lines): add `--strict-sara` flag in
   cli_stateless.py, wire through. Single commit.
4. **Slice 4** (~30 lines): audit log behind `SARA_AUDIT_LOG` env
   var in stateless_reader.py. Single commit.
5. **Slice 3** (docs): user guide section + commit.
6. **Verification run** (Phase 5 below).
7. Push branch to `origin/feature/v052-local-ollama-cortex`.
8. Merge to `feature/grammar-cortex` after user-driven verification
   on at least one device-class model.

Total: ~100 lines of code, 4 commits, ~1 hour, on a new branch.

## Verification

End-to-end pass criteria:

1. **Layer A — strict prompt is delivered to Ollama.** Add a
   `--trace` invocation and confirm the Ollama `/api/chat` request
   payload includes a `{"role": "system", "content": "<strict
   rules>"}` entry alongside the user message. Manual check via
   `ollama serve` logs or a quick `tcpdump`-style proxy isn't
   needed; the trace flag's existing output already shows the
   prompt structure.

2. **Layer B — no state across calls.** Run the canonical
   invocation twice in succession asking the same question. Both
   responses should be byte-identical (modulo Ollama sampling
   variance) IF the synthesis temperature is 0; with default
   sampling they should be substantively the same. The key check:
   the second call should NOT reference "as I said before" or any
   continuation phrasing — that would be evidence of state leak.

3. **Substrate is queried.** With `SARA_AUDIT_LOG=/tmp/sara_audit.log`
   set, run the canonical invocation. The log should have ≥1 row
   per question — typically several for the routing loop's
   iterative `execute_tool` calls. `cat /tmp/sara_audit.log` after
   running confirms.

4. **Force-Sara worked.** Hand-grade the response: every factual
   claim should appear (verbatim or as paraphrase) in the audit
   log's recorded substrate output. No claim should be a training-
   derived guess. Substitute whichever model the user actually
   pulled:
   ```
   .venv/bin/python -m sara_reader.cli_stateless \
     --brain /tmp/sara_demo.db \
     --router-model <model> --synthesis-model <model> \
     --strict-sara --trace \
     "what is the molecular snare?"
   ```
   Expected: response that lists the substrate's specific framing
   (signal fold region, 5'3' static stem, mechanical forces) WITHOUT
   the training-derived "SNARE protein vesicle fusion" content.
   Smaller models will follow the strict prompt less reliably than
   larger ones — a 1B model will sometimes leak training content
   even with strict prompting; a 70B model rarely will. The strict
   prompt is the same regardless; compliance varies with model
   capability. Document this honestly.

5. **Empty-substrate honesty.** Ask a question whose answer isn't
   in the brain (e.g., "what is the capital of France?" against
   `/tmp/sara_demo.db`). Expected response under strict mode: "The
   substrate does not contain this information." NOT a training-
   derived "Paris."

6. **No memory.md leakage.** Confirm that running the same question
   from two different working directories produces the same answer.
   If they differ, some per-project memory file is being read.
   Expected: no difference. (This is structurally guaranteed by the
   bare-Python invocation but worth checking once.)

## Out of scope

- **Layer C (post-hoc grounding verification).** A second LLM pass
  that grades each claim against the substrate. Mentioned in the
  prior discussion as the strongest forcing layer; deferred to v053
  if Layer A + B prove insufficient.
- **Replacing the Claude/MCP path.** The MCP server stays
  unchanged. cli_stateless is a parallel client, not a replacement.
  Users can still use Claude Code via MCP; this just adds the
  local-Llama option.
- **Auto-routing improvements.** The router model (llama3.2:3b by
  default) sometimes mis-picks tools. Improving the router is
  separate work; this slice focuses on synthesis-side strictness.
- **Continue.dev / OpenHands / other agent frameworks.** Those
  bring their own memory systems (the very thing the user wants to
  bypass). Stay with the bare Python invocation.
- **Per-model evaluation / benchmarking.** Comparing how different
  Ollama models comply with the strict prompt is its own slice.
  This plan ships the model-agnostic wrapper; cross-model
  evaluation builds on it.

## What this gets us

1. **One working tool that scales across device classes.** Mobile
   to workstation, the same code path. Users pick the model that
   fits their hardware; the wrapper is indifferent.

2. **Local-only, no API dependency.** Privacy, cost, and
   reproducibility benefits — no rate limits, no upstream model
   changes, no network round-trip.

3. **Force-Sara as a first-class mode.** When you query Sara via
   the wrapper, you get answers grounded in the substrate, not
   the model's training. This is the workflow the user asked for:
   "freeze the LLM, teach Sara cheaply, query through the
   wrapper."

4. **Ollama version compatibility built in.** The wrapper uses
   bare HTTP via `urllib` (stdlib) against the stable
   `/api/chat` endpoint. No `ollama` PyPI dependency. Works on
   any Ollama version from 0.1.x onward.
