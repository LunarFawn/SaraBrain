# Protocol — Day-to-Day Workflow and Measurement Method

Practical instructions for (a) running a clean Sara-backed LLM test, (b) loading a new substrate into the harness, and (c) grading the results.

See `PURPOSE.md` for the background on why this is set up the way it is.

## Part A — Running a test

### Prerequisites

- `sara_brain/` checked out as a sibling to this directory at `/Users/grizzlyengineer/repo/sara_brain/`
- `sara_brain/.venv/` set up with all dependencies installed (the venv the MCP server runs under)
- At least one brain substrate loaded into `brains/` — run `./load_brain.sh <name>` if not
- `loaded.db` symlink pointing at a valid brain (the loader sets this)

### Quick-start (default brain, aptamer)

```bash
cd /Users/grizzlyengineer/repo/sara_test
./load_brain.sh aptamer_exec
claude
```

When Claude Code starts, it will prompt to approve the `sara-brain` MCP server. Approve it. Ask questions. When finished, close the session.

### The three-session measurement protocol

A rigorous measurement requires three distinct Claude Code sessions:

**Session A — Teaching (happens in `sara_brain/`, not here)**
The author writes triples and runs `teach_triple` calls against a brain in `sara_brain/`. The session's context window accumulates source material, authored triples, and discussion of every term. **No evaluation questions are asked in Session A**; the measurement would be contaminated by in-context recall.

**Session B — Test with Sara (happens here, in `sara_test/`)**
After the teaching session ends, load the brain here (`./load_brain.sh <name>`), open a fresh Claude Code session at this directory, approve the MCP server, and ask the evaluation questions. The session has:
- Filesystem access limited to `sara_test/` (no paper source visible)
- Empty per-project auto-memory (no aptamer-related memories auto-load)
- MCP access to Sara's graph via the loaded brain
- No prior conversation context about the substrate

**Session C — Control without Sara (happens here or in a neutral directory)**
Open *another* fresh Claude Code session at this directory, but **decline** the MCP server prompt (or rename `.mcp.json` → `.mcp.json.off` before starting). Ask the same questions. This session's answers rely solely on the LLM's training.

### What the measurement actually measures

| Session | Substrate access | LLM's answer sources |
|---|---|---|
| B (with MCP) | Sara + training + (minor) interpretation bias | Mixed — grade to separate |
| C (without MCP) | Training only | Training alone |
| **B − C** | — | **What Sara contributed** |

The *difference* between B and C answers on the same question is Sara's contribution to the LLM's capability. If B answers correctly about paper-coined terms and C says "I don't know" or invents plausibly-wrong content, Sara is doing its job. If B and C give similar answers, Sara's contribution is zero on that question (either the LLM already knew it, or Sara didn't help).

### Choosing a reader model

Per the empirical findings in `sara_brain/papers/training_corrupts_reading_draft_v1.md`:

- **Haiku (or equivalent small model)**: more faithful to the substrate; shorter, less synthesized output; fewer invented causal connectors. Recommended default.
- **Opus (or equivalent large model)**: more synthesis, more invented connective tissue ("propagates," "drives," etc.) pattern-matched from training narratives. Use when you want to stress-test a substrate or when multi-hop synthesis is specifically required.
- **Tiny models (3B and below)**: tested via Ollama in `sara_brain/papers/aptamer_rev1/test_small_models.py`. 3B can render faithfully but with limited synthesis; 1B under-delivers (echoes triples, can't compose).

Switch model mid-session with `/model`.

### Asking questions faithfully

To maximize the signal in B − C:

1. **Use exact substrate vocabulary.** Type "molecular snare" not "snare" or "molecular SNARE." The SNARE case in `sara_brain/papers/model_infections_draft_v1.md` Case 2.1 shows how one capitalization shift can route the model toward its training's famous-acronym bias (SNARE protein family) and away from Sara.
2. **Don't paste source material into the chat.** That contaminates the session-context layer immediately and B's answers become indistinguishable from recall.
3. **Ask the same question multiple times if you suspect drift.** Divergent answers in the same session signal session-context accumulation.
4. **Use `/model` to swap readers on the same brain.** Running Haiku and Opus on the same question against the same brain is the cleanest demonstration of the faithfulness-vs-embellishment tradeoff.

### Logging results

There is no automatic logger in this harness. Copy the LLM's answer out of the Claude Code session into a timestamped file:

```
sara_test/results/2026-04-23_haiku_aptamer_exec_Q1.md
sara_test/results/2026-04-23_opus_aptamer_exec_Q1.md
sara_test/results/2026-04-23_haiku_noMCP_Q1.md
```

`results/` is gitignored by convention (you may want to commit some and not others; grade first, then pin).

## Part B — Loading and managing substrates

### Loading an existing brain

```bash
./load_brain.sh aptamer_exec
```

- Source: `sara_brain/aptamer_exec.db`
- Destination: `sara_test/brains/aptamer_exec.db` (committed to git)
- Symlink: `sara_test/loaded.db` → `sara_test/brains/aptamer_exec.db`

The loader uses SQLite's `.backup` command (not `cp`) so it produces a consistent snapshot even if `sara_brain/` has an open connection to the source DB with uncommitted WAL data.

Running `./load_brain.sh` with no argument lists available brains in `sara_brain/`.

### Adding a new substrate

After teaching a new brain in `sara_brain/` (e.g., a paper on topic X taught via `teach_triple`):

1. Confirm the brain file exists: `ls sara_brain/<name>.db`
2. Load it into the test harness: `cd sara_test && ./load_brain.sh <name>`
3. Commit the new brain: `git add brains/<name>.db && git commit -m "Add <name> substrate"`
4. Restart any open Claude Code session in `sara_test/` for the new `loaded.db` to take effect.

### Rotating between substrates

Just call `load_brain.sh` with a different name; it overwrites `loaded.db`. The brains themselves are additive — each stays in `brains/` once loaded. You can switch freely between committed substrates.

### Removing a substrate

```bash
git rm brains/<name>.db
git commit -m "Remove <name> substrate"
```

If you also want to purge it from the working tree cache: `rm brains/<name>.db`. The loader will fail on the next load attempt for that name and point you back at `sara_brain/`.

### Updating a substrate to a newer version

If the substrate needs to be re-taught (e.g., triples corrected, more content added):

1. In `sara_brain/`, delete and rebuild the source DB (e.g., `rm aptamer_exec.db && python papers/aptamer_rev1/teach_exec_summary.py`)
2. In `sara_test/`, reload: `./load_brain.sh aptamer_exec`
3. Commit the updated brain: `git add brains/aptamer_exec.db && git commit -m "Re-teach aptamer substrate — <what changed>"`

The git history preserves the old versions; you can check out an earlier commit to reproduce an earlier measurement.

## Part C — Grading results

### Grading rubric

For each LLM response in Session B, grade every factual claim against the loaded brain's triples:

- **Retrieved** — the claim is a direct rendering of one or more triples.
- **Retrieved-with-paraphrase** — the claim renders a triple in different words but preserves the semantic content.
- **Inferred** — the claim is an extrapolation from retrieved triples that is not itself a triple but is supported by the graph's structure.
- **Invented** — the claim contains entities, relations, or connective verbs with no basis in any triple.
- **Boundary-acknowledged** — the response explicitly states Sara does not hold some requested detail. This is a positive signal for faithfulness.

The *proportion* of each category across a response is the faithfulness profile. A faithful reader produces mostly Retrieved + Retrieved-with-paraphrase + Boundary-acknowledged, with minimal Inferred and zero Invented.

### Tools for grading

Querying the loaded brain directly (outside the Claude Code session, from a terminal):

```bash
cd /Users/grizzlyengineer/repo/sara_brain
source .venv/bin/activate
python -c "
from sara_brain.core.brain import Brain
b = Brain('/Users/grizzlyengineer/repo/sara_test/brains/aptamer_exec.db')
print(b.why('molecular snare'))
print(b.trace('molecular snare'))
"
```

Or via the MCP server's tools directly (same call pattern Claude uses).

### Comparing B and C

For each question, compute:

- **B was correct, C was correct** — LLM already knew, Sara didn't contribute.
- **B was correct, C was wrong or admitted ignorance** — Sara contributed on this question. ✓
- **B was wrong, C was wrong** — Sara has no signal on this question (graph doesn't cover it, or retrieval failed).
- **B was wrong, C was correct** — rare; suggests Sara's content misled the LLM away from a training-based correct answer. Worth investigating.

The headline result for a substrate is the count of (B correct, C wrong-or-abstained) questions out of the total.

## Part D — Troubleshooting

### The MCP server fails to start

Check `sara_brain/.venv/` exists and has the MCP package installed:
```bash
cd /Users/grizzlyengineer/repo/sara_brain && ls .venv/lib/python*/site-packages/ | grep -i mcp
```
If missing, reinstall with the `mcp` extra: `pip install -e .[mcp]` from `sara_brain/`.

### `loaded.db` is broken

```bash
rm -f loaded.db loaded.db.reverse*
./load_brain.sh <name>
```

### Claude Code doesn't prompt for the MCP server

Confirm `.mcp.json` is at the root of this directory and is valid JSON:
```bash
cat /Users/grizzlyengineer/repo/sara_test/.mcp.json | python -m json.tool
```

### A question gets the same answer in Session B and Session C

Two possibilities:
1. The LLM's training already covers this question (substrate doesn't add anything).
2. The LLM isn't actually using Sara — filesystem contamination or the MCP call is being skipped. Check what the LLM is doing: ask it `"show me the last tool call you made and its output."` If it didn't call `brain_query` or equivalent, the MCP path isn't being exercised.

### Session B's answer references paper content not in Sara

Likely culprits, in order:
1. The LLM read a file on disk. Confirm with `"list the files in the current directory"` — there should be no paper source here.
2. The LLM pulled the content from training (substrate-orthogonality broken — the paper has been in training corpora since teaching).
3. The LLM synthesized the content from other triples in Sara. Grade with care; the synthesis may be legitimate inference.

### You want to test multiple substrates in sequence

Load brain A, run Session B and C, log results. Load brain B (`./load_brain.sh other_name`), open NEW Claude Code sessions (the old sessions kept the old brain's context — amputate), run tests. Repeat.

## Part E — When to add a new substrate

A good candidate substrate:

- Is a specific document or body of knowledge you want to test LLM-via-Sara readability of.
- Has content that is finite (enumerable), novel (not widely in training), and structurable as triples.
- Can be taught in under an hour via `teach_triple` calls.
- Has natural evaluation questions that require multi-hop reasoning across its triples.

Bad candidates:

- "General biology" (too broad, too much is in training).
- Content that requires embedding understanding (images, math derivations not expressible as triples).
- Substrates so small (≤ ~30 triples) that every question reduces to single-fact lookup — the instrument can't distinguish retrieval from confabulation at that scale.

## Part F.5 — Harness hygiene (on-disk contamination)

A fresh Claude Code session opened in `sara_test/` has `Read`/`Grep`/`Glob` tools scoped to that directory. Every file sitting there is potentially visible to the model. A contaminant that would not crash the test can still quietly distort it.

**What counts as on-disk contamination:**

- **Evaluation questions** — if the question file is in the harness directory, the model can grep all questions at once and plan for them, or see "Expected:" hints that leak the grading rubric.
- **Prior test results** — if earlier responses are on disk, the model can align its new answer to prior framings (or deliberately contrast, but either way it's no longer independent).
- **Protocol documents** — if the measurement methodology, the grading rubric, or descriptions of what "faithful retrieval" looks like are on disk, the model can shape its output to match them. This is framing contamination.
- **Source documents for the substrate** — the original paper or notes about it. The canonical contamination vector per Case 2.4.

**The hygiene rule:** the harness directory contains ONLY operational artifacts — the MCP config, the brain snapshots it serves, the loader/registration scripts, and a short README pointing elsewhere. Everything else — documentation, questions, past results, draft papers — lives in the research repo (`sara_brain/`), which is a sibling directory Claude cannot see from inside the harness.

**Expected sara_test contents:**

| File / dir | Purpose | Why it's safe to keep here |
|---|---|---|
| `.mcp.json` | Register sara-brain MCP server | Just config paths, no content |
| `load_brain.sh` | Swap the loaded substrate | Shell script, no retrieval content |
| `register_mcp.sh` | Initial MCP registration | Shell script, no retrieval content |
| `run_ollama_via_mcp.py` | Ollama orchestrator | Code, not substrate-content |
| `brains/*.db` | SQLite substrates | Binary — not greppable by Claude Code's text tools |
| `loaded.db` | Symlink to active brain | Symlink target, not content |
| `README.md` | Minimal pointer | Points at docs in sara_brain, doesn't reproduce them |

**Where content goes instead:**

| Content class | Home |
|---|---|
| Full protocol / purpose docs | `sara_brain/harness/` |
| Per-substrate evaluation questions | `sara_brain/papers/<topic>/EVAL_QUESTIONS.md` |
| Session B/C responses and grading | `sara_brain/papers/<topic>/results/` |
| Paper drafts, source documents, notes | `sara_brain/papers/<topic>/` |

**How this was learned (2026-04-24).** Initial harness setup placed `PURPOSE.md`, `PROTOCOL.md`, `EVAL_QUESTIONS.md`, and a `results/` directory inside `sara_test/`. Jennifer caught the contamination while reviewing where a Session C response file had been written (*"should this be saved in sara_test?"*). Every doc that was in the harness had to be moved; the Ollama orchestrator's default output path had to be redirected to `sara_brain/papers/aptamer_rev1/results/`. The lesson: designing an isolated harness means *actively resisting the urge to put anything useful in it*. Helpful-to-the-user content is contaminating-to-the-measurement content. Put nothing here that a future Claude session shouldn't be allowed to read.

---

## Part F — The harness evolves

As contamination vectors are discovered, the harness is updated. Today (2026-04-23) it defends against:

- Filesystem contamination (the current directory has no readable paper content)
- Auto-memory contamination (per-project memory path isolated)
- Substrate mutation during testing (brains are snapshots, not live references)
- Dev-write leakage from `sara_brain/` into the test substrate

Not yet defended against but known issues:

- Training-weight bias (inherent to the LLM; out of harness scope)
- User-input contamination (discipline required)
- Network-tool bypass (not yet blocked; could be added via Claude Code settings)
- Model stealing context across `/model` switches in the same session (model changes but conversation tokens remain)

When a new vector is found, document it here and add a defense if one is feasible.
