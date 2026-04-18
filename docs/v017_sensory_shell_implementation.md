# v017 — Sensory Shell: Implementation Guide

**Date:** April 17, 2026
**Author:** Jennifer Pearl
**Branch:** signed_refutation_paths

---

## Quick Start

```bash
# From the repo root with .venv activated
pip install -e .
sara-shell --db my_brain.db
```

The shell starts empty. Teach it facts, then query.

```
you> /teach apples are red
Learned: apples are red
  path #1

you> /teach apples are round
Learned: apples are round
  path #2

you> /query apple
apples are red.  [path #1, source: apples are red]
apples are round.  [path #2, source: apples are round]
  Confidence: 2

you> red round
apple (confidence 2)
apples are red.  [path #1, source: apples are red]
    trace: red -> apple_color -> apple
apples are round.  [path #2, source: apples are round]
    trace: round -> apple_shape -> apple
  Confidence: 2
```

---

## Module Structure

```
src/sara_brain/sensory/
    __init__.py       Public API: SensoryShell, ShellResponse, Tokenizer, etc.
    shell.py          The empty processing engine
    tokenizer.py      Text -> wavefront seeds (no learned vocabulary)
    renderer.py       Converging paths -> text with provenance
    session.py        Conversation context (not persisted)
    cli.py            sara-shell interactive interface
```

---

## API Usage

### Basic: Process Text

```python
from sara_brain.core.brain import Brain
from sara_brain.sensory import SensoryShell

brain = Brain("sara.db")
shell = SensoryShell(brain)

# Teach facts (uses Brain's existing teach API)
brain.teach("methane has one carbon atom")
brain.teach("methane has four hydrogen atoms")
brain.teach("methane is a chemical compound")

# Process input — wavefronts propagate, paths converge
response = shell.process("one carbon atom four hydrogen atoms")
print(response.text)
# methane (confidence 2)
# methane has one carbon atom.  [path #1, source: ...]
#     trace: one carbon atom -> methane_attribute -> methane
# methane has four hydrogen atoms.  [path #2, source: ...]
#     trace: four hydrogen atoms -> methane_attribute -> methane

# Every source is traceable
for src in response.sources:
    print(f"  path #{src.path_id}: {src.source_text}")
```

### Direct Query

```python
response = shell.query("methane")
# Returns all paths TO and FROM "methane" with provenance

print(f"Confidence: {response.confidence}")
print(f"Gaps: {response.gaps}")  # concepts Sara doesn't know
```

### ShellResponse Fields

```python
@dataclass
class ShellResponse:
    text: str                              # rendered output with provenance
    sources: list[SourcedLine]             # each line with path_id, source_text
    confidence: int                        # max convergence count
    gaps: list[str]                        # tokens Sara doesn't know
    tokens: list[str]                      # input tokens after tokenization
    recognition: list[RecognitionResult]   # raw convergence results
```

### SourcedLine Fields

```python
@dataclass
class SourcedLine:
    text: str                    # the rendered fact
    path_id: int | None          # which path produced this
    source_text: str | None      # original taught statement
    weight: float                # path weight (strength sum)
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/teach <fact>` | Teach Sara a fact (e.g., `/teach apples are red`) |
| `/query <topic>` | What does Sara know about this topic? |
| `/stats` | Show neuron, segment, and path counts |
| `/clear` | Clear session context |
| `/help` | Show available commands |
| `/quit` | Exit |

Anything not starting with `/` is processed through the wavefront engine.

---

## CLI Flags

```
sara-shell --db <path>          Database path (default: ~/.sara_brain/sara.db)
sara-shell --no-provenance      Hide path IDs and source text in output
```

---

## How Processing Works (Step by Step)

### 1. Tokenization

Input: `"What has one carbon atom and four hydrogen atoms?"`

Tokenizer strips stopwords (what, has, and), lowercases, and does greedy longest-match against Sara's neuron labels:

```
Tokens: ["one carbon atom", "four hydrogen atoms"]
         (if these are stored as multi-word neurons)
   or:  ["one", "carbon", "atom", "four", "hydrogen", "atoms"]
         (if they're separate neurons)
```

If a singularized form ("atoms" -> "atom") isn't found in the graph but the original is, the original is kept.

### 2. Wavefront Seeding

Each token becomes a seed. The shell opens a short-term scratchpad (read-only, no graph mutation) and launches wavefronts.

### 3. Multi-Threshold Propagation

Three passes at different inhibition levels:
- **0.5** (focused) — only strong edges
- **0.3** (relaxed) — medium edges included
- **0.1** (open) — weak edges included

Each pass runs bidirectional echo propagation for up to 3 rounds.

### 4. Convergence

The short-term scratchpad tracks which neurons were reached by which seeds. Neurons reached by 2+ independent wavefronts are convergence points. Only CONCEPT neurons are returned — intermediate RELATION and PROPERTY nodes are shown in the trace, not as top-level results.

### 5. Output

Each converging concept is rendered with:
- Its confidence (convergence count)
- Each contributing path's source_text
- The full trace through intermediate nodes

---

## Testing

```bash
# Run sensory module tests only
python -m pytest tests/test_sensory.py -v

# Run all tests (verify nothing else broke)
python -m pytest tests/ -v
```

### Test Coverage

| Test Class | Count | What It Tests |
|-----------|-------|---------------|
| `TestTokenizer` | 9 | Word splitting, stopword removal, phrase matching, singularization |
| `TestRenderer` | 3 | No-paths output, provenance formatting |
| `TestSensoryShell` | 8 | Empty brain, teach-then-query, convergence, gaps, response types |
| `TestSession` | 4 | Topic tracking, dedup, clear, max history |

---

## Dependencies

### What the sensory module imports from (shared modules only):

| Import | Purpose |
|--------|---------|
| `core.brain.Brain` | Wavefront propagation, teach, why, trace |
| `core.short_term.ShortTerm` | Session scratchpad |
| `models.neuron.NeuronType` | Filter to CONCEPT neurons only |
| `models.result.RecognitionResult, PathTrace` | Convergence results |
| `config.default_db_path` | Default database location |

### What it does NOT import:

- Nothing from `cortex/` — clean break
- Nothing from `nlp/` — no LLM clients
- Nothing from `agent/` — no ReAct loop
- No external packages — stdlib only

---

## Extending

### Adding a New Sense Modality

The shell currently handles text. To add vision, audio, or other modalities:

1. Extract observations from the new modality into word tokens
2. Feed those tokens through `shell.process()` or directly as seed labels
3. The wavefront engine handles the rest — same convergence, same provenance

The shell doesn't care where the tokens came from. Text, vision labels, audio transcriptions — they all become wavefront seeds.

### Connecting to the Cortex

The cortex module handles NL parsing (question detection, statement extraction, disambiguation). The shell does not. To combine them:

```python
from sara_brain.cortex import Cortex
from sara_brain.sensory import SensoryShell

brain = Brain("sara.db")
cortex = Cortex(brain)      # language understanding
shell = SensoryShell(brain)  # graph traversal with provenance

# Use cortex to parse, shell to query
parsed = cortex.parser.parse("what is methane")
if parsed.is_question:
    response = shell.query(parsed.topics[0])
```

### Batch Teaching

Use the existing batch_teach script or teach programmatically:

```python
brain = Brain("sara.db")
facts = [
    "directional selection is selection for one extreme phenotype",
    "stabilizing selection is selection for the average phenotype",
    "disruptive selection is selection for both extreme phenotypes",
]
for fact in facts:
    result = brain.teach(fact)
    if result:
        print(f"Taught: {fact} (path #{result.path_id})")
    else:
        print(f"Parse failed: {fact}")
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/sara_brain/sensory/__init__.py` | New — public API |
| `src/sara_brain/sensory/shell.py` | New — empty processing engine |
| `src/sara_brain/sensory/tokenizer.py` | New — text to wavefront seeds |
| `src/sara_brain/sensory/renderer.py` | New — paths to text with provenance |
| `src/sara_brain/sensory/session.py` | New — conversation context |
| `src/sara_brain/sensory/cli.py` | New — sara-shell CLI |
| `tests/test_sensory.py` | New — 24 tests |
| `pyproject.toml` | Added `sara-shell` console script entry |
