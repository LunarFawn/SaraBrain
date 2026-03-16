# SaraBrain
Finally trying to code up my idea for AI I had back in the 90's

---

## What is Sara Brain?

A **path-of-thought** brain simulation. It learns facts expressed as natural-language statements, stores them as directed neuron-segment chains in SQLite, and recognizes concepts by launching parallel wavefronts from input neurons and finding intersections.

This is *not* a neural network, LLM, or activation-based system. Recognition follows actual recorded paths through neuron chains — no decay, no forgetting, no pattern matching.

## Core Principles

1. **Path-of-thought, not activation levels** — Recognition traces real recorded paths, not spreading activation
2. **Never forgets** — Strength only increases. Path similarity replaces forgetting
3. **Parallel wavefront propagation** — Multiple wavefronts launch simultaneously; commonality across paths determines recognition

## Quick Start

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run the demo
.venv/bin/python demos/apple_demo.py

# Run tests
.venv/bin/pytest tests/ -v

# Launch the REPL
.venv/bin/sara
```

Requires Python 3.11+.

## How It Works

### Data Model

- **Neurons** — three types: `concept` (apple), `property` (red), `relation` (apple_color)
- **Segments** — directed edges between neurons with strength (`1 + ln(1 + traversals)`) and traversal counts
- **Paths** — recorded chains of segments representing a learned fact, with source text provenance

### Learning

Teach it facts in natural language:

```
sara> teach an apple is red
sara> teach an apple is round
sara> teach a banana is yellow
sara> teach a banana is long
```

Each statement creates a 3-neuron chain: property → relation → concept.

### Recognition

Give it properties and it finds matching concepts via parallel wavefront intersection:

```
sara> recognize red, round
Recognized: apple (score: 1.00)
```

### Exploration

```
sara> trace apple       # All outgoing paths from a neuron
sara> why apple         # All paths leading to a neuron with provenance
sara> similar apple     # Neurons with shared downstream paths
sara> tree apple        # ASCII tree visualization
sara> dot               # Full Graphviz DOT export
```

## Architecture

```
src/sara_brain/
  models/        — Pure dataclasses (Neuron, Segment, Path, PathTrace, RecognitionResult)
  storage/       — SQLite repos (NeuronRepo, SegmentRepo, PathRepo, Database)
  parsing/       — Statement parser and taxonomy
  core/          — Brain orchestrator, Learner, Recognizer, SimilarityAnalyzer
  visualization/ — ASCII tree and Graphviz DOT export
  repl/          — Interactive shell
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `teach <statement>` | Learn a fact |
| `recognize <inputs>` | Recognize from comma-separated properties |
| `trace <label>` | Show all outgoing paths from a neuron |
| `why <label>` | Show all paths leading to a neuron |
| `similar <label>` | Find neurons with shared downstream paths |
| `tree <label>` | ASCII tree visualization |
| `dot` | Full Graphviz DOT export |
| `stats` | Brain statistics |
| `neurons` / `segments` / `paths` | List all |
| `quit` / `exit` | Exit |

## Tests

56 tests covering models, storage, parsing, learning, recognition, similarity, and end-to-end integration.

```bash
.venv/bin/pytest tests/ -v
```

## Storage

SQLite with WAL mode and foreign keys. Schema in `src/sara_brain/storage/schema.sql`. Current plan is SQLite now, with migration to [data-nut-squirrel](https://github.com/LunarFawn/data-nut-squirrel) later.

## License

See [LICENSE](LICENSE) if present.
