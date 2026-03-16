# Sara Brain — Foundation

## Overview

Sara Brain is a path-of-thought brain simulation. It learns facts expressed as natural language statements, stores them as directed neuron-segment chains in SQLite, and recognizes concepts by launching parallel wavefronts from input neurons and finding intersections.

## Core Principles

1. **Path-of-thought, not activation levels** — Recognition follows actual recorded paths through neuron chains, not spreading activation or pattern matching.
2. **Never forgets** — No decay, no pruning. Strength only increases. Path similarity is used instead of forgetting.
3. **Parallel wavefront propagation** — Multiple wavefronts launch simultaneously from input neurons. Commonality across paths determines recognition.

## Data Model

### Neurons
Three types:
- **concept** — The subject being described (e.g., "apple")
- **property** — An attribute (e.g., "red", "round")
- **relation** — An intermediate node linking property to concept (e.g., "apple_color")

### Segments
Directed edges between neurons. Each segment has:
- `source_id`, `target_id` — the connected neurons
- `relation` — semantic label (e.g., "has_color", "describes")
- `strength` — starts at 1.0, increases with traversals via `1 + ln(1 + traversals)`
- `traversals` — count of times traversed

### Paths
Recorded chains of segments representing a learned fact.
- `origin_id` — starting neuron (property)
- `terminus_id` — ending neuron (concept)
- `source_text` — the original statement
- Steps ordered by `step_order`, each referencing a segment

## Architecture

```
src/sara_brain/
  models/       — Pure dataclasses (Neuron, Segment, Path, PathTrace, RecognitionResult)
  storage/      — SQLite repos (NeuronRepo, SegmentRepo, PathRepo, Database)
  parsing/      — Statement parser and taxonomy
  core/         — Brain orchestrator, Learner, Recognizer, SimilarityAnalyzer
  visualization/— ASCII tree and Graphviz DOT export
  repl/         — Interactive shell
```

## Storage

SQLite with WAL mode and foreign keys. Schema in `storage/schema.sql`. Five tables: `neurons`, `segments`, `paths`, `path_steps`, `similarities`.

## Summary of Capabilities

- **Parsing**: taxonomy, statement parser, singularization
- **Learning**: 3-neuron chains, segment strengthening, neuron reuse
- **Recognition**: parallel wavefronts, intersection detection, trace, why
- **Similarity & visualization**: shared-path analysis, ASCII tree, Graphviz DOT

## REPL

Interactive shell (`sara_brain.repl.shell`) invoked via `sara` CLI command. Commands:

- `teach <statement>` — Learn a fact
- `recognize <inputs>` — Recognize from comma-separated properties
- `trace <label>` — Show all outgoing paths from a neuron
- `why <label>` — Show all paths that lead to a neuron with provenance
- `similar <label>` — Find neurons with shared downstream paths
- `tree <label>` — ASCII tree visualization
- `dot` — Full Graphviz DOT export
- `stats` — Brain statistics
- `neurons` — List all neurons
- `segments` — List all segments
- `paths` — List all paths
- `quit` / `exit` — Exit

## Test Suite

56 tests across 8 test files:

| File | Coverage |
|------|----------|
| `test_models.py` | Pure dataclass behavior (Neuron, Segment, Path, PathTrace, RecognitionResult) |
| `test_storage.py` | SQLite repos (CRUD, WAL mode, strengthening) |
| `test_parser.py` | Taxonomy lookups, statement parsing, singularization |
| `test_learner.py` | Chain creation, neuron reuse, segment strengthening |
| `test_recognizer.py` | Wavefront propagation, intersection, trace, why |
| `test_similarity.py` | Shared-path analysis |
| `test_integration.py` | End-to-end: teach -> recognize -> persistence -> visualization |

## Demo

`demos/apple_demo.py` — Full walkthrough teaching fruits and shapes, then demonstrating recognition, trace, why, similarity, path tree, and DOT export.

## Project Setup

```
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v
.venv/bin/python demos/apple_demo.py
```

## Bugs Fixed During Initial Bring-Up

1. **Build backend** — `pyproject.toml` had invalid `setuptools.backends._legacy:_Backend`, fixed to `setuptools.build_meta`
2. **Singularization** — `_singularize("apples")` returned `"appl"` because the `-es` rule fired before `-s`. Fixed to only strip `-es` for specific endings (`-ches`, `-shes`, `-xes`, `-zes`, `-ses`)
3. **Threading data race** — `Recognizer` used `ThreadPoolExecutor` sharing a single SQLite connection across threads, causing `ProgrammingError` and corrupted reads. Fixed by removing threading (Python GIL provides no benefit for CPU-bound SQLite operations)
4. **False recognition fanout** — Shared category-based relation neurons (e.g., `fruit_color`) caused unrelated properties to reach all concepts in the same category. Fixed by making relation neurons concept-specific (e.g., `apple_color`, `banana_color`)
