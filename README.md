# SaraBrain
Finally trying to code up my idea for AI I had back in the 90's

---

## What is Sara Brain?

A **path-of-thought** brain simulation. It learns facts expressed as natural-language statements, stores them as directed neuron-segment chains in SQLite, and recognizes concepts by launching parallel wavefronts from input neurons and finding intersections.

This is *not* a neural network, LLM, or activation-based system. Recognition follows actual recorded paths through neuron chains — no decay, no forgetting, no pattern matching.

## Authenticity

Every essay I publish on Substack (at **Path of Thought**) or anywhere else is first committed to this repository, in the `substack/` directory, before it appears on the public venue. The git history is a timestamped, public record of everything I have written under the Path of Thought name.

**If you see a post attributed to me that is not in this repository, it is not from me.** Screenshots can be faked. Substack accounts can be impersonated. AI-generated text can be attributed to anyone. But this repository has been visibly mine for a long time, with consistent work, published research, and real code — it cannot be spoofed by anyone who does not have years of my commit history behind them.

To verify a post: search this repository for a distinctive phrase from the post. If you find it, look at the commit that introduced it — the commit author will be me, and the timestamp will precede the Substack publication date. If you cannot find the text in this repository at all, the text did not come from me.

This authenticity claim applies to written essays under the Path of Thought name. It does not mean every word I have ever written lives in this repo — private conversations, unpublished drafts, and work in other contexts are not required to be here. The claim is narrower and stronger: **anything I publish publicly as a Path of Thought essay will be committable-and-committed here first, and anything claimed to be a Path of Thought essay that is not here is not from me.**

— Jennifer Pearl

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

### Documentation

| Doc | Contents |
|-----|----------|
| **[docs/v001_foundation.md](docs/v001_foundation.md)** | Architecture foundation, data model, storage schema |
| **[docs/v002_user_guide.md](docs/v002_user_guide.md)** | Design philosophy, algorithms, associations, categories, LLM, full REPL reference |
| **[docs/v003_perception.md](docs/v003_perception.md)** | Vision perception, cognitive development model, tribal trust, correction, security |
| **[docs/v004_web_app.md](docs/v004_web_app.md)** | Web app guide: guided UI, image viewer, region selection, neural graph, vision proxy |
| **[docs/v005_design_philosophy.md](docs/v005_design_philosophy.md)** | Design philosophy + user guide: origin story, why paths not activation, never forgets, parallel thought, tribal trust, "You Need More Than Attention" (transformers as sensory cortex, inventor skepticism, shared computational roots — 22 academic references), complete REPL reference |

## How It Works

### Data Model

- **Neurons** — four types: `concept` (apple), `property` (red), `relation` (apple_color), `association` (taste)
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
```

## Architecture

```
src/sara_brain/
  models/        — Pure dataclasses (Neuron, Segment, Path, PathTrace, RecognitionResult)
  storage/       — SQLite repos (NeuronRepo, SegmentRepo, PathRepo, AssociationRepo, CategoryRepo, SettingsRepo, Database)
  parsing/       — Statement parser and taxonomy
  core/          — Brain orchestrator, Learner, Recognizer, SimilarityAnalyzer, Perceiver
  nlp/           — VisionObserver (Claude Vision), LLM translator (Claude-only)
  visualization/ — ASCII tree and Graphviz DOT export
  repl/          — Interactive shell with commands and formatters
```

## REPL Commands

| Command | Description |
|---------|-------------|
| `teach <statement>` | Learn a fact |
| `recognize <inputs>` | Recognize from comma-separated properties |
| `trace <label>` | Show all outgoing paths from a neuron |
| `why <label>` | Show all paths leading to a neuron |
| `similar <label>` | Find neurons with shared downstream paths |
| `analyze` | Scan all neurons for path similarities |
| `define <name> <qword>` | Create a new association with a question word |
| `describe <name> as <props>` | Register properties under an association |
| `associations` | List all associations and their properties |
| `<question_word> <concept> <assoc>` | Query properties (e.g., `what apple color`) |
| `questions` | List all available question words |
| `categorize <concept> <cat>` | Tag a concept with a category |
| `categories` | List all categories |
| `ask <question>` | Translate natural language via Claude LLM |
| `llm set <key> [model]` | Configure Claude API |
| `llm status` / `llm clear` | Check or remove LLM config |
| `perceive <path> [label]` | Run multi-phase image perception via Claude Vision |
| `no <correct_label>` | Correct a misidentification |
| `see <property>` | Point out a missed property |
| `stats` | Brain statistics |
| `neurons` / `paths` | List all |
| `save` | Force flush to disk |
| `quit` / `exit` | Exit |

## Tests

166 tests covering models, storage, parsing, learning, recognition, similarity, associations, categories, queries, LLM translation, vision, perception, proxy, and end-to-end integration.

```bash
.venv/bin/pytest tests/ -v
```

## Storage

SQLite with WAL mode and foreign keys. Schema in `src/sara_brain/storage/schema.sql`. Current plan is SQLite now, with migration to [data-nut-squirrel](https://github.com/LunarFawn/data-nut-squirrel) later.

## License

See [LICENSE](LICENSE) if present.
