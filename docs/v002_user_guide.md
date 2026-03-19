# Sara Brain v002 — User Guide

> A path-of-thought brain simulation that learns, remembers, and reasons through actual neuron chains — not activation levels, not pattern matching, not forgetting.

---

## Table of Contents

1. [Philosophy & Design Reasoning](#philosophy--design-reasoning)
2. [How the Brain Works](#how-the-brain-works)
3. [Associations & Question Words](#associations--question-words)
4. [Categories](#categories)
5. [LLM Translation (Claude-only)](#llm-translation-claude-only)
6. [Complete REPL Reference](#complete-repl-reference)
7. [Storage Schema](#storage-schema)
8. [Testing](#testing)

---

## Philosophy & Design Reasoning

Sara Brain was designed from five principles, each chosen to solve a specific problem found in traditional AI/neural-network approaches.

### Why path-of-thought instead of activation levels

Traditional neural networks and spreading-activation models assign floating-point "activation levels" to nodes. When you ask "why did you recognize apple?", the answer is a matrix of weights — opaque, unexplainable.

Sara Brain records actual paths through neuron chains. Recognition traces **real recorded paths**, not statistical weights. You can ask `why apple` and get back:

```
red → apple_color → apple  (from: "an apple is red")
round → apple_shape → apple  (from: "an apple is round")
```

Every conclusion is explainable because it's a walk through neurons you can see.

There's a second problem: **false fanout**. In activation-based systems, if "red" activates a shared "fruit_color" node, that activation leaks to every fruit — banana gets boosted when only apple should. Sara Brain solves this with concept-specific relation neurons: `apple_color`, `banana_color`, never a shared `fruit_color`. Paths stay separate. Cross-concept leaking is impossible.

### Why no forgetting / no decay

Most cognitive models implement memory decay — neurons weaken over time, unused paths get pruned. This forces the system to make lossy decisions about what to keep.

Sara Brain **never forgets**. No decay, no pruning, no garbage collection. Strength only increases (via the formula `1 + ln(1 + traversals)`). Instead of using forgetting to manage relevance, Sara Brain uses **path similarity**: the overlap between downstream paths tells you how related two concepts are, without discarding anything.

This means:
- Teaching a fact once is permanent
- Teaching the same fact again strengthens it
- Old facts remain accessible forever
- Relevance is computed dynamically from path structure, not from arbitrary decay curves

### Why parallel wavefront propagation

Sequential BFS (breadth-first search) from a single starting neuron finds all reachable nodes, but it doesn't tell you which nodes are *convergently reached* — the nodes where multiple independent lines of evidence meet.

Sara Brain launches **parallel wavefronts**, one per input neuron, simultaneously. Each wavefront independently explores the graph. Recognition happens at **intersections** — neurons reached by 2+ wavefronts. This is analogous to quantum superposition: each wavefront is a "possibility" and observation (intersection) is where they collapse into a recognized concept.

```
Input: "red, round"

Wavefront 1 (red):   red → apple_color → apple
Wavefront 2 (round): round → apple_shape → apple
                                             ↑
                                        INTERSECTION → recognized
```

### Why concept-specific relation neurons

When you teach "an apple is red", the system creates the relation neuron `apple_color`, not `fruit_color`. This is critical:

- `fruit_color` would be shared by apple, banana, cherry — any property reaching it would fan out to all fruits
- `apple_color` is private to apple — red reaches apple through `apple_color`, and only apple

The relation label is generated as `{subject}_{property_type}`. The taxonomy tells us "red" is a `color`, the subject is "apple", so the relation neuron is `apple_color`.

### Why zero external dependencies

Sara Brain's core is **stdlib-only Python + SQLite** (which is built into Python). No numpy, no torch, no networkx. This is deliberate:

- SQLite handles persistence, concurrency (WAL mode), and indexing with zero setup
- Python's standard library provides everything needed: `dataclasses`, `math.log`, `cmd.Cmd`, `sqlite3`, `json`
- The only optional dependency is the Anthropic API for LLM translation, and that uses `urllib.request` — still stdlib

This means the brain runs anywhere Python 3.11+ runs, with no install step beyond `pip install -e .`

---

## How the Brain Works

### Data Model

The brain stores three core entities:

#### Neurons (4 types)

| Type | Purpose | Example |
|------|---------|---------|
| `concept` | The subject being described | `apple`, `banana`, `circle` |
| `property` | An attribute | `red`, `round`, `sweet` |
| `relation` | Intermediate node linking property → concept | `apple_color`, `circle_shape` |
| `association` | Groups properties under a named category | `taste`, `mood` |

Each neuron has a unique `label`, a `neuron_type`, a `created_at` timestamp, and optional JSON `metadata`.

#### Segments (directed edges)

A segment connects two neurons with:

| Field | Description |
|-------|-------------|
| `source_id` | Origin neuron |
| `target_id` | Destination neuron |
| `relation` | Semantic label (e.g., `has_color`, `describes`, `is_a`) |
| `strength` | `1 + ln(1 + traversals)` — starts at 1.0, only goes up |
| `traversals` | How many times this edge has been walked |
| `created_at` | When the segment was first created |
| `last_used` | When the segment was last traversed |

The strength formula `1 + ln(1 + traversals)` is logarithmic — early traversals matter most, later ones add diminishing increments. After 1 traversal: 1.69. After 10: 3.40. After 100: 5.62. Strength never decreases.

#### Paths (provenance)

A path is a recorded chain of segments representing a single learned fact.

| Field | Description |
|-------|-------------|
| `origin_id` | Starting neuron (property) |
| `terminus_id` | Ending neuron (concept) |
| `source_text` | The original natural-language statement |
| `created_at` | When the path was learned |

Each path has ordered **path steps**, each referencing one segment. A typical path has 2 steps (property → relation → concept).

### Learning Pipeline

When you type `teach an apple is red`, the following happens:

```
1. PARSING
   "an apple is red"
   → strip articles: "apple is red"
   → find verb: "is" at position 1
   → subject = "apple" (singularized)
   → object = "red"
   → original text preserved for provenance

2. TAXONOMY LOOKUP
   → property_type("red") = "color"   (from built-in property map)
   → subject_category("apple") = "fruit"   (from built-in category map)
   → relation_label("apple", "red") = "apple_color"

3. CHAIN BUILDING
   → Get or create PROPERTY neuron: "red"
   → Get or create CONCEPT neuron: "apple"
   → Get or create RELATION neuron: "apple_color"
   → Get or create segment: red → apple_color (relation: "has_color")
   → Get or create segment: apple_color → apple (relation: "describes")
   → If segments already exist: strengthen them (+1 traversal)

4. PERSISTENCE
   → Create path record: origin=red, terminus=apple, source_text="an apple is red"
   → Create path_step records linking to segments
   → Commit to SQLite
```

Key behaviors:
- **Neuron reuse**: If "red" already exists as a property neuron, it's reused
- **Segment strengthening**: Teaching the same fact again increments traversals
- **Concept-specific relations**: Every subject gets its own relation neuron

### Recognition Algorithm

When you type `recognize red, round`:

```
1. RESOLVE INPUTS
   → "red" → property neuron (id=1)
   → "round" → property neuron (id=2)

2. LAUNCH PARALLEL WAVEFRONTS
   Wavefront from "red":
     BFS depth 0: [red]
     BFS depth 1: red → apple_color
     BFS depth 2: apple_color → apple
     Reached: {apple_color: [[red, apple_color]], apple: [[red, apple_color, apple]]}

   Wavefront from "round":
     BFS depth 0: [round]
     BFS depth 1: round → apple_shape
     BFS depth 2: apple_shape → apple
     Reached: {apple_shape: [[round, apple_shape]], apple: [[round, apple_shape, apple]]}

3. FIND INTERSECTIONS
   Neuron "apple" reached by both wavefronts → RECOGNIZED
   Confidence = 2 (number of converging paths)

4. STRENGTHEN TRAVERSED SEGMENTS
   All segments walked during recognition get +1 traversal
   This means recognition makes the brain more confident next time

5. RETURN RESULTS
   apple (2 converging paths)
     red → apple_color → apple
     round → apple_shape → apple
```

Results are sorted by confidence (most converging paths first).

### Worked Example: Teach 3 Facts, Then Recognize

```
sara> teach an apple is red
  Created path: red → apple_color → apple (3 new neurons, 2 new segments)

sara> teach an apple is round
  Created path: round → apple_shape → apple (2 new neurons, 2 new segments)

sara> teach a banana is yellow
  Created path: yellow → banana_color → banana (3 new neurons, 2 new segments)
```

After these 3 teachings, the brain has:
- **8 neurons**: red, apple_color, apple, round, apple_shape, yellow, banana_color, banana
- **6 segments**: red→apple_color, apple_color→apple, round→apple_shape, apple_shape→apple, yellow→banana_color, banana_color→banana
- **3 paths** with provenance back to original statements

```
sara> recognize red, round
  #1 apple (2 converging paths)
    red → apple_color → apple
    round → apple_shape → apple
```

Only apple appears because only apple is reached by both wavefronts. Banana is reached by neither (no "red" or "round" path leads to it). The relation neurons `apple_color` and `apple_shape` are private to apple, preventing false fanout.

```
sara> why apple
  3 paths of thought lead to "apple":
    1. red → apple_color → apple (from: "an apple is red")
    2. round → apple_shape → apple (from: "an apple is round")

sara> trace red
  Paths from "red":
    red → apple_color → apple

sara> similar red
  red ↔ round (shared: 2, overlap: 50.0%)
```

---

## Associations & Question Words

Associations let you create custom property groupings beyond the built-in taxonomy.

### Built-in Associations

The taxonomy ships with these property types:

| Type | Properties |
|------|-----------|
| `color` | red, blue, green, yellow, orange, purple, black, white, brown, pink, crimson |
| `shape` | round, square, triangular, flat, oval, cylindrical, spherical |
| `taste` | sweet, sour, bitter, salty, savory, spicy |
| `texture` | smooth, rough, soft, hard, fuzzy, crunchy |
| `size` | big, small, large, tiny, huge |
| `temperature` | hot, cold, warm, cool |

### Dynamic Associations

You can define new associations at runtime:

```
sara> define mood how
  Created association: mood (question word: "how")

sara> describe mood as happy, sad, angry, calm
  Registered 4 properties under "mood":
    happy → mood
    sad → mood
    angry → mood
    calm → mood
```

Now you can teach with these properties:

```
sara> teach a puppy is happy
  Created path: happy → puppy_mood → puppy (3 new neurons, 2 new segments)
```

The taxonomy now knows "happy" is a `mood` property, so the relation neuron becomes `puppy_mood`.

### The `define` / `describe` Workflow

1. **`define <name> <question_word>`** — Creates an ASSOCIATION neuron and registers a question word
2. **`describe <name> as <prop1>, <prop2>, ...`** — Registers properties under the association, creating PROPERTY neurons and `is_a` segments

### Query Resolution

Once associations and properties are registered, you can query them:

```
sara> what apple color
  apple color: red

sara> how apple taste
  apple taste: sweet
```

The resolution chain:
1. Question word `what` → resolves to associations: `[color, shape, size]`
2. Subject `apple` → look up the concept neuron
3. Association `color` → find all properties registered under `color`
4. Find paths ending at `apple` whose origin is one of those properties
5. Return matches: `red`

### Built-in Question Words

| Question Word | Associations |
|---------------|-------------|
| `what` | color, shape, size |
| `how` | taste, texture, temperature |

Dynamic associations add their own question words via `define`.

---

## Categories

Categories are simple tags on concepts. They influence relation neuron naming.

### Default Behavior

If a concept isn't categorized, it defaults to `"thing"`. When you teach "a rock is hard", and "rock" has no category, the relation neuron becomes `rock_texture` (subject + property type). The category doesn't appear in the relation label — categories are used for the built-in taxonomy lookup only.

### Built-in Categories

The taxonomy ships with categories for common subjects:

| Category | Subjects |
|----------|---------|
| `fruit` | apple, banana, orange, grape, strawberry, lemon, cherry, mango, pear, peach |
| `geometric` | circle, square, triangle, rectangle, sphere, cube |
| `animal` | dog, cat, bird, fish, horse |
| `vehicle` | car, truck, bus, firetruck |

### Overriding Categories

```
sara> categorize apple item
  Categorized "apple" as "item".

sara> categories
  animal: bird, cat, dog, fish, horse
  fruit: banana, cherry, grape, lemon, mango, orange, pear, peach, strawberry
  geometric: circle, cube, rectangle, sphere, square, triangle
  item: apple
  vehicle: bus, car, firetruck, truck
```

Categories persist to the `categories` table in SQLite and reload on startup.

---

## LLM Translation (Claude-only)

Sara Brain has an optional natural language translation layer that converts free-form questions into structured REPL commands using Claude.

### Design Decisions

- **Anthropic Messages API only** — Uses `urllib.request` to call `https://api.anthropic.com/v1/messages` directly. No SDK dependency.
- **OpenAI domains explicitly blocked** — `api.openai.com`, `openai.azure.com`, and `api.openai.org` are rejected with a `ValueError`. This is intentional: the project uses Claude exclusively.
- **Zero-temperature deterministic translation** — The LLM is called with `temperature=0` for consistent structured output.
- **System prompt tells Claude the available commands** — The translator builds a system prompt listing every available structured command, then tells Claude to respond with ONLY the command or `UNKNOWN`.

### Setup

```
sara> llm set sk-ant-api03-xxxxx claude-sonnet-4-20250514
  LLM configured: claude-sonnet-4-20250514 @ https://api.anthropic.com

sara> llm status
  LLM: claude-sonnet-4-20250514 @ https://api.anthropic.com
```

### Usage

```
sara> ask what does an apple taste like?
  → how apple taste
  apple taste: sweet
```

The `ask` command:
1. Builds the list of available structured commands from registered question words
2. Sends the user's question + available commands to Claude
3. Claude returns a structured command (e.g., `how apple taste`)
4. The structured command is dispatched through the REPL

### Management

```
sara> llm status        # Check current config
sara> llm clear         # Remove LLM config
```

API key, model, and URL are stored in the `settings` table and persist across sessions.

---

## Complete REPL Reference

### Learning

| Command | Description | Example |
|---------|-------------|---------|
| `teach <statement>` | Learn a fact from natural language | `teach an apple is red` |

### Recognition

| Command | Description | Example |
|---------|-------------|---------|
| `recognize <inputs>` | Find concepts matching comma-separated properties | `recognize red, round` |

### Exploration

| Command | Description | Example |
|---------|-------------|---------|
| `trace <label>` | Show all outgoing paths from a neuron | `trace red` |
| `why <label>` | Show all paths leading to a neuron (with provenance) | `why apple` |
| `similar <label>` | Find neurons with shared downstream paths | `similar red` |
| `analyze` | Scan all neurons for path similarities | `analyze` |

### Associations

| Command | Description | Example |
|---------|-------------|---------|
| `define <name> <question_word>` | Create a new association with a question word | `define taste how` |
| `describe <name> as <props>` | Register properties under an association | `describe mood as happy, sad` |
| `associations` | List all associations and their properties | `associations` |

### Queries

| Command | Description | Example |
|---------|-------------|---------|
| `<question_word> <concept> <association>` | Query properties via question word | `what apple color` |
| `questions` | List all available question words | `questions` |

### Categories

| Command | Description | Example |
|---------|-------------|---------|
| `categorize <concept> <category>` | Tag a concept with a category | `categorize apple item` |
| `categories` | List all categories and their members | `categories` |

### LLM

| Command | Description | Example |
|---------|-------------|---------|
| `ask <question>` | Translate natural language to structured command via Claude | `ask what color is an apple?` |
| `llm set <key> [model]` | Configure Claude API key and model | `llm set sk-ant-... claude-sonnet-4-20250514` |
| `llm status` | Show current LLM configuration | `llm status` |
| `llm clear` | Remove LLM configuration | `llm clear` |

### Inspection

| Command | Description | Example |
|---------|-------------|---------|
| `neurons` | List all neurons | `neurons` |
| `paths` | List all recorded paths | `paths` |
| `stats` | Show brain statistics | `stats` |

### System

| Command | Description |
|---------|-------------|
| `save` | Force flush to disk |
| `quit` / `exit` | Exit the REPL |

---

## Storage Schema

All state is persisted to a single SQLite database file (default: `sara.db`). The database uses **WAL (Write-Ahead Logging)** mode for concurrent read performance and has **foreign keys enabled**.

### Tables

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `neurons` | All neurons (concept, property, relation, association) | `id`, `label` (unique), `neuron_type`, `metadata` |
| `segments` | Directed edges between neurons | `source_id`, `target_id`, `relation`, `strength`, `traversals` |
| `paths` | Recorded fact chains with provenance | `origin_id`, `terminus_id`, `source_text` |
| `path_steps` | Ordered steps within a path | `path_id`, `step_order`, `segment_id` |
| `similarities` | Cached similarity analysis results | `neuron_a_id`, `neuron_b_id`, `shared_paths`, `overlap_ratio` |
| `associations` | Dynamic association → property mappings | `association`, `property_label`, `neuron_id` |
| `question_words` | Association → question word mappings | `association` (PK), `question_word` |
| `categories` | Concept → category tags | `label` (PK), `category` |
| `settings` | Key-value config (LLM settings, etc.) | `key` (PK), `value` |

### Indexes

- `idx_seg_source` — segments by source_id, strength DESC (fast outgoing lookups)
- `idx_seg_target` — segments by target_id (fast incoming lookups)
- `idx_neuron_label` — neurons by label (fast name resolution)
- `idx_neuron_type` — neurons by type (fast type filtering)
- `idx_path_terminus` — paths by terminus_id (fast "why" queries)

### Future: data-nut-squirrel Migration

The current SQLite storage is designed as a stepping stone. The plan is to migrate to [data-nut-squirrel](https://github.com/LunarFawn/data-nut-squirrel), a custom storage engine, when the project's needs outgrow SQLite's capabilities.

---

## Testing

166 tests across 16 test files. All tests use in-memory SQLite (`:memory:`) for isolation.

| File | Tests | Coverage |
|------|-------|----------|
| `test_models.py` | 8 | Pure dataclass behavior — Neuron, Segment strengthen(), Path, PathTrace, RecognitionResult |
| `test_storage.py` | 10 | SQLite repos — CRUD, WAL mode, segment strengthening, get_outgoing, get_paths_to |
| `test_parser.py` | 6 | Taxonomy lookups, statement parsing, singularization, relation label generation |
| `test_learner.py` | 7 | Chain creation, neuron reuse, segment strengthening on reteach, path step recording |
| `test_recognizer.py` | 7 | Wavefront propagation, intersection detection, trace, why, segment strengthening on recognition |
| `test_similarity.py` | 3 | Shared-path analysis, get_similar, no-similarity case |
| `test_integration.py` | 8 | End-to-end: teach → recognize → persistence → visualization → stats |
| `test_associations.py` | 10 | Define, describe, list, persistence across restart, taxonomy integration |
| `test_categories.py` | 10 | Categorize, override builtins, persistence, CategoryRepo CRUD |
| `test_query.py` | 12 | Built-in queries, dynamic queries, question word resolution, persistence |
| `test_translator.py` | 12 | System prompt, translate, blocked domains, URL building, error handling |
| `test_vision.py` | 20 | VisionObserver methods, sanitization security, API payload, blocked domains |
| `test_perceiver.py` | 22 | Full perception loop, correction, observation addition |
| `test_vision_proxy.py` | 8 | CORS proxy health, headers, forwarding, error passthrough |

### Running Tests

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest tests/ -v
```

All 166 tests should pass in under 5 seconds.
