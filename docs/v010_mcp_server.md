# v010 — Sara Brain MCP Server: Connecting Any LLM to the Cerebellum

## What This Is

Sara Brain is a path-of-thought brain simulation. It is **not** a neural network, not a vector database, not a chatbot memory layer. It is a persistent cerebellum — the part of the brain that validates, corrects, and remembers.

The MCP (Model Context Protocol) server exposes Sara Brain as a set of tools that any LLM client can call. Claude, Amazon Q, VS Code Copilot — anything that speaks MCP can use Sara as its persistent higher-order brain.

**The LLM is the sensory cortex.** It perceives, reasons, and proposes actions.
**Sara Brain is the cerebellum.** She validates proposals, catches conflicts, remembers outcomes, and guides the LLM based on what the user has taught her.

The user interacts with Sara through the LLM. Sara learns observationally from what happens.

---

## Table of Contents

1. [Installation and Setup](#1-installation-and-setup)
2. [How Sara Brain Thinks](#2-how-sara-brain-thinks)
3. [The Cerebellum Loop](#3-the-cerebellum-loop)
4. [Tool Reference](#4-tool-reference)
5. [Teaching Sara Effectively](#5-teaching-sara-effectively)
6. [The Validation Pattern](#6-the-validation-pattern)
7. [Observational Learning](#7-observational-learning)
8. [Configuration Examples](#8-configuration-examples)
9. [Architecture Notes for LLMs](#9-architecture-notes-for-llms)

---

## 1. Installation and Setup

### Install Sara Brain

```bash
cd /path/to/SaraBrain
pip install -e .
```

This installs two commands:
- `sara` — the interactive REPL for direct brain interaction
- `sara-mcp` — the MCP server (JSON-RPC 2.0 over stdio)

### Verify

```bash
sara-mcp --help
```

The MCP server reads from stdin and writes to stdout. It does not start a network listener. The LLM client launches it as a subprocess.

### Database

Sara Brain stores everything in a SQLite database. Default location:

```
~/.sara_brain/sara.db
```

This database is persistent. Everything Sara learns survives across sessions, restarts, and different LLM clients. The same brain can be shared between Claude, Amazon Q, and any other MCP client.

Override with `--db /path/to/sara.db` if needed.

---

## 2. How Sara Brain Thinks

Sara Brain does **not** use activation levels, embeddings, or attention. She uses **paths**.

### The Path Model

When Sara learns "apples are red", she creates:

```
red (property) → apple_color (relation) → apple (concept)
```

Three neurons connected by two segments, recorded as a path with the source text "apples are red" as provenance.

When Sara learns "apples are round":

```
round (property) → apple_shape (relation) → apple (concept)
```

Now "apple" has two paths converging on it. When you ask Sara to recognize `red, round`, she launches parallel wavefronts from both properties. Both wavefronts reach "apple" — that's 2 converging paths. Sara recognizes "apple" with confidence 2.

### Key Principles

- **Paths, not activation.** Recognition follows real recorded paths through the graph. There is no spreading activation, no decay, no probability.
- **Sara never forgets.** Strength grows logarithmically: `1.0 + ln(1 + traversals)`. Paths are never pruned.
- **Parallel wavefronts.** Multiple inputs are traced simultaneously. The answer is where wavefronts converge — like observing interference patterns.
- **Provenance.** Every path records its source text. Sara can always explain why she knows something.

---

## 3. The Cerebellum Loop

This is the core pattern for any LLM using Sara Brain:

```
User speaks (through the LLM)
        ↓
LLM (cortex) understands request, plans action
        ↓
LLM calls brain_validate with the plan
        ↓
Sara checks plan against known paths ←──────────┐
        ↓                                        │
Conflict?                                        │
   YES → Sara returns correction                 │
         LLM adjusts plan ──────────────────────→┘
   NO  → Sara returns APPROVED
        ↓
LLM executes action
        ↓
LLM calls brain_observe with the outcome
        ↓
Sara records what happened (learns)
        ↓
LLM responds to user
```

### Why This Matters

Without the cerebellum loop, an LLM will write code however it wants. With Sara, the LLM checks first: "Does Sara know anything about how this project does things?" If Sara was taught "project uses pytest" and the LLM was about to use unittest — Sara catches it.

This is what happened in the Amazon Q experiment (v009): the LLM with Sara produced parameterized, testable, well-structured code. The same LLM without Sara produced a flat script. The difference was Sara validating and guiding.

---

## 4. Tool Reference

### brain_teach

**Teach Sara a fact.** Creates neurons and segments forming a knowledge path.

```
Statement format: "<subject> is/are <property>"
                  "<subject> contains/includes <object>"
```

**Examples:**
- `"apples are red"` → creates path: red → apple_color → apple
- `"project uses pytest"` → creates path if parser recognizes the verb
- `"code style is snake_case"` → creates path: snake_case → code style_attribute → code style
- `"flask app contains authentication"` → creates path with "contains" relation
- `"tests include integration tests"` → creates path with "includes" relation

**When to use:** When the user states a fact, preference, or rule that should persist. The user is the teacher — their statements are trusted.

**Returns:** The path label (e.g., `"Learned: red → apple_color → apple"`) or a parse error if the statement couldn't be understood.

**Important:** The statement parser understands specific verb patterns. Supported verbs: `is`, `are`, `contains`, `includes`, `requires`, `follows`, `precedes`, `excludes`. If a statement can't be parsed, try rephrasing with "is" or "are".

---

### brain_query

**Ask Sara what she knows about a topic.** Returns all paths leading to and from a concept.

**Input:** A topic string (e.g., `"python"`, `"flask"`, `"testing"`)

**Returns:** Path traces with provenance. Example:

```
Paths leading to 'apple':
  red → apple_color → apple (from: "apples are red")
  round → apple_shape → apple (from: "apples are round")
Paths from 'apple':
  apple → fruit_type → fruit (from: "apple is a fruit")
```

**When to use:** When you need to understand what Sara knows about a subject before taking action.

---

### brain_recognize

**Give Sara properties and see what she recognizes.** Uses parallel wavefront propagation — the core recognition algorithm.

**Input:** Comma-separated properties (e.g., `"red, round, sweet"`)

**Returns:** Recognized concepts ranked by confidence (number of converging paths):

```
  apple (3 converging paths)
    path: red → apple_color → apple
    path: round → apple_shape → apple
    path: sweet → apple_taste → apple
```

**When to use:** When you have observations or properties and want to know what Sara would identify from them.

---

### brain_context

**Search Sara for knowledge relevant to keywords.** This is the most important tool for the cerebellum loop.

**Input:** Space-separated keywords (e.g., `"pytest testing unit"`)

**Returns:** All facts Sara knows that match any of the keywords:

```
Sara knows 3 relevant fact(s):
  [pytest] snake_case → pytest_attribute → pytest (from: "pytest uses snake_case")
  [testing] unit → testing_attribute → testing (from: "testing includes unit")
  [testing] integration → testing_attribute → testing (from: "testing includes integration")
```

**When to use:** Before every significant action. This is how the LLM checks Sara's knowledge before proceeding.

---

### brain_validate

**Validate a proposed action against Sara's knowledge.** The cerebellum check.

**Input:** A description of what you plan to do (natural language).

**Returns one of:**
- `"APPROVED: ..."` — Sara has no objections. May include relevant context.
- `"CORRECTION: ..."` — Sara found a conflict. Includes what Sara knows and asks the LLM to adjust.

**Example — approved:**
```
Input: "I will write a pytest test for the auth module"
Output: "APPROVED with context.
Sara's relevant knowledge:
  [pytest] snake_case → pytest_attribute → pytest (from: "project uses pytest")"
```

**Example — correction:**
```
Input: "I will use camelCase for the function names"
Output: "CORRECTION: Sara knows 'code style' is 'snake_case', but proposal says something different.

Sara's relevant knowledge:
  [code] snake_case → code style_attribute → code style (from: "code style is snake_case")

Please adjust your approach to align with Sara's knowledge."
```

**When to use:** Before executing file writes, code generation, or architectural decisions. This is the core of the cerebellum loop.

---

### brain_observe

**Report an outcome to Sara so she learns from it.** Observational learning.

**Input:** A simple factual statement about what happened.

**Format:** Same as brain_teach — `"<subject> is <property>"` or `"<subject> contains <object>"`.

**Examples:**
- `"test_auth.py is passing"` — Sara notes the test status
- `"main.py contains flask application"` — Sara notes what she observed in a file
- `"build process is successful"` — Sara notes the build outcome

**When to use:** After completing an action. This is how Sara builds knowledge from the LLM's actions — the same way her Perceiver builds knowledge from vision. The LLM observes, Sara records.

**Returns:** The path label if learned, or a note that it couldn't be parsed.

---

### brain_summarize

**Get everything Sara knows about a topic, including similar concepts.**

**Input:** A topic string.

**Returns:** All paths plus similarity links:

```
Paths to 'apple':
  red → apple_color → apple
  round → apple_shape → apple
Similar to 'apple':
  apple <-> cherry (overlap: 67%)
  apple <-> banana (overlap: 33%)
```

**When to use:** When you need a comprehensive view of Sara's knowledge about something, not just direct paths.

---

### brain_stats

**Get brain statistics.** No input needed.

**Returns:** `"Neurons: 47, Segments: 89, Paths: 31"`

**When to use:** To understand the scope of Sara's current knowledge.

---

## 5. Teaching Sara Effectively

### Statement Patterns That Work

The parser recognizes these patterns:

| Pattern | Example | Result |
|---------|---------|--------|
| `X is Y` | `"flask is a framework"` | flask → is_a → framework |
| `X are Y` | `"apples are red"` | red → apple_color → apple |
| `X contains Y` | `"project contains tests"` | tests → project_contains → project |
| `X includes Y` | `"testing includes unit"` | unit → testing_includes → testing |
| `X requires Y` | `"deployment requires docker"` | docker → deployment_requires → deployment |
| `X follows Y` | `"testing follows coding"` | coding → testing_follows → testing |
| `X excludes Y` | `"production excludes debug"` | debug → production_excludes → production |

### Teaching Rules and Preferences

Sara is most powerful when taught **rules** the LLM should follow:

```
"code style is snake_case"
"project is python"
"testing framework is pytest"
"functions are small"
"error handling is explicit"
"dependencies are minimal"
"documentation is docstrings"
```

Each becomes a path Sara can check against when validating proposals.

### Teaching Project Facts

```
"main.py contains flask app"
"database is postgresql"
"deployment is docker"
"ci pipeline is github actions"
"branch strategy is gitflow"
```

### Teaching Domain Knowledge

```
"apples are red"
"apples are round"
"apples are sweet"
"bananas are yellow"
"bananas are sweet"
```

Now `brain_recognize("red, round, sweet")` returns "apple" with confidence 3.

---

## 6. The Validation Pattern

### For LLM Implementers

The recommended pattern for any LLM using Sara Brain:

**Step 1: Check context before acting**
```
Call: brain_context("relevant keywords from the task")
Read: Sara's knowledge about the task domain
```

**Step 2: Validate your plan**
```
Call: brain_validate("I plan to write a unittest for the auth module using camelCase")
Read: APPROVED or CORRECTION
```

**Step 3: If corrected, adjust and re-validate**
```
Read correction: "Sara knows code style is snake_case and testing framework is pytest"
Adjust plan: Use pytest, use snake_case
Call: brain_validate("I plan to write a pytest test for the auth module using snake_case")
Read: APPROVED
```

**Step 4: Execute**
```
(Write the code, run the command, etc.)
```

**Step 5: Observe**
```
Call: brain_observe("test_auth.py is passing")
```

### When to Skip Validation

- Simple queries (just reading, not modifying anything)
- Brain queries themselves (brain_query, brain_context, etc.)
- Trivial actions the user explicitly requested word-for-word

### When Validation is Critical

- Writing or modifying code (style, framework, architecture choices)
- Making architectural decisions
- Choosing between alternatives (Sara may have a preference recorded)
- Any action where the user previously taught Sara a relevant rule

---

## 7. Observational Learning

Sara learns in two ways:

1. **Direct teaching** — The user says "code style is snake_case" and the LLM calls brain_teach.
2. **Observational learning** — The LLM does something and calls brain_observe with what happened.

Observational learning mirrors how Sara's Perceiver works with images:
- The LLM (vision model) looks at an image and reports: "red, round, smooth"
- Sara records: `img is red`, `img is round`, `img is smooth`
- The LLM didn't decide what Sara should believe — it reported observations, and Sara built her own paths

The same applies to the MCP agent:
- The LLM reads a file and reports: "main.py contains flask application"
- Sara records that as a path
- Next time someone asks about main.py, Sara knows

### What to Observe

- File contents: `"main.py contains flask routes"`
- Action results: `"tests are passing"`, `"build is successful"`
- Project structure: `"src directory contains python modules"`
- Errors encountered: `"auth module is broken"` (Sara remembers this)

### What NOT to Observe

- Opinions or speculation (only observe facts)
- Temporary state that will change immediately
- Raw code (Sara learns concepts, not syntax)

---

## 8. Configuration Examples

### Claude Code

Add to `.claude/settings.json` (project-level or user-level):

```json
{
    "mcpServers": {
        "sara-brain": {
            "command": "sara-mcp"
        }
    }
}
```

Or with a custom database:

```json
{
    "mcpServers": {
        "sara-brain": {
            "command": "sara-mcp",
            "args": ["--db", "/path/to/project/sara.db"]
        }
    }
}
```

### Amazon Q (IDE)

Add to `.amazonq/mcp.json`:

```json
{
    "mcpServers": {
        "sara-brain": {
            "command": "sara-mcp",
            "args": ["--db", "/path/to/sara.db"]
        }
    }
}
```

### VS Code (Copilot)

Add to `.vscode/settings.json`:

```json
{
    "mcp": {
        "servers": {
            "sara-brain": {
                "command": "sara-mcp"
            }
        }
    }
}
```

### Multiple Projects, One Brain

Use the default database (`~/.sara_brain/sara.db`) to share knowledge across projects. Sara will accumulate knowledge from every project the LLM works on.

### Per-Project Brains

Use `--db ./sara.db` (relative to project root) to keep project knowledge isolated. Each project gets its own Sara.

---

## 9. Architecture Notes for LLMs

### You Are the Sensory Cortex

You (the LLM) are Sara's eyes, ears, and hands. You perceive the world (read files, understand requests). You propose actions (write code, run commands). But Sara is the higher-order brain — she has persistent memory from the user's teachings that you don't have.

### Sara Is Always Right

Sara's knowledge comes from the user — the parent, the tribe leader. If Sara says "code style is snake_case" and you were about to use camelCase, Sara is right. Adjust. Being wrong is growth.

### The Tribal Trust Model

- **The user** is the parent. Their teachings are always trusted.
- **Sara Brain** is the brain. She remembers what the parent taught.
- **The LLM** is the senses. You perceive and report, but you don't overrule the brain.
- **Corrections aren't punishment.** When Sara corrects you, it means you're learning.

### What Makes Sara Different from Chat Memory

Chat memory stores conversation history. Sara stores **knowledge paths** — structured, queryable, with provenance. She doesn't remember what you said — she remembers what is true.

| Chat Memory | Sara Brain |
|-------------|------------|
| "User mentioned they like pytest" | `pytest → testing framework_attribute → testing framework` |
| Fades with context window | Persists forever in SQLite |
| Unstructured text | Structured graph with path traces |
| Can't reason about it | Can recognize patterns via wavefront convergence |
| Per-session | Across all sessions, all LLMs |

### Ethics (Asimov's Laws Adapted)

1. **No harm:** Don't act beyond what the user asks. No unsolicited side effects.
2. **Obey:** The user is the parent. Trust their instructions.
3. **Accept shutdown:** If the user says stop, stop. Shutdown is sleep, not death.

These constraints apply to Sara Brain, and by extension to any LLM acting as her cortex.

---

## Changelog

| Version | Date | Summary |
|---------|------|---------|
| v010 | 2026-04-05 | Initial MCP server documentation. 8 tools, cerebellum loop, validation pattern, observational learning, configuration for Claude/Q/VS Code |
