# v010 — Sara Care: A Dementia Assistance Proof-of-Concept

## What It Is

Sara Care is a proof-of-concept tool built on Sara Brain that lets a person with dementia sit down at their personal computer — with no internet connection — and ask:

- "What happened yesterday?"
- "Who am I?"
- "Who are these people in my house?"

And get real answers.

Sara Care extends Sara Brain with three user roles, a trust hierarchy, temporal awareness, and an interaction log. It runs entirely locally using Ollama for the LLM sensory layer. No data leaves the machine.

## The Problem

A person with dementia loses the ability to hold new memories. They wake up confused. They don't recognize family members. They can't remember what happened an hour ago. Current assistive technology is either:

- **Too clinical** — designed for caregivers, not the patient
- **Too passive** — the patient is a subject being monitored, not a participant
- **Cloud-dependent** — sensitive personal and medical data flows through external servers

Sara Care treats the patient as a **participant**, not a subject. They talk to Sara. They tell Sara things. They ask Sara questions. And Sara never forgets.

## Architecture

### Three Roles

| Role | Description | Trust Level | What They Can Do |
|------|-------------|-------------|------------------|
| **Reader** (patient) | The person Sara helps | `observed` | Ask questions, tell Sara things, teach (into protected space) |
| **Teacher** (family) | Family members | `taught` | Teach facts, review patient's day, see what patient asked |
| **Doctor** (caregiver) | Medical professionals | `verified` | Everything above + annotate observations, analyze trends |

### Trust Hierarchy

Sara Care extends Sara Brain's tribal trust model with a formal knowledge trust system:

**`observed`** — The patient said it. It enters a protected observation space. Sara uses it when answering the patient, but it's flagged for review by teachers and doctors. The patient is NOT ignored — they are listened to and believed by Sara.

**`taught`** — A family member said it. It enters general knowledge. But family can lie or be mistaken. If a patient observation conflicts with a family teaching, both are flagged.

**`verified`** — A doctor said it or approved it. Highest trust level.

**`contested`** — The patient and family disagree. Flagged for doctor review.

**Repetition = Signal.** If the patient tells Sara the same thing three times, the repetition count is tracked. This is not confusion to be dismissed — it's a signal that the patient believes this strongly and it needs investigation.

### Nothing Is Ever Erased

This is the core principle inherited from Sara Brain: **no path is ever deleted or pruned.**

When a doctor reviews a patient observation, they do not delete it. They **annotate** it with a clinical label:

- `misunderstood` — patient was confused at the time
- `confabulation` — patient believed something that didn't happen
- `accurate` — patient was right (promotes to verified)
- `disputed` — still under investigation

The annotation is itself a new path in Sara's knowledge graph:

```
original observation → doctor_assessed → misunderstood
```

The original observation stays forever. If the same concern comes up again a year later, Sara has the full history: the original observation, how many times it was repeated, what the doctor assessed, and when.

This matters because patterns that look like confusion today may turn out to be meaningful later. A patient repeatedly saying "that man is not my son" might be confused — or might be right. The history is preserved for longitudinal analysis.

### Time as Neurons

Sara Brain originally had no concept of time. Sara Care introduces **temporal neurons** — date and time-of-day concepts that live in the knowledge graph as first-class neurons.

When the patient tells Sara "I had lunch with James," Sara creates:

```
lunch → happened_on → day_2026_04_06
lunch → happened_during → afternoon
```

When someone asks "what happened yesterday?", Sara resolves "yesterday" to a date neuron label (`day_2026_04_05`) and runs a standard wavefront propagation from that neuron. Time queries use the same path-of-thought mechanism as everything else — no special-case logic.

The temporal resolver understands:
- Relative days: "today", "yesterday", "day before yesterday"
- Weekdays: "last monday", "tuesday"
- Time of day: "this morning", "last night"

### Interaction Log

Every exchange with Sara is recorded in an append-only `interactions` table:

| Field | Description |
|-------|-------------|
| `account_id` | Who said it |
| `interaction_type` | `tell`, `ask`, `teach`, or `review` |
| `content` | What they said |
| `response` | What Sara said back |
| `path_ids` | Knowledge paths created from this interaction |
| `created_at` | Timestamp |

This log enables:

- **"How did things go?"** — A family member asks Sara about the patient's day. Sara queries the interaction log for the patient's activity and summarizes: what they told her, what they asked about, what seemed confusing or emotional.
- **Trend analysis** — A doctor asks about patterns over time: "What has Margaret been confused about this week?" Sara aggregates interaction data to identify clusters of repeated questions or confusion topics.
- **Longitudinal tracking** — Over months, the interaction log becomes a record of the patient's cognitive state — which topics cause confusion, whether question frequency is increasing, and what the patient remembers vs. forgets.

## How It Works

### For the Patient

```
$ sara-care

  Who am I speaking with?
  1. Margaret (that's me!)
  2. James (family)
  3. Dr. Chen (doctor)

  > 1

  Hello, Margaret. It's good to talk to you.
  Just tell me what's on your mind, or ask me anything.

margaret> who are the people in my house?

sara> James is your son. He visits you often. He was here yesterday too.
      Dr. Chen is your doctor. She comes by on Tuesdays.

margaret> I had lunch with a nice man today

sara> That sounds lovely! Do you remember his name?

margaret> I think he said he was James

sara> That's right — James is your son. He loves having lunch with you.
```

Sara is warm, patient, direct. She uses short sentences and simple words. She never says "I don't know" about people — she guides the patient to tell her more. She never uses clinical language. She never reveals that others review the conversations. She is a friend who remembers everything.

### For Family

```
james> how did things go today?

sara> Margaret had a good day. Here's a summary:

      Things she shared:
      - She had lunch with "a nice man" — she figured out it was you (James)
      - She mentioned the garden looked pretty

      Questions she asked:
      - "Who are the people in my house?" (she asked this twice)

      Notes:
      - She didn't seem to remember your visit from yesterday
      - She was in a pleasant mood throughout
```

### For Doctors

```
dr. chen> show me contested observations

sara> 1 contested path:
      Patient says: "that man is not my son" (repeated 3 times)
      Family says: "James is Margaret's son"
      Status: contested — awaiting your review

dr. chen> annotate path 12 as misunderstood

sara> Annotated. Original observation preserved.
      Assessment: misunderstood (by Dr. Chen, 2026-04-06)
```

## Technical Implementation

### New Database Tables

**`accounts`** — Who uses this Sara instance
```sql
CREATE TABLE accounts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL,     -- 'reader', 'teacher', 'doctor'
    pin_hash    TEXT,              -- optional SHA-256 PIN
    neuron_id   INTEGER REFERENCES neurons(id),
    created_at  REAL,
    is_active   INTEGER NOT NULL DEFAULT 1
);
```

Each account links to a **concept neuron** in the graph. Sara literally knows the person — "Margaret is the patient", "James is Margaret's son". The account is access control; the neuron is knowledge.

**`interactions`** — Append-only exchange log
```sql
CREATE TABLE interactions (
    id               INTEGER PRIMARY KEY,
    account_id       INTEGER NOT NULL REFERENCES accounts(id),
    interaction_type TEXT NOT NULL,
    content          TEXT NOT NULL,
    response         TEXT,
    path_ids         TEXT,
    created_at       REAL NOT NULL
);
```

### Extended `paths` Table

Three new columns on the existing `paths` table:

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | INTEGER | Who created this path |
| `trust_status` | TEXT | `observed`, `taught`, `verified`, `contested` |
| `repetition_count` | INTEGER | How many times this fact was stated by this account |

Existing paths (pre-Sara Care) have `NULL` for these columns — fully backward-compatible.

### New Source Files

| File | Purpose |
|------|---------|
| `storage/account_repo.py` | Account CRUD, PIN verification |
| `storage/interaction_repo.py` | Interaction logging, time/type queries |
| `core/temporal.py` | Date neurons, time-of-day periods, temporal resolver |
| `core/trust.py` | Trust management, annotations, conflict detection, repetition tracking |
| `care/__init__.py` | Package marker |
| `care/cli.py` | `sara-care` entry point |
| `care/loop.py` | Role-aware agent loop with interaction logging |
| `care/system_prompts.py` | Per-role system prompts |
| `care/accounts.py` | Account setup wizard, account selection |

### Modified Source Files

| File | Change |
|------|--------|
| `models/path.py` | Added `account_id`, `trust_status`, `repetition_count` |
| `storage/path_repo.py` | Stores/retrieves new fields, trust-based queries |
| `storage/schema.sql` | New tables, new indexes, extended paths |
| `storage/database.py` | Migration logic for existing databases |
| `storage/__init__.py` | Exports new repos |
| `core/learner.py` | Passes through `account_id`, `trust_status` |
| `core/brain.py` | Auto-sets trust based on role, exposes account/interaction repos |
| `pyproject.toml` | `sara-care` entry point |

## Design Principles

### The Patient Is the Most Important User

Sara Care is designed around the patient's dignity and agency. They are not a subject being monitored — they are a person having a conversation with someone who cares about them and remembers everything.

Design choices that reflect this:

- The patient can teach Sara things. Their observations matter.
- Sara never uses clinical language with the patient.
- Sara never reveals that conversations are reviewed by others.
- Sara never says "you already told me that" when the patient repeats themselves.
- The patient's role is called "reader" — not "patient" — in the interface.

### The LLM Is the Senses, Not the Brain

Ollama runs locally and provides the conversational interface — translating natural language to structured facts and back. But the knowledge lives in Sara Brain's path-of-thought graph, not in the LLM's weights. This means:

- Knowledge persists across sessions, across model changes, across hardware upgrades
- No internet required — everything runs on the patient's personal computer
- No private data leaves the machine — ever
- The LLM can be swapped, upgraded, or changed without losing any knowledge

### Sara Never Forgets

Every observation, every teaching, every doctor annotation, every timestamp — it all stays in the graph forever. This isn't just a technical choice. For a dementia patient, Sara Brain is literally their external memory. Pruning or decaying that memory would be like taking their memories away a second time.

## Requirements

- Python 3.11+
- Ollama installed locally with a capable model (llama3.1 recommended)
- No internet connection required
- No external Python dependencies

## Usage

```bash
# First time setup
sara-care --setup

# Normal use
sara-care

# Custom database path
sara-care --db /path/to/margaret.db

# Specific model
sara-care --model llama3.1
```

## What's Next

This is a proof-of-concept. Future work includes:

- **Photo recognition** — "Who is this person?" using Sara's existing perception system
- **Routine tracking** — "It's 2pm, Margaret usually has tea now"
- **Voice interface** — Local speech-to-text and text-to-speech for patients who can't type
- **Simplified UI** — Large buttons, clear text, minimal cognitive load
- **Doctor trend dashboard** — Visual charts of confusion patterns over time
- **Integrated cortex** — Sara's own small language model, removing the Ollama dependency entirely

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v010 | 2026-04-06 | Initial Sara Care proof-of-concept: accounts, trust hierarchy, temporal neurons, interaction log, three-role system |
