# v016 — Replace SQLite with data-nut-squirrel: Design Plan

**Date:** April 12, 2026
**Author:** Jennifer Pearl
**Status:** Design plan — not yet implemented. To be built on `squirrel_storage` branch.

---

## Context

Sara Brain currently stores neurons, segments, paths, and path_steps in SQLite via raw SQL queries in the storage/ layer. Jennifer's own framework `data_nut_squirrel` (https://github.com/LunarFawn/rna_squirrel) was always intended as the replacement — it provides:

- YAML schema definitions → generated strongly-typed Python API classes
- Hierarchical attribute access with type validation at runtime
- YAML file persistence (no binary database, human-readable)
- Transparent persistence via `__setattr__`/`__getattribute__` hooks

This moves Sara from "raw SQL queries against a binary file" to "typed Python attributes that auto-persist to readable YAML." The brain becomes inspectable with a text editor.

## New branch

`squirrel_storage` — branched from `signed_refutation_paths`

---

## Phase 1: YAML Schema for Sara's Data Model

Define `sara_brain_schema.yaml` following data-nut-squirrel's format:

```yaml
NUT:
  nut_main_struct:
    name: SaraBrain
    object_list:
      - name: neurons
        object_type: CONTAINER
        object_info: NeuronStore
      - name: segments
        object_type: CONTAINER
        object_info: SegmentStore
      - name: paths
        object_type: CONTAINER
        object_info: PathStore
      - name: settings
        object_type: CONTAINER
        object_info: SettingsStore

DEFINITIONS:
  nut_containers_definitions:
    - name: NeuronStore
      object_list:
        - name: neurons_by_id
          object_type: dict
          object_info: [NeuronRecord, CLASS]
        - name: neurons_by_label
          object_type: dict
          object_info: [str]

    - name: NeuronRecord
      object_list:
        - name: id
          object_type: VALUE
          object_info: int
        - name: label
          object_type: VALUE
          object_info: str
        - name: neuron_type
          object_type: VALUE
          object_info: str
        - name: created_at
          object_type: VALUE
          object_info: float
        - name: metadata
          object_type: VALUE
          object_info: str

    - name: SegmentStore
      object_list:
        - name: segments_by_id
          object_type: dict
          object_info: [SegmentRecord, CLASS]
        - name: outgoing_index
          object_type: dict
          object_info: [list]
        - name: incoming_index
          object_type: dict
          object_info: [list]

    - name: SegmentRecord
      object_list:
        - name: id
          object_type: VALUE
          object_info: int
        - name: source_id
          object_type: VALUE
          object_info: int
        - name: target_id
          object_type: VALUE
          object_info: int
        - name: relation
          object_type: VALUE
          object_info: str
        - name: strength
          object_type: VALUE
          object_info: float
        - name: traversals
          object_type: VALUE
          object_info: int
        - name: refutations
          object_type: VALUE
          object_info: int
        - name: created_at
          object_type: VALUE
          object_info: float
        - name: last_used
          object_type: VALUE
          object_info: float

    - name: PathStore
      object_list:
        - name: paths_by_id
          object_type: dict
          object_info: [PathRecord, CLASS]
        - name: paths_by_terminus
          object_type: dict
          object_info: [list]

    - name: PathRecord
      object_list:
        - name: id
          object_type: VALUE
          object_info: int
        - name: origin_id
          object_type: VALUE
          object_info: int
        - name: terminus_id
          object_type: VALUE
          object_info: int
        - name: source_text
          object_type: VALUE
          object_info: str
        - name: created_at
          object_type: VALUE
          object_info: float
        - name: steps
          object_type: list
          object_info: [int]
```

---

## Phase 2: Generate API + Write Adapter Layer

1. Run `make_data_squirrel_api` on the schema to generate the typed API class
2. Create `src/sara_brain/storage/squirrel_adapter.py` — an adapter that implements the same interface as the current repos (NeuronRepo, SegmentRepo, PathRepo) but backed by the generated squirrel API instead of SQL
3. The adapter maintains in-memory indexes (by_label, by_id, outgoing, incoming) that the squirrel persistence layer auto-saves

---

## Phase 3: Swap Storage Backend in Database class

Update `storage/database.py` to accept a backend parameter:

```python
class Database:
    def __init__(self, db_path, backend="sqlite"):
        if backend == "squirrel":
            self._init_squirrel(db_path)
        else:
            self._init_sqlite(db_path)
```

Both backends expose the same repo interfaces. Brain.py doesn't change — it talks to repos, not SQL.

---

## Phase 4: Migration Tool

`sara-migrate` CLI that reads an existing SQLite brain and writes it to squirrel YAML format. Lossless, round-trippable.

---

## Files to create

| File | Purpose |
|------|---------|
| `sara_brain_schema.yaml` | Schema definition for data-nut-squirrel |
| `src/sara_brain/storage/squirrel_adapter.py` | Adapter implementing repo interfaces over squirrel |
| `src/sara_brain/storage/squirrel_generated.py` | Generated API (output of make_data_squirrel_api) |
| Migration CLI entry point | `sara-migrate` command |

## Files to modify

| File | Change |
|------|--------|
| `src/sara_brain/storage/database.py` | Add backend selection (sqlite vs squirrel) |
| `pyproject.toml` | Add data-squirrel as optional dependency, add sara-migrate entry point |

## Files that DON'T change

- `src/sara_brain/core/brain.py` — talks to repos, not SQL
- `src/sara_brain/core/learner.py` — same
- `src/sara_brain/core/recognizer.py` — same
- `src/sara_brain/cortex/` — same
- All models (neuron.py, segment.py, path.py) — same
- All tests — should pass against both backends

---

## What this gives you

1. **Human-readable brain** — open the YAML in a text editor and read Sara's knowledge
2. **Type-safe persistence** — the generated API validates types at runtime
3. **Your own framework** — Sara runs on your code, not someone else's SQLite
4. **Portable** — YAML files copy between machines easier than SQLite binary
5. **Version-controllable** — you could even git-track a brain's YAML files
6. **Still works on a Pi** — YAML is lighter than SQLite for small brains

---

## Trade-offs

- **Performance**: YAML file I/O is slower than SQLite for large brains (1000+ neurons). For the Pi demo with a clean brain, this won't matter. For the polluted 1500-neuron brain, might be noticeable.
- **No ACID transactions**: YAML writes aren't atomic. A crash mid-write could corrupt. Mitigated by write-then-rename pattern.
- **No SQL queries**: Fuzzy resolver, edit distance, LIKE queries currently use SQL. These become Python-side operations against the in-memory indexes.
- **Dependency**: data-squirrel becomes a dependency (but it's Jennifer's own, so that's a feature not a bug).

---

## Verification

1. Create the new branch from signed_refutation_paths
2. Generate the API from the schema
3. Write the adapter with full repo interface
4. Run all 291 tests against the squirrel backend
5. Migrate the existing clean brain (sara_new.db) to YAML
6. Run sara-cortex against the YAML brain, verify same behavior
7. Open the YAML in a text editor and read the knowledge

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| v016 | 2026-04-12 | data-nut-squirrel storage replacement design plan |
| v015 | 2026-04-12 | Primitive layers mapped to brain structures. CLEANUP layer. JKD parallel. |
| v014 | 2026-04-12 | Sleep and consolidation design note |
| v013 | 2026-04-11 | Provenance gaming and long-term typo pollution |
| v012 | 2026-04-11 | Failure modes and the cortex |
