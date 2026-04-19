# Archived Teaching Tools

These tools were superseded on 2026-04-18. They are retained (not deleted) because:

- The teaching-vs-training philosophy they explored is valid and published as a Zenodo preprint (see `reference_teaching_paper`).
- They may contain specific prompt engineering or pipeline ideas worth revisiting.
- `feedback_never_delete_branches` applies in spirit — we archive, we don't delete.

## Why archived

All four depend on an Ollama LLM (originally `qwen2.5-coder:3b`, which drifted from the agreed `mistral`) for fact extraction or cortex verification. The current direction (see `project_spacy_sensory_cortex`, `project_eyeball_cortex`) replaces LLM-based extraction with spaCy-based parsing + deliberate human-or-Claude-surrogate teaching.

## What replaces them

- Primitive teaching: `benchmarks/batch_teach.py` (extended with `--region` flag).
- Claude-surrogate teaching (per `feedback_claude_as_teacher_surrogate`): Claude reads source material, selects facts with judgment, and uses `batch_teach` to load them into the correct compartmentalized region.
- Future: a `study_spacy.py` that uses spaCy to extract SVO triples without any LLM, for truly unsupervised ingest. Not written yet.

## Contents

- `comprehension_teacher.py` — LLM-verified sentence-by-sentence teaching.
- `error_learning_loop.py` — iterative test → diagnose → teach → retest loop with LLM cortex.
- `study_tool.py` — two-pass memorize/recite/consolidate via LLM fact extraction.
- `study_compartmentalized.py` — section → region variant of `study_tool`, LLM-coupled.

## Do not restore without cause

If a future session wants to use one of these tools, stop and reconsider. The architectural direction is away from LLM-coupled teaching. The correct move is almost always to teach facts judgmentfully via `batch_teach` or to write a new spaCy-based ingest path, not to reactivate these.
