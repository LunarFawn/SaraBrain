# v042 — `/teach-vocab` REPL command

**Date:** 2026-05-04
**Branch:** `feature/grammar-cortex`
**Builds on:** [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md),
[v041_predicate_slots_impl.md](v041_predicate_slots_impl.md)

## Context

v040 made the vocab brain a first-class L3 substrate. Per v040: "a
user can `/teach-vocab consumes 'consumes'` from the chat REPL —
same machinery as `/teach`, just targeting the vocab brain instead
of the content brain." This implements that.

When the chat REPL is running with `--use-hamrobysum` (HamRoby-Sum
loaded + vocab brain loaded), the user can teach new relation →
English mappings live. The next `/something` query uses them.

## Scope

In, v0:

- New `/teach-vocab` slash command in `chat.py`.
- Syntax: `/teach-vocab RELATION PHRASE...`. Relation is one token
  (substrate convention, may contain underscores). Everything after
  is the English phrase (no quotes needed).
- Behavior: REPLACES any existing `english_form` segment(s) for
  that relation. Multi-form/style-variation support is future.
- Persists immediately to the vocab brain SQLite. Updates
  `self._vocab_lookup` in-memory so the next query uses the new
  mapping without restart.
- Error when `--use-hamrobysum` is not set: "vocab brain not
  loaded — restart with `--use-hamrobysum`".

Out, deferred:

- `/refute-vocab` (remove a mapping).
- `/list-vocab` (show all mappings).
- Multi-form / preferred-form selection.
- Per-domain / per-language vocab brain swap mid-session.

## Files

**Modified:** `src/sara_brain/cortex/transformer/chat.py` only.

**Reused functions:**
- `inference_synth.load_vocab_brain` — re-read on retrain or restart.
- The same SQLite schema as `scripts/build_vocab_brain_en.py` (the
  vocab brain is just a brain.db; teach-vocab uses raw SQLite,
  bypassing the chain-learning machinery — same pattern as the
  build script).

## Verification

End-to-end:

1. Launch chat with `--use-hamrobysum`.
2. Ask "what is X" where the substrate has a relation outside
   vocab_en.db (e.g. `forms` — falls back to "forms").
3. Run `/teach-vocab forms "is formed by"`.
4. Re-ask the same query. Output should now use "is formed by"
   instead of the fallback "forms".
5. `/teach-vocab is_a "represents"` — replace the pre-shipped
   "is a" with "represents". Re-query. Output uses "represents".
6. Restart chat. Vocab brain still has the user-taught mappings
   (persistence works).

Without `--use-hamrobysum`: `/teach-vocab anything` errors cleanly.
