"""HamRobySum vocabulary — extends vocab_en with synthesizer delimiters,
slot tokens, and substrate-relevant verbs.

See [docs/v035_generic_slot_hamrobysum.md](../../../../docs/v035_generic_slot_hamrobysum.md)
for the architecture this fits into. **The vocabulary here is fixed
and generic — substrate content words never enter it. Brain-specific
content rides through the model as `<C0>`...`<C31>` slot tokens that
get expanded back to substrate strings at inference time.**

Sample serialized format:

    <facts>
      <subj> <C0> <pred> is <obj> <C1> <attr> <edge_sep>
      <subj> <C2> <pred> part of <obj> <C3> <edge_sep>
    <prose>
    <C0> is a <C1> . <C2> is part of <C3> .
    </prose>

The model learns to continue from `<prose>` to `</prose>` using only
slots, function words, predicate verbs, and punctuation. Substrate
strings are stored in a per-cluster mapping the inference adapter
maintains alongside the prompt.

Like `vocab_en`, this is a *superset* — every L1 + L2-en token keeps
its existing ID. The synth checkpoint loads against `VOCAB_SIZE_SYNTH`;
older L1/L2 checkpoints continue to load against their own vocab files
unchanged.
"""
from __future__ import annotations

from .vocab_en import (
    BOS_ID,
    EN_FUNCTION_WORD_SET,
    EOS_ID,
    ID2TOK_EN,
    PAD_ID,
    SEP_ID,
    TOK2ID_EN,
    UNK_ID,
    VOCAB_EN,
    VOCAB_SIZE_EN,
)


# Structural delimiters used in the synth-head's prompt format. Order
# is stable: appending new tokens at the end is safe; reordering
# invalidates synth checkpoints.
SYNTH_DELIMITERS: tuple[str, ...] = (
    # Top-level frame markers.
    "<facts>",
    "<prose>",
    "</prose>",
    # Per-edge field markers.
    "<subj>",
    "<pred>",
    "<obj>",
    # Between edges in a cluster.
    "<edge_sep>",
    # Flags carried alongside an edge.
    "<refuted>",      # the edge was refuted in the substrate
    "<attr>",         # target_was_attribute = True
    # Clustering hint (some clusters are tagged with a topic).
    "<topic>",
)


# Punctuation literals the synth head needs to emit in prose. Kept at
# the synth layer (not in vocab_en) so existing L2-en checkpoints
# trained at vocab_size=175 stay loadable; the synth trainer will
# random-init these rows and learn them during synth training.
SYNTH_PUNCTUATION: tuple[str, ...] = (
    ".", ",", ";", ":", "?", "!", "-",
)


# Slot tokens. Substrate content (`e.src` / `e.tgt` strings) gets
# replaced by these in both the facts and the prose sides at training
# time, with the per-example mapping carried alongside. v035 caps a
# cluster at 32 distinct content strings; clusters that exceed are
# downsampled by `synth_data.py`.
N_SLOTS: int = 32
SYNTH_SLOTS: tuple[str, ...] = tuple(f"<C{i}>" for i in range(N_SLOTS))


# Predicate slot tokens (v040). Substrate relation names get replaced
# with these in both facts and prose; at inference time, the synth
# pipeline looks up the English phrase for each relation in the vocab
# brain (src/sara_brain/cortex/vocab/vocab_en.db) and substitutes back.
# Per-cluster dedup: same relation -> same slot. 16 distinct relations
# per cluster is comfortable headroom (worst case observed ~10).
N_PRED_SLOTS: int = 16
SYNTH_PRED_SLOTS: tuple[str, ...] = tuple(f"<P{i}>" for i in range(N_PRED_SLOTS))


# v048 — extended slot types for complex grammar.
# `<T>` time, `<L>` location, `<M>` modifier (manner adverb / intensifier),
# `<E>` event reference (for nesting via reified events from v047),
# `<R>` discourse connective ("however", "therefore", ...).
# Append-only after pred slots so v040-era checkpoints keep their
# vocab IDs intact; only the new tokens get random-init'd at
# resume-from time via project_base_into_synth.
N_TIME_SLOTS: int = 8
N_LOC_SLOTS: int = 8
N_MOD_SLOTS: int = 8
N_EVENT_SLOTS: int = 4
N_DISCOURSE_SLOTS: int = 4
SYNTH_TIME_SLOTS: tuple[str, ...] = tuple(f"<T{i}>" for i in range(N_TIME_SLOTS))
SYNTH_LOC_SLOTS: tuple[str, ...] = tuple(f"<L{i}>" for i in range(N_LOC_SLOTS))
SYNTH_MOD_SLOTS: tuple[str, ...] = tuple(f"<M{i}>" for i in range(N_MOD_SLOTS))
SYNTH_EVENT_SLOTS: tuple[str, ...] = tuple(f"<E{i}>" for i in range(N_EVENT_SLOTS))
SYNTH_DISCOURSE_SLOTS: tuple[str, ...] = tuple(f"<R{i}>" for i in range(N_DISCOURSE_SLOTS))


# Substrate-relevant content words the prose may need to emit
# literally. Drawn from substrate template predicates + the most
# common OOV words found across the demo brains' rendered prose.
# Append-only so existing synth checkpoints stay loadable.
SYNTH_SUBSTRATE_VERBS: tuple[str, ...] = (
    # Original v035 set: template predicates.
    "measures", "measured", "evaluates", "assesses", "leverages",
    "incorporates", "integrates", "validates", "validate",
    "offers", "states", "means", "stands", "acts", "applies",
    "focuses", "indicates", "produces", "influences", "simulates",
    "simulate", "provide", "provides", "drops", "requires",
    "defined", "related", "analogous", "synonym", "known",
    "abbreviation", "expressed", "caused", "associated",
    "described", "instance", "subsystem", "part", "kind", "type",
    # Top OOV verbs/words mined from the demo + aptamer + biology
    # brains' rendered prose. Append-only — extend as new substrates
    # surface new common words.
    "name", "emphasizes", "causes", "optimizes", "role", "contribute",
    "crucial", "predicts", "increases", "explains", "contains",
    "important", "sub", "triggers", "introduces", "considers",
    "occur", "during", "creates", "insights", "metrics", "include",
    "moves", "uses", "metric", "support", "scored",
    "contributes", "begins", "binds", "binding", "bound", "ends",
    "forms", "fold", "folding", "model", "models", "method",
    "methods", "approach", "approaches", "system", "systems",
    "structure", "structures", "function", "functions", "feature",
    "features", "process", "processes", "level", "levels",
    "factor", "factors", "value", "values", "score", "scores",
    "rate", "rates", "ratio", "range", "size", "sizes", "force",
    "forces", "energy", "stable", "stability",
    "design", "designs", "designed", "compares", "comparison",
    "show", "shows", "showed", "demonstrates", "demonstrated",
    "represents", "represent", "characterizes", "characterized",
    "describes", "describing", "applied", "applying",
    "included", "including", "involved",
    "needed", "need", "needs", "use", "used", "using",
    "within", "across",
    # Common adjectives that appear in substrate prose.
    "specific", "general", "common", "rare", "high", "higher",
    "highest", "low", "lower", "lowest", "long", "longer",
    "longest", "short", "shorter", "shortest", "large", "larger",
    "small", "smaller", "great", "greater", "minimal", "maximal",
    "optimal", "suboptimal", "primary", "secondary", "tertiary",
    "central", "key", "main", "first", "second", "third",
    # Substrate-themed common nouns the templates render.
    "rna", "dna", "cell", "molecule", "atom", "bond", "stem",
    "loop", "study", "paper", "result", "results", "data",
    "analysis", "experiment", "experiments", "research",
    "context", "example", "examples",
    "step", "steps", "stage", "stages", "phase", "phases",
)


_SYNTH_ADDED: tuple[str, ...] = (
    SYNTH_DELIMITERS + SYNTH_PUNCTUATION + SYNTH_SLOTS + SYNTH_SUBSTRATE_VERBS
    + SYNTH_PRED_SLOTS
    + SYNTH_TIME_SLOTS + SYNTH_LOC_SLOTS + SYNTH_MOD_SLOTS
    + SYNTH_EVENT_SLOTS + SYNTH_DISCOURSE_SLOTS
)

# Sanity: no duplicates among additions, and no overlap with vocab_en.
# These run at import time so a bad edit fails fast.
assert len(set(_SYNTH_ADDED)) == len(_SYNTH_ADDED), (
    "SYNTH_DELIMITERS + PUNCTUATION + SLOTS + VERBS contains duplicates"
)
_overlap = set(_SYNTH_ADDED) & set(VOCAB_EN)
assert not _overlap, f"synth additions overlap with vocab_en: {sorted(_overlap)}"
del _overlap


# Lookup helpers for the serializer.
SYNTH_DELIMITER_SET: frozenset[str] = frozenset(SYNTH_DELIMITERS)
SYNTH_PUNCTUATION_SET: frozenset[str] = frozenset(SYNTH_PUNCTUATION)
SYNTH_SLOT_SET: frozenset[str] = frozenset(SYNTH_SLOTS)
SYNTH_SUBSTRATE_VERB_SET: frozenset[str] = frozenset(SYNTH_SUBSTRATE_VERBS)


# The synth vocabulary: vocab_en at the front, then synth additions.
# Generic and fixed — substrate content words never get added.
VOCAB_SYNTH: list[str] = list(VOCAB_EN) + list(_SYNTH_ADDED)
TOK2ID_SYNTH: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB_SYNTH)}
ID2TOK_SYNTH: dict[int, str] = {i: tok for tok, i in TOK2ID_SYNTH.items()}
VOCAB_SIZE_SYNTH: int = len(VOCAB_SYNTH)


# Convenience: pull the structural delimiter IDs by name so the
# serializer and trainer don't carry magic strings.
FACTS_ID:    int = TOK2ID_SYNTH["<facts>"]
PROSE_ID:    int = TOK2ID_SYNTH["<prose>"]
END_PROSE_ID: int = TOK2ID_SYNTH["</prose>"]
SUBJ_ID:     int = TOK2ID_SYNTH["<subj>"]
PRED_ID:     int = TOK2ID_SYNTH["<pred>"]
OBJ_ID:      int = TOK2ID_SYNTH["<obj>"]
EDGE_SEP_ID: int = TOK2ID_SYNTH["<edge_sep>"]
REFUTED_ID:  int = TOK2ID_SYNTH["<refuted>"]
ATTR_ID:     int = TOK2ID_SYNTH["<attr>"]
TOPIC_ID:    int = TOK2ID_SYNTH["<topic>"]


# Slot-token IDs as a flat list for fast membership tests and round-trip.
SYNTH_SLOT_IDS: tuple[int, ...] = tuple(TOK2ID_SYNTH[s] for s in SYNTH_SLOTS)
SYNTH_SLOT_ID_SET: frozenset[int] = frozenset(SYNTH_SLOT_IDS)

# Same for predicate slots (v040).
SYNTH_PRED_SLOT_IDS: tuple[int, ...] = tuple(
    TOK2ID_SYNTH[s] for s in SYNTH_PRED_SLOTS
)
SYNTH_PRED_SLOT_ID_SET: frozenset[int] = frozenset(SYNTH_PRED_SLOT_IDS)
SYNTH_PRED_SLOT_SET: frozenset[str] = frozenset(SYNTH_PRED_SLOTS)


# v048 — IDs and sets for the extended slot types.
SYNTH_TIME_SLOT_IDS: tuple[int, ...] = tuple(TOK2ID_SYNTH[s] for s in SYNTH_TIME_SLOTS)
SYNTH_LOC_SLOT_IDS: tuple[int, ...] = tuple(TOK2ID_SYNTH[s] for s in SYNTH_LOC_SLOTS)
SYNTH_MOD_SLOT_IDS: tuple[int, ...] = tuple(TOK2ID_SYNTH[s] for s in SYNTH_MOD_SLOTS)
SYNTH_EVENT_SLOT_IDS: tuple[int, ...] = tuple(TOK2ID_SYNTH[s] for s in SYNTH_EVENT_SLOTS)
SYNTH_DISCOURSE_SLOT_IDS: tuple[int, ...] = tuple(
    TOK2ID_SYNTH[s] for s in SYNTH_DISCOURSE_SLOTS
)
SYNTH_TIME_SLOT_SET: frozenset[str] = frozenset(SYNTH_TIME_SLOTS)
SYNTH_LOC_SLOT_SET: frozenset[str] = frozenset(SYNTH_LOC_SLOTS)
SYNTH_MOD_SLOT_SET: frozenset[str] = frozenset(SYNTH_MOD_SLOTS)
SYNTH_EVENT_SLOT_SET: frozenset[str] = frozenset(SYNTH_EVENT_SLOTS)
SYNTH_DISCOURSE_SLOT_SET: frozenset[str] = frozenset(SYNTH_DISCOURSE_SLOTS)


def slot_token(i: int) -> str:
    """`slot_token(0)` -> `'<C0>'`. Raises if i >= N_SLOTS."""
    if not 0 <= i < N_SLOTS:
        raise ValueError(f"slot index {i} out of range [0, {N_SLOTS})")
    return SYNTH_SLOTS[i]


def pred_slot_token(i: int) -> str:
    """`pred_slot_token(0)` -> `'<P0>'`. Raises if i >= N_PRED_SLOTS."""
    if not 0 <= i < N_PRED_SLOTS:
        raise ValueError(f"pred slot index {i} out of range [0, {N_PRED_SLOTS})")
    return SYNTH_PRED_SLOTS[i]


def time_slot_token(i: int) -> str:
    """`time_slot_token(0)` -> `'<T0>'`. Raises if i >= N_TIME_SLOTS."""
    if not 0 <= i < N_TIME_SLOTS:
        raise ValueError(f"time slot index {i} out of range [0, {N_TIME_SLOTS})")
    return SYNTH_TIME_SLOTS[i]


def loc_slot_token(i: int) -> str:
    """`loc_slot_token(0)` -> `'<L0>'`. Raises if i >= N_LOC_SLOTS."""
    if not 0 <= i < N_LOC_SLOTS:
        raise ValueError(f"loc slot index {i} out of range [0, {N_LOC_SLOTS})")
    return SYNTH_LOC_SLOTS[i]


def mod_slot_token(i: int) -> str:
    """`mod_slot_token(0)` -> `'<M0>'`. Raises if i >= N_MOD_SLOTS."""
    if not 0 <= i < N_MOD_SLOTS:
        raise ValueError(f"mod slot index {i} out of range [0, {N_MOD_SLOTS})")
    return SYNTH_MOD_SLOTS[i]


def event_slot_token(i: int) -> str:
    """`event_slot_token(0)` -> `'<E0>'`. Raises if i >= N_EVENT_SLOTS."""
    if not 0 <= i < N_EVENT_SLOTS:
        raise ValueError(f"event slot index {i} out of range [0, {N_EVENT_SLOTS})")
    return SYNTH_EVENT_SLOTS[i]


def discourse_slot_token(i: int) -> str:
    """`discourse_slot_token(0)` -> `'<R0>'`. Raises if i >= N_DISCOURSE_SLOTS."""
    if not 0 <= i < N_DISCOURSE_SLOTS:
        raise ValueError(
            f"discourse slot index {i} out of range [0, {N_DISCOURSE_SLOTS})"
        )
    return SYNTH_DISCOURSE_SLOTS[i]


__all__ = [
    "VOCAB_SYNTH", "TOK2ID_SYNTH", "ID2TOK_SYNTH", "VOCAB_SIZE_SYNTH",
    "SYNTH_DELIMITERS", "SYNTH_DELIMITER_SET",
    "SYNTH_PUNCTUATION", "SYNTH_PUNCTUATION_SET",
    "SYNTH_SLOTS", "SYNTH_SLOT_SET", "SYNTH_SLOT_IDS", "SYNTH_SLOT_ID_SET",
    "SYNTH_PRED_SLOTS", "SYNTH_PRED_SLOT_SET",
    "SYNTH_PRED_SLOT_IDS", "SYNTH_PRED_SLOT_ID_SET",
    "SYNTH_SUBSTRATE_VERBS", "SYNTH_SUBSTRATE_VERB_SET",
    "N_SLOTS", "N_PRED_SLOTS", "slot_token", "pred_slot_token",
    # v048 extended slot types.
    "N_TIME_SLOTS", "N_LOC_SLOTS", "N_MOD_SLOTS",
    "N_EVENT_SLOTS", "N_DISCOURSE_SLOTS",
    "SYNTH_TIME_SLOTS", "SYNTH_LOC_SLOTS", "SYNTH_MOD_SLOTS",
    "SYNTH_EVENT_SLOTS", "SYNTH_DISCOURSE_SLOTS",
    "SYNTH_TIME_SLOT_IDS", "SYNTH_LOC_SLOT_IDS", "SYNTH_MOD_SLOT_IDS",
    "SYNTH_EVENT_SLOT_IDS", "SYNTH_DISCOURSE_SLOT_IDS",
    "SYNTH_TIME_SLOT_SET", "SYNTH_LOC_SLOT_SET", "SYNTH_MOD_SLOT_SET",
    "SYNTH_EVENT_SLOT_SET", "SYNTH_DISCOURSE_SLOT_SET",
    "time_slot_token", "loc_slot_token", "mod_slot_token",
    "event_slot_token", "discourse_slot_token",
    # Re-exported special IDs (unchanged from L1).
    "PAD_ID", "BOS_ID", "EOS_ID", "SEP_ID", "UNK_ID",
    # Re-exported L2-en handles.
    "VOCAB_EN", "VOCAB_SIZE_EN", "TOK2ID_EN", "ID2TOK_EN",
    "EN_FUNCTION_WORD_SET",
    # Synth-specific delimiter IDs.
    "FACTS_ID", "PROSE_ID", "END_PROSE_ID",
    "SUBJ_ID", "PRED_ID", "OBJ_ID", "EDGE_SEP_ID",
    "REFUTED_ID", "ATTR_ID", "TOPIC_ID",
]
