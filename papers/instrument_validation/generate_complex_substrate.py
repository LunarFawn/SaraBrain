"""v048 — generate a complex-grammar synthetic corpus for HamRobySum-EN.

Where v036's `generate_synthetic_substrate.py` produced flat triplet
streams, this generator produces **scenes** — clusters of related
triplets sharing temporal and spatial frame, optionally chained via
discourse connectives. The model trained on these scenes learns
compound, complex, conditional, temporal, located, and modified
grammar without seeing any real-language content (every entity,
location, and time is a pronounceable nonsense word).

Output:
- `<out>.db` — Sara brain with the substrate edges flattened into
  binary triplets (events reified per v047 convention).
- `<out>.pairs.jsonl` — direct (edges, prose, subject) training
  pairs ready for `synth_data.serialize_example`. Bypasses the
  template-rendering path because complex grammar templates are
  defined here, not in `synthesizer.py`.
- `<out>.manifest.json` — canonical record for grading.

Usage:
    .venv/bin/python papers/instrument_validation/generate_complex_substrate.py \\
        --out /tmp/complex_substrate.db --scenes 800 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sara_brain.core.brain import Brain
from sara_brain.cortex.transformer.synthesizer import Edge
from sara_brain.cortex.transformer.vocab_synth import (
    ATTR_ID, BOS_ID, EDGE_SEP_ID, END_PROSE_ID, EOS_ID, FACTS_ID,
    OBJ_ID, PRED_ID, PROSE_ID, REFUTED_ID, SUBJ_ID, TOK2ID_SYNTH, UNK_ID,
    N_PRED_SLOTS, N_SLOTS,
    pred_slot_token, slot_token,
)


# ── Random nonsense word generation ──────────────────────────────────
_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"


def _random_word(rng: random.Random, min_len: int = 5, max_len: int = 8) -> str:
    """Pronounceable nonsense word."""
    length = rng.randint(min_len, max_len)
    word = []
    use_consonant = rng.random() < 0.5
    for _ in range(length):
        word.append(rng.choice(_CONSONANTS if use_consonant else _VOWELS))
        use_consonant = not use_consonant
    return "".join(word)


def _random_compound(rng: random.Random, n_words: int = 2) -> str:
    return " ".join(_random_word(rng) for _ in range(n_words))


# ── Verbs / modifiers / connectives ──────────────────────────────────
# These are real English so the rendered prose READS as English. The
# orthogonality property only needs to hold on the entity/location/
# time labels — substrate content is what the model must not memorise.
# The verbs / modifiers / connectives are STRUCTURAL parts of grammar.

_ACTION_VERBS = [
    "walked_to", "ran_to", "arrived_at", "left", "sat", "stood",
    "spoke_to", "saw", "heard", "remembered", "forgot",
    "opened", "closed", "carried", "found", "lost",
    "began", "finished", "waited_for",
    "examined", "watched", "joined", "followed", "approached",
    "greeted", "thanked", "answered", "asked",
]

_MANNER_MODIFIERS = [
    "quickly", "slowly", "carefully", "reluctantly", "eagerly",
    "calmly", "nervously", "quietly", "loudly", "gently",
    "suddenly", "finally", "again", "still",
]

_DISCOURSE_CONNECTIVES = [
    "however", "therefore", "meanwhile", "furthermore", "then",
    "afterwards", "later", "before_that",
]

# Time-frame templates — pick one and substitute a nonsense token in
# place of a specific date. These read as English temporal phrases
# but bind to nonsense substrate content for orthogonality.
_TIME_FRAME_PREFIXES = [
    "on", "at", "during", "after", "before",
]


# ── Scene schema ─────────────────────────────────────────────────────


@dataclass
class Scene:
    """A single scene = subject + action + object, optionally with
    location, time, and modifier. Renderable as one or more grammar
    forms (simple / temporal / located / modified)."""
    subject: str
    action: str
    object: str
    location: str | None = None
    time: str | None = None
    modifier: str | None = None
    event_id: str = ""    # the reified event node label

    def to_edges(self) -> list[Edge]:
        """Flatten into substrate edges. Main triple uses the action
        AS the relation so it slots as `<P>` (predicate) at training
        time. Qualifier fields (time / location / modifier) become
        separate triples on the same subject so the cluster contains
        all the info the complex-grammar template needs.

        At INFERENCE time, event reification (v047) lets a single
        event node bundle the multi-valued fact; at TRAINING time we
        flatten so the slot mechanism's predicate/content split
        cleanly maps onto subj-pred-obj-and-qualifiers."""
        edges: list[Edge] = []
        edges.append(Edge(src=self.subject, rel=self.action, tgt=self.object))
        if self.location:
            edges.append(Edge(src=self.subject, rel="at_location", tgt=self.location))
        if self.time:
            edges.append(Edge(src=self.subject, rel="at_time", tgt=self.time))
        if self.modifier:
            edges.append(Edge(src=self.subject, rel="in_manner", tgt=self.modifier))
        return edges


# ── Template renderers ───────────────────────────────────────────────
# Each template takes one or two scenes plus a slot-mapping function
# that converts substrate strings to slot tokens. Returns slotted prose.


SlotFn = "callable[[str], str]"   # maps substrate string -> slot token


def _slot_for(scene: Scene, slot: SlotFn) -> dict[str, str]:
    """Build a slot-token dict for scene fields. The `pred` field
    uses scene.action (the relation in the substrate edge) so it gets
    slotted as `<P>` at serialization time. Missing optional fields
    are omitted (caller-side templates only reference fields that
    exist)."""
    out = {
        "subj": slot(scene.subject),
        "obj": slot(scene.object),
        "pred": scene.action,   # relation name; serializer maps to <P>
    }
    if scene.location:
        out["loc"] = slot(scene.location)
    if scene.time:
        out["time"] = slot(scene.time)
    if scene.modifier:
        out["mod"] = slot(scene.modifier)
    return out


def t_simple(scene: Scene, slot: SlotFn) -> str:
    s = _slot_for(scene, slot)
    return f"{s['subj']} {s['pred']} {s['obj']} ."


def t_temporal_prefix(scene: Scene, slot: SlotFn) -> str | None:
    if not scene.time:
        return None
    s = _slot_for(scene, slot)
    return f"{s['time']} , {s['subj']} {s['pred']} {s['obj']} ."


def t_located_suffix(scene: Scene, slot: SlotFn) -> str | None:
    if not scene.location:
        return None
    s = _slot_for(scene, slot)
    return f"{s['subj']} {s['pred']} {s['obj']} at {s['loc']} ."


def t_modified(scene: Scene, slot: SlotFn) -> str | None:
    if not scene.modifier:
        return None
    s = _slot_for(scene, slot)
    return f"{s['subj']} {s['mod']} {s['pred']} {s['obj']} ."


def t_temporal_located(scene: Scene, slot: SlotFn) -> str | None:
    if not (scene.time and scene.location):
        return None
    s = _slot_for(scene, slot)
    return f"{s['time']} , {s['subj']} {s['pred']} {s['obj']} at {s['loc']} ."


def t_temporal_modified(scene: Scene, slot: SlotFn) -> str | None:
    if not (scene.time and scene.modifier):
        return None
    s = _slot_for(scene, slot)
    return f"{s['time']} , {s['subj']} {s['mod']} {s['pred']} {s['obj']} ."


# v048.1 — fill in the missing qualifier-presence combinations so
# every cluster size has at least one template that uses ALL the
# present qualifiers in one sentence. Without these, 4-edge clusters
# at inference fall back to the closest known pattern (compound) and
# leak qualifier relation names verbatim.

def t_modified_located(scene: Scene, slot: SlotFn) -> str | None:
    """subj mod pred obj at loc — no time."""
    if not (scene.modifier and scene.location):
        return None
    if scene.time:
        return None  # let t_temporal_located_modified handle the all-3 case
    s = _slot_for(scene, slot)
    return f"{s['subj']} {s['mod']} {s['pred']} {s['obj']} at {s['loc']} ."


def t_temporal_located_modified(scene: Scene, slot: SlotFn) -> str | None:
    """All three qualifiers in one sentence:
    time , subj mod pred obj at loc ."""
    if not (scene.time and scene.location and scene.modifier):
        return None
    s = _slot_for(scene, slot)
    return (
        f"{s['time']} , {s['subj']} {s['mod']} {s['pred']} {s['obj']} "
        f"at {s['loc']} ."
    )


def t_located_modified_alt(scene: Scene, slot: SlotFn) -> str | None:
    """Alternate ordering: subj pred obj at loc, mod ."""
    if not (scene.modifier and scene.location):
        return None
    if scene.time:
        return None
    s = _slot_for(scene, slot)
    return f"{s['subj']} {s['pred']} {s['obj']} at {s['loc']} , {s['mod']} ."


def t_temporal_located_modified_alt(scene: Scene, slot: SlotFn) -> str | None:
    """Alternate ordering: subj pred obj at loc , mod , time ."""
    if not (scene.time and scene.location and scene.modifier):
        return None
    s = _slot_for(scene, slot)
    return (
        f"{s['subj']} {s['pred']} {s['obj']} at {s['loc']} , "
        f"{s['mod']} , {s['time']} ."
    )


def t_compound(s1: Scene, s2: Scene, slot: SlotFn) -> str | None:
    """Compound — same subject, two actions joined with `and`."""
    if s1.subject != s2.subject:
        return None
    a = _slot_for(s1, slot)
    b = _slot_for(s2, slot)
    return f"{a['subj']} {a['pred']} {a['obj']} and {b['pred']} {b['obj']} ."


def t_complex_because(s1: Scene, s2: Scene, slot: SlotFn) -> str:
    a = _slot_for(s1, slot)
    b = _slot_for(s2, slot)
    return (
        f"{a['subj']} {a['pred']} {a['obj']} because "
        f"{b['subj']} {b['pred']} {b['obj']} ."
    )


def t_conditional_if(s1: Scene, s2: Scene, slot: SlotFn) -> str:
    a = _slot_for(s1, slot)
    b = _slot_for(s2, slot)
    return (
        f"if {a['subj']} {a['pred']} {a['obj']} then "
        f"{b['subj']} {b['pred']} {b['obj']} ."
    )


def t_discourse(s1: Scene, s2: Scene, connective_slot: str, slot: SlotFn) -> str:
    """Two scenes joined by a discourse connective slot like `<R0>`."""
    a = _slot_for(s1, slot)
    b = _slot_for(s2, slot)
    return (
        f"{a['subj']} {a['pred']} {a['obj']} . "
        f"{connective_slot} , {b['subj']} {b['pred']} {b['obj']} ."
    )


def t_temporal_sequence(s1: Scene, s2: Scene, slot: SlotFn) -> str | None:
    """Simple sequencing — both scenes share a time anchor or are
    chronologically obvious."""
    if not s1.time:
        return None
    a = _slot_for(s1, slot)
    b = _slot_for(s2, slot)
    return (
        f"{a['time']} , {a['subj']} {a['pred']} {a['obj']} . "
        f"then , {b['subj']} {b['pred']} {b['obj']} ."
    )


_SCENE_TEMPLATES = [
    t_simple, t_temporal_prefix, t_located_suffix, t_modified,
    t_temporal_located, t_temporal_modified,
    # v048.1 — full-qualifier templates that use ALL present
    # qualifiers in one sentence. Critical for 4-edge cluster
    # rendering — without these, the model can't bind a 3-qualifier
    # scene as one coherent sentence.
    t_modified_located, t_located_modified_alt,
    t_temporal_located_modified, t_temporal_located_modified_alt,
]
_PAIR_TEMPLATES = [
    t_compound, t_complex_because, t_conditional_if, t_temporal_sequence,
]


# v048.1 — multi-event subject arcs. A "subject arc" is N scenes
# (default 2-4) sharing a single subject across distinct time
# anchors, rendered as a chained narrative. Teaches the model that
# one subject + many edges = many sentences, NOT one big compound
# sentence. Without this, the model treats every multi-edge cluster
# as a candidate compound and emits noisy "and ... and ..." output.

def t_temporal_chain(scenes: list[Scene], slot: SlotFn) -> str | None:
    """Render an arc of scenes as one sentence per scene, time-
    prefixed. Each scene contributes ONE sentence; chronological
    order comes from the generator that built the arc."""
    if not scenes:
        return None
    parts: list[str] = []
    for sc in scenes:
        s = _slot_for(sc, slot)
        if "time" in s:
            parts.append(f"{s['time']} , {s['subj']} {s['pred']} {s['obj']} .")
        else:
            parts.append(f"{s['subj']} {s['pred']} {s['obj']} .")
    return " ".join(parts)


def t_temporal_chain_modified(scenes: list[Scene], slot: SlotFn) -> str | None:
    """Same as t_temporal_chain but uses the modifier when present."""
    if not scenes:
        return None
    parts: list[str] = []
    used_any_mod = False
    for sc in scenes:
        s = _slot_for(sc, slot)
        time_part = f"{s['time']} , " if "time" in s else ""
        if "mod" in s:
            used_any_mod = True
            parts.append(
                f"{time_part}{s['subj']} {s['mod']} {s['pred']} {s['obj']} ."
            )
        else:
            parts.append(f"{time_part}{s['subj']} {s['pred']} {s['obj']} .")
    if not used_any_mod:
        return None
    return " ".join(parts)


# ── Generator ────────────────────────────────────────────────────────


def _make_pool(rng: random.Random, n: int, n_words: int = 1) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        w = _random_compound(rng, n_words) if n_words > 1 else _random_word(rng)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _generate_scenes(
    rng: random.Random,
    n_scenes: int,
    n_subjects: int = 12,
    n_objects: int = 18,
    n_locations: int = 6,
    n_times: int = 8,
    location_p: float = 0.55,
    time_p: float = 0.55,
    modifier_p: float = 0.45,
) -> tuple[list[Scene], dict[str, list[str]]]:
    """Produce a list of scenes with bound slot fields plus the pools
    they were sampled from (so the arc generator can reuse the same
    pools for chronological subject-arcs).

    Returns (scenes, pools) where pools = {'subjects', 'objects',
    'locations', 'times'}."""
    subjects = _make_pool(rng, n_subjects, n_words=1)
    objects = _make_pool(rng, n_objects, n_words=rng.choice([1, 2]))
    locations = _make_pool(rng, n_locations, n_words=rng.choice([1, 2]))
    times = _make_pool(rng, n_times, n_words=1)

    scenes: list[Scene] = []
    for i in range(n_scenes):
        subj = rng.choice(subjects)
        action = rng.choice(_ACTION_VERBS)
        obj = rng.choice(objects)
        if obj == subj:
            continue
        scenes.append(Scene(
            subject=subj,
            action=action,
            object=obj,
            location=rng.choice(locations) if rng.random() < location_p else None,
            time=rng.choice(times) if rng.random() < time_p else None,
            modifier=rng.choice(_MANNER_MODIFIERS) if rng.random() < modifier_p else None,
            event_id=f"event:scene_{i}",
        ))
    pools = {
        "subjects": subjects, "objects": objects,
        "locations": locations, "times": times,
    }
    return scenes, pools


def _render_scene_to_pairs(
    scene: Scene, rng: random.Random,
) -> list[tuple[list[Edge], str, str]]:
    """For one scene, pick 1-2 single-scene templates that are valid
    given which optional fields are populated. Returns (edges, prose,
    subject) tuples."""
    out = []
    candidates = [t for t in _SCENE_TEMPLATES if t(scene, _identity_slot) is not None]
    if not candidates:
        return out
    picked = rng.sample(candidates, k=min(2, len(candidates)))
    edges = scene.to_edges()
    for tpl in picked:
        prose = tpl(scene, _identity_slot)
        if prose:
            out.append((edges, prose, scene.subject))
    return out


def _render_pair_to_pairs(
    s1: Scene, s2: Scene, rng: random.Random,
) -> list[tuple[list[Edge], str, str]]:
    out = []
    edges = s1.to_edges() + s2.to_edges()
    for tpl in _PAIR_TEMPLATES:
        prose = tpl(s1, s2, _identity_slot)
        if prose:
            out.append((edges, prose, s1.subject))
    # Discourse connective — one variant per scene pair.
    conn = f"<R{rng.randrange(0, 4)}>"
    prose = t_discourse(s1, s2, conn, _identity_slot)
    out.append((edges, prose, s1.subject))
    return out


def _generate_subject_arc(
    rng: random.Random,
    subjects: list[str],
    objects: list[str],
    locations: list[str],
    times: list[str],
    n_events: int = 3,
) -> list[Scene]:
    """v048.1 Slice 2 — generate N scenes that share a subject and
    chain through distinct time anchors. Times are sorted so the
    chained-sentence template reads chronologically."""
    if len(times) < n_events:
        n_events = len(times)
    if n_events < 2:
        return []
    subj = rng.choice(subjects)
    arc_times = rng.sample(times, n_events)
    arc_times.sort()
    out: list[Scene] = []
    for i, t in enumerate(arc_times):
        obj = rng.choice(objects)
        if obj == subj:
            continue
        out.append(Scene(
            subject=subj,
            action=rng.choice(_ACTION_VERBS),
            object=obj,
            time=t,
            location=rng.choice(locations) if rng.random() < 0.5 else None,
            modifier=rng.choice(_MANNER_MODIFIERS) if rng.random() < 0.4 else None,
            event_id=f"event:arc_{rng.randrange(0, 10**9)}_{i}",
        ))
    return out


def _render_arc_to_pairs(
    arc: list[Scene], rng: random.Random,
) -> list[tuple[list[Edge], str, str]]:
    """Render a subject arc with the chained-sentence templates.
    Each scene contributes its own edges to the cluster; the prose
    is one chain per template variant."""
    out: list[tuple[list[Edge], str, str]] = []
    if not arc:
        return out
    edges: list[Edge] = []
    for sc in arc:
        edges.extend(sc.to_edges())
    for tpl in (t_temporal_chain, t_temporal_chain_modified):
        prose = tpl(arc, _identity_slot)
        if prose:
            out.append((edges, prose, arc[0].subject))
    return out


def _identity_slot(s: str) -> str:
    """Used during rendering: emit the substrate string verbatim;
    `synth_data.serialize_example`'s `build_slot_mapping` will
    convert content strings to `<C0>`...`<C31>` slot tokens later.

    Discourse connective slots `<R0>`...<R3>` and time/loc/mod slots
    are emitted directly when the templates choose to use them; the
    serializer treats already-slotted tokens specially via its
    `inverse` mapping."""
    return s


# ── Serializer for complex pairs ─────────────────────────────────────


_PUNCT = {".", ",", ";", ":", "?", "!", "-"}


def _split_prose_tokens(prose: str) -> list[str]:
    """Split rendered prose into atomic tokens. Already pre-spaced
    around punctuation by the templates, so a whitespace split is
    enough. Slot tokens like `<R0>` stay intact."""
    return [t for t in prose.split() if t]


def _serialize_complex_pair(
    edges: list[Edge], prose: str, subject: str,
) -> dict | None:
    """Tokenize one (edges, prose, subject) pair into a training row.

    Substrate strings in `prose` get substituted with content slots
    (`<C0>`...) in encounter order; relation names get substituted
    with predicate slots (`<P0>`...). Returns None if the row would
    be empty after tokenization, which can happen when the prose
    references a substrate string that's not in any edge (template
    bug — we skip rather than emit garbage).
    """
    if not edges:
        return None

    # Build content slot mapping in encounter order over edges.
    content_map: dict[str, str] = {}    # substrate string -> <Cn>
    next_c = 0
    for e in edges:
        for s in (e.src, e.tgt):
            s = s.strip()
            if not s or s in content_map:
                continue
            if next_c >= N_SLOTS:
                break
            content_map[s] = slot_token(next_c)
            next_c += 1
        if next_c >= N_SLOTS:
            break

    # Predicate slot mapping in encounter order.
    pred_map: dict[str, str] = {}
    next_p = 0
    for e in edges:
        if e.rel in pred_map or next_p >= N_PRED_SLOTS:
            continue
        pred_map[e.rel] = pred_slot_token(next_p)
        next_p += 1

    # Phrase-level slot substitution BEFORE tokenizing so multi-word
    # substrate strings ("hitedoza ubake") match as one slot. Sort by
    # length descending so longer phrases get matched before any
    # sub-string of them.
    import re
    substituted = prose
    for substrate_str, slot in sorted(
        content_map.items(), key=lambda kv: -len(kv[0]),
    ):
        if not substrate_str:
            continue
        substituted = re.sub(
            r"\b" + re.escape(substrate_str) + r"\b",
            slot,
            substituted,
        )
    for relation, slot in sorted(
        pred_map.items(), key=lambda kv: -len(kv[0]),
    ):
        if not relation:
            continue
        substituted = re.sub(
            r"\b" + re.escape(relation) + r"\b",
            slot,
            substituted,
        )
    out_tokens = _split_prose_tokens(substituted)

    # Build the facts prefix.
    ids: list[int] = [BOS_ID, FACTS_ID]
    for e in edges:
        ids.append(SUBJ_ID)
        ids.append(TOK2ID_SYNTH[content_map[e.src]] if e.src in content_map
                   else _encode_word(e.src))
        ids.append(PRED_ID)
        if e.rel in pred_map:
            ids.append(TOK2ID_SYNTH[pred_map[e.rel]])
        else:
            for w in e.rel.replace("_", " ").split():
                ids.append(_encode_word(w))
        ids.append(OBJ_ID)
        ids.append(TOK2ID_SYNTH[content_map[e.tgt]] if e.tgt in content_map
                   else _encode_word(e.tgt))
        ids.append(EDGE_SEP_ID)
    n_facts = len(ids)

    ids.append(PROSE_ID)
    prose_start = len(ids)
    for tok in out_tokens:
        # Lowercase non-bracket tokens to match vocab_synth convention.
        if tok.startswith("<") and tok.endswith(">"):
            t_lookup = tok
        elif tok in _PUNCT:
            t_lookup = tok
        else:
            t_lookup = tok.lower()
        ids.append(TOK2ID_SYNTH.get(t_lookup, UNK_ID))
    ids.append(END_PROSE_ID)
    ids.append(EOS_ID)
    n_prose = len(ids) - prose_start

    if n_prose <= 1:
        return None

    loss_mask = [0] * len(ids)
    for i in range(prose_start - 1, len(ids) - 1):
        loss_mask[i] = 1

    return {
        "input_ids": ids,
        "loss_mask": loss_mask,
        "slot_mapping": content_map,
        "pred_mapping": pred_map,
        "n_facts": n_facts,
        "n_prose": n_prose,
    }


def _encode_word(w: str) -> int:
    """Encode a single substrate word/literal as a vocab id, lower-
    cased. Out-of-vocab labels round-trip as UNK at training time;
    the model still learns the structural slot positioning. Substrate
    content rides through slots, not vocab."""
    return TOK2ID_SYNTH.get(w.lower(), UNK_ID)


def write_complex_corpus(
    out_db: Path, n_scenes: int = 800, seed: int | None = None,
) -> dict:
    if out_db.exists():
        raise FileExistsError(out_db)
    seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
    rng = random.Random(seed)

    scenes, pools = _generate_scenes(rng, n_scenes)

    # Build (edges, prose, subject) pairs.
    all_pairs: list[tuple[list[Edge], str, str]] = []
    for scene in scenes:
        all_pairs.extend(_render_scene_to_pairs(scene, rng))

    # Pair scenes for compound / complex / discourse templates.
    rng.shuffle(scenes)
    n_pair = len(scenes) // 2
    for i in range(n_pair):
        s1 = scenes[2 * i]
        s2 = scenes[2 * i + 1]
        all_pairs.extend(_render_pair_to_pairs(s1, s2, rng))

    # Some compound templates need same-subject pairs — synthesize a
    # few by hand so the model sees them.
    for _ in range(n_scenes // 6):
        s1 = rng.choice(scenes)
        s2 = Scene(
            subject=s1.subject,
            action=rng.choice(_ACTION_VERBS),
            object=rng.choice([sc.object for sc in scenes if sc != s1]),
            event_id=f"event:compound_{rng.randrange(0, 10**9)}",
        )
        prose = t_compound(s1, s2, _identity_slot)
        if prose:
            all_pairs.append((s1.to_edges() + s2.to_edges(), prose, s1.subject))

    # v048.1 Slice 2: subject arcs — N scenes sharing a subject across
    # distinct time anchors, rendered as chained narrative. ~n_scenes/4
    # arcs of 2-4 events each gives the model robust exposure to the
    # "many edges, one subject -> many sentences" pattern.
    n_arcs = max(1, n_scenes // 4)
    for _ in range(n_arcs):
        n_events = rng.choice([2, 3, 3, 4])
        arc = _generate_subject_arc(
            rng,
            subjects=pools["subjects"], objects=pools["objects"],
            locations=pools["locations"], times=pools["times"],
            n_events=n_events,
        )
        all_pairs.extend(_render_arc_to_pairs(arc, rng))

    # Write substrate edges (deduped) into a brain.db.
    out_db.parent.mkdir(parents=True, exist_ok=True)
    brain = Brain(str(out_db))
    seen_edges: set[tuple[str, str, str]] = set()
    for edges, _, _ in all_pairs:
        for e in edges:
            key = (e.src, e.rel, e.tgt)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            brain.teach_triple(e.src, e.rel, e.tgt, source_label=f"complex_seed_{seed}")

    # Write the inspection pairs.jsonl (free-form, for debugging).
    pairs_path = out_db.with_suffix(".pairs.jsonl")
    with pairs_path.open("w", encoding="utf-8") as f:
        for edges, prose, subject in all_pairs:
            row = {
                "edges": [asdict(e) for e in edges],
                "prose": prose,
                "subject": subject,
            }
            f.write(json.dumps(row) + "\n")

    # Write the tokenized JSONL ready for `train_synth.py`.
    tok_path = out_db.with_suffix(".tokenized.jsonl")
    written = 0
    skipped = 0
    seq_lens: list[int] = []
    with tok_path.open("w", encoding="utf-8") as f:
        for edges, prose, subject in all_pairs:
            row = _serialize_complex_pair(edges, prose, subject)
            if row is None:
                skipped += 1
                continue
            f.write(json.dumps(row) + "\n")
            written += 1
            seq_lens.append(len(row["input_ids"]))

    n_neurons = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    n_segments = brain.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]

    manifest = {
        "schema_version": 1,
        "substrate_type": "complex_synthetic",
        "seed": seed,
        "n_scenes": len(scenes),
        "n_pairs": len(all_pairs),
        "n_neurons": n_neurons,
        "n_segments": n_segments,
        "out_db": str(out_db.resolve()),
        "pairs_path": str(pairs_path.resolve()),
        "tokenized_path": str(tok_path.resolve()),
        "tokenized_written": written,
        "tokenized_skipped": skipped,
        "tokenized_avg_seq_len": (sum(seq_lens) / max(1, len(seq_lens))),
        "tokenized_max_seq_len": (max(seq_lens) if seq_lens else 0),
    }
    manifest_path = out_db.with_suffix(".manifest.json")
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, type=Path,
                   help="Output brain.db path (e.g. /tmp/complex_substrate.db)")
    p.add_argument("--scenes", type=int, default=800,
                   help="Number of scenes (default: 800)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    info = write_complex_corpus(args.out, n_scenes=args.scenes, seed=args.seed)
    print("Complex substrate generated.")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
