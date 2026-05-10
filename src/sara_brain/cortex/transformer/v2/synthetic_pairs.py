"""Synthetic (prose, triple) pair generator for extraction-head training.

Reuses the scene generator and templates already in
[papers/instrument_validation/generate_complex_substrate.py]
to produce nonsense-content scenes, renders each with one of the
existing template functions, and emits one JSONL record per (prose,
triple) pair with character spans for subject / relation / object.

The model trains on extraction-by-position. Subjects and objects are
nonsense words — by construction the model can't memorize content. The
relation is a real English word from the existing relations pool, which
is fine: the head learns where the relation TOKEN sits in the sentence,
not which relations have which meanings.

Output line format::

    {
      "prose":           "<text>",
      "subject":         "<subject substrate label>",
      "relation":        "<relation>",
      "object":          "<object substrate label>",
      "subject_span":    [start_char, end_char],
      "relation_span":   [start_char, end_char],
      "object_span":     [start_char, end_char],
      "template":        "<name of template function>",
      "scene_qualifiers": ["time"|"location"|"modifier", ...]
    }

Auxiliary qualifier facts (`at_location`, `at_time`, `in_manner`) are
not emitted as separate triples here — they're qualifier mentions in
the prose, and the extraction head's job is the central s/r/o triple.
A future enhancement can emit them as additional triples per clause.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Import the existing scene generator + templates by file path so this
# module is reusable from the package without restructuring papers/.
_GEN_PATH = Path(__file__).resolve().parents[5] / "papers" / "instrument_validation" / "generate_complex_substrate.py"


def _load_gen_module():
    name = "_gen_complex_substrate"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_GEN_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scene generator from {_GEN_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # required so @dataclass can resolve __module__
    spec.loader.exec_module(mod)
    return mod


# Templates that operate on a single scene. (Multi-scene templates like
# t_compound require alignment we skip here; the head trains on
# single-clause prose anyway since EnhancedParser._split_compound
# splits compound input upstream.)
_SINGLE_SCENE_TEMPLATES: list[str] = [
    "t_simple",
    "t_temporal_prefix",
    "t_located_suffix",
    "t_modified",
    "t_temporal_located",
    "t_temporal_modified",
    "t_modified_located",
    "t_temporal_located_modified",
    "t_located_modified_alt",
    "t_temporal_located_modified_alt",
]


@dataclass
class Pair:
    prose: str
    subject: str
    relation: str
    obj: str
    subject_span: tuple[int, int]
    relation_span: tuple[int, int]
    object_span: tuple[int, int]
    template: str
    qualifiers: list[str]

    def to_json(self) -> dict:
        return {
            "prose": self.prose,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.obj,
            "subject_span": list(self.subject_span),
            "relation_span": list(self.relation_span),
            "object_span": list(self.object_span),
            "template": self.template,
            "scene_qualifiers": self.qualifiers,
        }


def _find_span(text: str, needle: str, start_at: int = 0) -> tuple[int, int] | None:
    """Locate `needle` in `text` starting at start_at. Returns (start, end)
    of the FIRST occurrence at or after start_at, or None."""
    if not needle:
        return None
    idx = text.find(needle, start_at)
    if idx < 0:
        return None
    return (idx, idx + len(needle))


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _build_pair_strings(
    prose: str,
    subject: str,
    relation: str,
    obj: str,
    template_name: str,
    qualifiers: list[str],
    *,
    object_after: int = 0,
) -> "Pair | None":
    """Order-independent pair builder.

    Finds `subject` anywhere in `prose`, then `obj` somewhere
    non-overlapping (search starts at `object_after` so list templates
    can ask for the Nth object), then `relation` somewhere
    non-overlapping with both. Returns None if any span is missing or
    spans collide.

    Used for templates whose word order is not strictly SVO (passive,
    copular, list-object).
    """
    s_span = _find_span(prose, subject)
    if s_span is None:
        return None
    o_span = _find_span(prose, obj, start_at=object_after)
    if o_span is None or _spans_overlap(s_span, o_span):
        # Try to find a non-overlapping object occurrence anywhere.
        cand = _find_span(prose, obj, start_at=s_span[1])
        if cand is None or _spans_overlap(s_span, cand):
            return None
        o_span = cand
    r_span = _find_span(prose, relation)
    if r_span is None:
        return None
    if _spans_overlap(r_span, s_span) or _spans_overlap(r_span, o_span):
        # Try other relation occurrences.
        cursor = r_span[1]
        while True:
            cand = _find_span(prose, relation, start_at=cursor)
            if cand is None:
                return None
            if not (_spans_overlap(cand, s_span) or _spans_overlap(cand, o_span)):
                r_span = cand
                break
            cursor = cand[1]
    return Pair(
        prose=prose,
        subject=subject,
        relation=relation,
        obj=obj,
        subject_span=s_span,
        relation_span=r_span,
        object_span=o_span,
        template=template_name,
        qualifiers=qualifiers,
    )


def _make_pair(scene, template_name: str, template_fn: Callable, gen) -> Pair | None:
    """Render a scene with the named template and return a Pair, or None
    if the template is not applicable (qualifier missing) or spans
    cannot be located cleanly."""
    # The slot fn is identity here — we want the actual nonsense words
    # to appear in the prose so we can locate spans.
    prose = template_fn(scene, lambda s: s)
    if prose is None:
        return None

    s_span = _find_span(prose, scene.subject)
    if s_span is None:
        return None
    # Search for relation AFTER the subject ends to avoid false hits in
    # qualifier prefixes (e.g., the time word coincidentally containing
    # the relation string).
    r_span = _find_span(prose, scene.action, start_at=s_span[1])
    if r_span is None:
        return None
    o_span = _find_span(prose, scene.object, start_at=r_span[1])
    if o_span is None:
        return None

    qualifiers: list[str] = []
    if scene.time:
        qualifiers.append("time")
    if scene.location:
        qualifiers.append("location")
    if scene.modifier:
        qualifiers.append("modifier")

    return Pair(
        prose=prose,
        subject=scene.subject,
        relation=scene.action,
        obj=scene.object,
        subject_span=s_span,
        relation_span=r_span,
        object_span=o_span,
        template=template_name,
        qualifiers=qualifiers,
    )


# ── Weird-token generator ────────────────────────────────────────────
# Real scientific prose is full of single tokens with special characters
# inside them — RNA strand notation ("5'3'"), kinetic constants ("k_off"),
# alphanumeric concept IDs ("SSNG3", "Creed2"), units ("1.5mM", "37°C"),
# significance markers ("p<0.05"), hyphenated compounds ("fold-signal").
# The head needs to see these in subject/object positions during training
# so it learns to tag them like ordinary content tokens.
#
# By construction these weird tokens stay atomic with the whitespace-first
# tokenizer — they have no internal whitespace. The synthetic generator
# weaves them into NPs alongside ordinary pronounceable nonsense, so
# every span position can sometimes be filled by a weird-shape token.

_WEIRD_TOKEN_BUILDERS: tuple = ()  # populated below


def _wt_strand_notation(rng) -> str:
    """5', 3', 5'3', 3'5', N'M' — RNA strand orientation."""
    a = rng.randint(1, 30)
    if rng.random() < 0.5:
        return f"{a}'"
    b = rng.randint(1, 30)
    return f"{a}'{b}'"


def _wt_alphanum_id(rng, gen) -> str:
    """ssng3, creed42, kdon, kdoff — letter base plus digits."""
    base = gen._random_word(rng, min_len=3, max_len=6)
    if rng.random() < 0.5:
        return base + str(rng.randint(0, 99))
    return base.upper() + str(rng.randint(0, 99))


def _wt_subscript(rng, gen) -> str:
    """k_off, K_d, p_value — letter base + underscore + suffix."""
    base = gen._random_word(rng, min_len=1, max_len=4)
    suffix = gen._random_word(rng, min_len=2, max_len=5)
    if rng.random() < 0.3:
        base = base.upper()
    return f"{base}_{suffix}"


def _wt_concentration(rng) -> str:
    """1.5mM, 0.5μM, 37°C, 100nM — value + unit."""
    val = round(rng.uniform(0.1, 999.0), rng.choice([0, 1, 2]))
    unit = rng.choice(["mM", "μM", "uM", "nM", "pM", "°C", "%", "mg/mL", "ng/mL"])
    return f"{val}{unit}"


def _wt_significance(rng) -> str:
    """p<0.05, p<0.001, p>0.05 — comparison-with-decimal."""
    op = rng.choice(["<", ">", "<=", ">="])
    val = rng.choice([0.05, 0.01, 0.001, 0.5, 0.1])
    return f"p{op}{val}"


def _wt_hyphen_compound(rng, gen) -> str:
    """fold-signal, high-end, cap-and-trade — hyphen-joined nonsense."""
    n = rng.choice([2, 3])
    parts = [gen._random_word(rng, min_len=3, max_len=6) for _ in range(n)]
    return "-".join(parts)


def _wt_slash_pair(rng, gen) -> str:
    """a/b, mg/mL, x/y — slash-joined."""
    a = gen._random_word(rng, min_len=2, max_len=5)
    b = gen._random_word(rng, min_len=2, max_len=5)
    return f"{a}/{b}"


def _wt_keyval(rng, gen) -> str:
    """ph=7.4, x=y, k=2 — key=value-ish."""
    key = gen._random_word(rng, min_len=2, max_len=4)
    val_kind = rng.choice(["int", "decimal", "word"])
    if val_kind == "int":
        val = rng.randint(1, 99)
    elif val_kind == "decimal":
        val = round(rng.uniform(0.01, 99.9), rng.choice([1, 2]))
    else:
        val = gen._random_word(rng, min_len=2, max_len=4)
    return f"{key}={val}"


def _wt_capital_acronym(rng) -> str:
    """ATP, RNA, DNA, MRNA-shaped — short capitalized strings, sometimes
    with digits or hyphens."""
    n = rng.randint(2, 5)
    letters = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(n))
    suffix_kind = rng.choice(["none", "digit", "hyphen"])
    if suffix_kind == "digit":
        return letters + str(rng.randint(1, 99))
    if suffix_kind == "hyphen":
        return f"{letters}-{rng.randint(1, 99)}"
    return letters


_WEIRD_TOKEN_BUILDERS = (
    _wt_strand_notation,
    _wt_alphanum_id,
    _wt_subscript,
    _wt_concentration,
    _wt_significance,
    _wt_hyphen_compound,
    _wt_slash_pair,
    _wt_keyval,
    _wt_capital_acronym,
)


def _random_weird_token(rng, gen) -> str:
    """Pick a weird-token pattern uniformly and generate one. The
    pattern set covers the most common scientific-notation shapes: RNA
    strand, alphanumeric ID, subscript, concentration, significance,
    hyphenated compound, slash pair, key=value, capital acronym."""
    builder = rng.choice(_WEIRD_TOKEN_BUILDERS)
    # Some builders need access to the upstream nonsense generator;
    # they take (rng, gen). Others take just (rng).
    try:
        return builder(rng, gen)
    except TypeError:
        return builder(rng)


# Per-NP probability of substituting one of the words for a weird token.
_WEIRD_TOKEN_PROB = 0.20


def _maybe_swap_for_weird_token(np: str, rng, gen) -> str:
    """With some probability, replace one word in `np` (or the whole NP
    if single-word) with a weird-shape token. Subjects/objects pass
    through this filter so the head sees scientific notation across
    span positions."""
    if rng.random() >= _WEIRD_TOKEN_PROB:
        return np
    weird = _random_weird_token(rng, gen)
    parts = np.split(" ")
    # Pick a position to swap (last word is the head noun for compounds)
    idx = rng.randrange(len(parts))
    parts[idx] = weird
    return " ".join(parts)


# ── Determiner attachment ────────────────────────────────────────────
# Real-prose noun phrases routinely lead with "the" / "a" / "an". The
# head needs to learn that subject/object spans extend across the
# determiner — otherwise predictions fragment as ('the', verb, 'X')
# instead of (X-with-the, verb, X). We attach articles to a fraction of
# noun phrases at scene-construction time and treat the article as part
# of the canonical subject/object string, so the span ground truth
# includes it.

_DEFINITE_PROB = 0.45    # chance any given NP gets a "the"
_INDEFINITE_PROB = 0.15  # chance an NP without "the" gets "a"/"an" instead
_NUMBERED_NP_PROB = 0.18  # chance an NP gets a trailing nummod ("X 2", "Y 1")


# Bare-stem (present-tense) verbs with particles, paralleling the
# upstream `_ACTION_VERBS` pool which is mostly past-tense. Real-paper
# prose uses these constantly ("X folds into Y", "X relies on Z") and
# the head needs supervision that the particle is part of the relation.
# Underscores get rendered as spaces in prose; the action string in
# the triple is the spaced form, so the head learns B-R + I-R for
# multi-word relations.
_VERB_PARTICLE_VERBS: tuple[str, ...] = (
    "folds_into", "relies_on", "looks_at", "depends_on",
    "leans_on", "binds_to", "switches_between", "consists_of",
    "points_at", "starts_with", "ends_with", "agrees_with",
    "applies_to", "leads_to", "results_in", "translates_to",
    "interacts_with", "couples_with", "connects_to", "contributes_to",
    "differs_from", "stems_from", "corresponds_to", "responds_to",
)


def _maybe_attach_number(np: str, rng: random.Random) -> str:
    """Occasionally append a small integer to an NP ("Creed 2", "SSNG 3").
    Real-paper subjects often carry numbered identifiers; the synthetic
    NPs never had them, so the head couldn't learn that nummod
    extends a span. Numbers chosen from a small set so they appear
    repeatedly in training and the head can learn the nummod pattern."""
    if not np:
        return np
    if rng.random() >= _NUMBERED_NP_PROB:
        return np
    n = rng.choice([1, 2, 3, 4, 5, 6, 7])
    return f"{np} {n}"


def _maybe_attach_article(np: str, rng: random.Random) -> str:
    """Randomly prepend 'the' / 'a' / 'an' to a noun phrase.

    Skips NPs that already start with a determiner (defensive — none
    of the upstream nonsense generators produce them, but cheap to
    check). Indefinite article picks 'a' vs 'an' by next-vowel rule.
    """
    if not np:
        return np
    first_word = np.split(" ", 1)[0].lower()
    if first_word in ("the", "a", "an"):
        return np
    r = rng.random()
    if r < _DEFINITE_PROB:
        return f"the {np}"
    if r < _DEFINITE_PROB + _INDEFINITE_PROB:
        article = "an" if np[:1] in "aeiou" else "a"
        return f"{article} {np}"
    return np


# ── Rich templates ──────────────────────────────────────────────────
# Single-clause patterns that the upstream gen module doesn't cover.
# These don't need multi-scene alignment — each one renders one clause
# from one underlying triple (or a list of triples sharing prose).


# Verbs that don't passivize cleanly (intransitive or particle-heavy).
_INTRANSITIVE_VERBS = {
    "walked", "ran", "arrived", "left", "sat", "stood",
    "began", "finished",
}


def _is_passivizable(action: str) -> bool:
    """Heuristic: keep transitive single-word verbs; drop intransitive
    and "verb particle" forms whose passive sounds wrong."""
    if " " in action:  # "walked to", "arrived at" — drop
        return False
    return action.lower() not in _INTRANSITIVE_VERBS


def _render_copular(subject: str, obj: str, *, indef: bool) -> str:
    if indef:
        article = "an" if obj[:1] in "aeiou" else "a"
        return f"{subject} is {article} {obj} ."
    return f"{subject} is {obj} ."


def _render_list_two(subject: str, action: str, o1: str, o2: str) -> str:
    return f"{subject} {action} {o1} and {o2} ."


def _render_list_three(subject: str, action: str, o1: str, o2: str, o3: str) -> str:
    return f"{subject} {action} {o1} , {o2} , and {o3} ."


def _render_passive(subject: str, action: str, obj: str) -> str:
    return f"{obj} was {action} by {subject} ."


def _emit_rich_pairs(
    subject: str,
    action: str,
    obj: str,
    extra_objects: list[str],
    qualifiers: list[str],
    rng: random.Random,
) -> list[Pair]:
    """Render the three rich-template families for one base triple.

    Returns zero or more Pair records (one per derived triple). Some
    templates emit multiple Pairs (list-object) sharing the same prose.
    """
    out: list[Pair] = []

    # Copular — simple "X is Y ." and indefinite "X is a Y ."
    cop_simple = _render_copular(subject, obj, indef=False)
    p = _build_pair_strings(cop_simple, subject, "is", obj, "t_copular", qualifiers)
    if p is not None:
        out.append(p)
    # Skip the indefinite variant if obj already leads with a determiner —
    # otherwise we'd double-article ("is a the dekubuju").
    obj_lead = obj.split(" ", 1)[0].lower()
    if obj_lead not in ("the", "a", "an"):
        cop_indef = _render_copular(subject, obj, indef=True)
        p = _build_pair_strings(cop_indef, subject, "is", obj, "t_copular_indef", qualifiers)
        if p is not None:
            out.append(p)

    # Passive — only for transitive single-word verbs.
    if _is_passivizable(action):
        passive = _render_passive(subject, action, obj)
        p = _build_pair_strings(
            passive, subject, action, obj, "t_passive", qualifiers,
        )
        if p is not None:
            out.append(p)

    # List-object — one prose, one Pair per object in the list.
    if extra_objects:
        if len(extra_objects) >= 2:
            o1, o2, o3 = obj, extra_objects[0], extra_objects[1]
            prose = _render_list_three(subject, action, o1, o2, o3)
            for target_obj in (o1, o2, o3):
                start_after = 0
                if target_obj == o2:
                    # Skip past o1 occurrence
                    occ1 = _find_span(prose, o1)
                    start_after = occ1[1] if occ1 else 0
                elif target_obj == o3:
                    occ2 = _find_span(prose, o2)
                    start_after = occ2[1] if occ2 else 0
                p = _build_pair_strings(
                    prose, subject, action, target_obj,
                    "t_list_object_three", qualifiers,
                    object_after=start_after,
                )
                if p is not None:
                    out.append(p)
        else:  # len == 1
            o1, o2 = obj, extra_objects[0]
            prose = _render_list_two(subject, action, o1, o2)
            for target_obj in (o1, o2):
                start_after = 0
                if target_obj == o2:
                    occ1 = _find_span(prose, o1)
                    start_after = occ1[1] if occ1 else 0
                p = _build_pair_strings(
                    prose, subject, action, target_obj,
                    "t_list_object_two", qualifiers,
                    object_after=start_after,
                )
                if p is not None:
                    out.append(p)
    return out


def generate_pairs(
    n_scenes: int = 1000,
    seed: int = 0,
    qualifier_prob: float = 0.6,
    extra_object_prob: float = 0.4,
    rich_templates: bool = True,
) -> list[Pair]:
    """Generate `n_scenes` random scenes, render each with every
    applicable template, return the resulting pairs.

    Templates included:
      - 10 single-scene action templates from the upstream complex
        substrate generator (simple, temporal, located, modified,
        and combinations).
      - **Rich templates** added in v2: copular ("X is Y."), copular
        indef ("X is a Y."), passive ("Y was VERBed by X."),
        list-object two ("X verbed A and B.") and list-object three
        ("X verbed A, B, and C."). These cover patterns that real
        paper prose uses heavily.

    Args:
      qualifier_prob: probability each optional qualifier
        (time/location/modifier) is set on a scene. Higher = richer
        action-template variants.
      extra_object_prob: probability the scene gets an `extra_objects`
        list for list-object templates. With prob/2 it gets one
        extra (two-object list), with prob/2 it gets two extras
        (three-object list).
      rich_templates: include the new copular/passive/list templates.
        Set False to reproduce the v1 single-scene-action-only
        behaviour.
    """
    gen = _load_gen_module()
    rng = random.Random(seed)

    template_fns: dict[str, Callable] = {
        name: getattr(gen, name) for name in _SINGLE_SCENE_TEMPLATES
        if hasattr(gen, name)
    }
    if not template_fns:
        raise RuntimeError("no template functions found in scene generator")

    # Augmented action pool: upstream past-tense verbs PLUS bare-stem
    # verb-particle constructions ("folds_into", "relies_on", ...).
    # ~30% of scenes pick from the verb-particle subset so the head
    # gets explicit B-R + I-R training on multi-word present-tense
    # relations.
    augmented_actions = list(gen._ACTION_VERBS) + list(_VERB_PARTICLE_VERBS)

    pairs: list[Pair] = []
    for i in range(n_scenes):
        subject_core = gen._random_compound(rng, n_words=rng.choice([1, 2]))
        obj_core = gen._random_compound(rng, n_words=rng.choice([1, 2]))
        # Weird-token swap (20% prob per NP): replace one word with a
        # scientific-notation-shaped token (RNA strand 5'3', subscript
        # k_off, alphanumeric SSNG3, concentration 1.5mM, ...). Done
        # BEFORE article/number attachment so the wrapper still
        # composes cleanly.
        subject_core = _maybe_swap_for_weird_token(subject_core, rng, gen)
        obj_core = _maybe_swap_for_weird_token(obj_core, rng, gen)
        # Occasional trailing-number modifier ("ekefu 2") so the head
        # learns that nummod children of a head noun extend the NP span.
        subject_core = _maybe_attach_number(subject_core, rng)
        obj_core = _maybe_attach_number(obj_core, rng)
        # Attach articles to ~60% of noun phrases.
        subject = _maybe_attach_article(subject_core, rng)
        obj = _maybe_attach_article(obj_core, rng)
        # Pick from the augmented pool (past-tense actions + bare-stem
        # verb-particles). Render underscores as spaces in prose; the
        # action stays the spaced form in the triple ground truth.
        action_canonical = rng.choice(augmented_actions)
        action = action_canonical.replace("_", " ")
        # Optional qualifiers. The temporal templates from the upstream
        # generator render `f"{time} ,"` at sentence start, so a bare
        # noun ("ahonige ,") is grammatically ambiguous with a fronted
        # subject. Prepending a temporal preposition ("on ahonige ,",
        # "during ahonige ,") makes the qualifier unambiguous and gives
        # the head a learnable surface cue. Locations already have "at"
        # baked into the upstream templates, so a bare noun for location
        # is fine. Modifier comes from the manner-adverb pool already.
        if rng.random() < qualifier_prob:
            prep = rng.choice(gen._TIME_FRAME_PREFIXES)
            time = f"{prep} {gen._random_word(rng)}"
        else:
            time = None
        location = gen._random_word(rng) if rng.random() < qualifier_prob else None
        modifier = (rng.choice(gen._MANNER_MODIFIERS)
                    if rng.random() < qualifier_prob else None)

        # Optional extra objects for list templates. Each extra object
        # also runs through the weird-token swap so the head sees
        # weird shapes in list positions too.
        extra_objects: list[str] = []
        if rich_templates and rng.random() < extra_object_prob:
            n_extras = rng.choice([1, 2])
            for _ in range(n_extras):
                core = gen._random_compound(rng, n_words=rng.choice([1, 2]))
                core = _maybe_swap_for_weird_token(core, rng, gen)
                extra_objects.append(_maybe_attach_article(core, rng))
            extra_objects = [
                e for e in extra_objects if e != obj and e != subject
            ]

        scene = gen.Scene(
            subject=subject,
            action=action,
            object=obj,
            location=location,
            time=time,
            modifier=modifier,
            event_id=f"e_{i}",
        )

        # Render every applicable upstream action template.
        for name, fn in template_fns.items():
            pair = _make_pair(scene, name, fn, gen)
            if pair is not None:
                pairs.append(pair)

        # Rich templates (copular / passive / list-object).
        if rich_templates:
            qualifiers: list[str] = []
            if time:
                qualifiers.append("time")
            if location:
                qualifiers.append("location")
            if modifier:
                qualifiers.append("modifier")
            pairs.extend(_emit_rich_pairs(
                subject, action, obj, extra_objects, qualifiers, rng,
            ))
    return pairs


def write_jsonl(pairs: list[Pair], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_json()) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="synthetic_pairs",
        description="Generate (prose, triple) pairs for extraction-head training.",
    )
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--scenes", type=int, default=1000, help="Number of base scenes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--qualifier-prob", type=float, default=0.6)
    args = p.parse_args(argv)

    pairs = generate_pairs(
        n_scenes=args.scenes,
        seed=args.seed,
        qualifier_prob=args.qualifier_prob,
    )
    write_jsonl(pairs, Path(args.out))
    print(
        f"wrote {len(pairs)} pairs to {args.out} "
        f"(scenes={args.scenes}, qualifier_prob={args.qualifier_prob})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
