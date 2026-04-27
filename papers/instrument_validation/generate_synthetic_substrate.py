"""Generate a synthetic Sara substrate of random labels.

Why this exists: instrument validation needs substrates whose contents
cannot be in any LLM training corpus. Random labels generated at
runtime are training-orthogonal by construction — no model has been
trained on them because they didn't exist before this run.

This replaces the methodological-discipline approach (test substrate
age, document publication timeline, etc.) with an engineering one
(generate substrates that are orthogonal by definition).

Usage:
    python generate_synthetic_substrate.py \\
        --out instrument_validation/synth_001.db \\
        --concepts 30 --triples 80 --seed 42

Produces:
    - <out>.db                — Sara brain with the random triples taught
    - <out>.manifest.json     — canonical record of every concept and
                                 triple in the substrate, for later
                                 grading of Session B/C responses

Reproducibility: pass --seed to regenerate the same substrate. Default
seed is current time, so each run is fresh.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from sara_brain.core.brain import Brain


# Pronounceable-but-meaningless syllable inventory. Avoids real words by
# using consonant clusters and vowel combinations that don't form
# common English morphemes.
_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"


def _random_word(rng: random.Random, min_len: int = 5, max_len: int = 8) -> str:
    """One pronounceable nonsense word, e.g. 'zilkrap', 'bortle'."""
    length = rng.randint(min_len, max_len)
    word = []
    use_consonant = rng.random() < 0.5
    for _ in range(length):
        word.append(rng.choice(_CONSONANTS if use_consonant else _VOWELS))
        use_consonant = not use_consonant
    return "".join(word)


def _random_compound(rng: random.Random, n_words: int = 2) -> str:
    """Multi-word compound label."""
    return " ".join(_random_word(rng) for _ in range(n_words))


# Relation labels. These ARE real English words, intentionally — the
# orthogonality property only needs to hold on the CONCEPT labels (the
# subjects and objects). Relations being real words is fine and makes
# the substrate readable for inspection.
_RELATIONS_POOL = [
    "is_a",
    "has_property",
    "part_of",
    "produces",
    "requires",
    "interacts_with",
    "used_for",
    "described_by",
    "predicts",
    "contains",
    "opposes",
    "enables",
]


def generate_synthetic_substrate(
    out_path: str,
    num_concepts: int = 30,
    num_triples: int = 80,
    seed: int | None = None,
    compound_fraction: float = 0.5,
    triples_per_concept_min: int = 1,
) -> dict:
    """Generate a random training-orthogonal Sara substrate.

    Args:
        out_path: where to write the resulting brain.db file.
        num_concepts: number of distinct concept labels to generate.
        num_triples: total number of (subject, relation, object) triples.
        seed: RNG seed; if None, uses current epoch time.
        compound_fraction: fraction of concepts that are multi-word compounds
            (preserves the test of compound-term retrieval).
        triples_per_concept_min: each concept appears in at least this many
            triples (prevents isolated-neuron substrates).

    Returns:
        dict with seed, paths, counts, and the canonical concept and
        triple lists. Also written to <out_path>.manifest.json.
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    rng = random.Random(seed)

    out = Path(out_path)
    if out.exists():
        raise FileExistsError(f"{out} exists; remove or rename before regenerating")

    # Generate the concept pool. Mix of single-word and compound labels.
    concepts: list[str] = []
    while len(concepts) < num_concepts:
        if rng.random() < compound_fraction:
            label = _random_compound(rng, n_words=rng.choice([2, 2, 3]))
        else:
            label = _random_word(rng)
        if label not in concepts:  # ensure distinct
            concepts.append(label)

    # Generate triples. Guarantee each concept appears in at least
    # triples_per_concept_min triples to avoid isolated-neuron drift.
    triples: list[tuple[str, str, str]] = []
    appearances: dict[str, int] = {c: 0 for c in concepts}

    def _emit_triple(s: str, r: str, o: str) -> None:
        triples.append((s, r, o))
        appearances[s] = appearances.get(s, 0) + 1
        appearances[o] = appearances.get(o, 0) + 1

    # First pass — guarantee minimum coverage by walking concepts in order
    for c in concepts:
        while appearances[c] < triples_per_concept_min:
            partner = rng.choice([x for x in concepts if x != c])
            r = rng.choice(_RELATIONS_POOL)
            if rng.random() < 0.5:
                _emit_triple(c, r, partner)
            else:
                _emit_triple(partner, r, c)
            if len(triples) >= num_triples:
                break
        if len(triples) >= num_triples:
            break

    # Second pass — fill remaining triples randomly
    while len(triples) < num_triples:
        s = rng.choice(concepts)
        o = rng.choice(concepts)
        if s == o:
            continue
        r = rng.choice(_RELATIONS_POOL)
        _emit_triple(s, r, o)

    # Teach into a fresh brain
    out.parent.mkdir(parents=True, exist_ok=True)
    brain = Brain(str(out))
    source_label = f"synthetic_seed_{seed}"
    for s, r, o in triples:
        brain.teach_triple(s, r, o, source_label=source_label)

    n_count = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    s_count = brain.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    p_count = brain.conn.execute("SELECT COUNT(*) FROM paths").fetchone()[0]

    manifest = {
        "schema_version": 1,
        "substrate_type": "synthetic",
        "seed": seed,
        "num_concepts": len(concepts),
        "num_triples": len(triples),
        "neurons_in_brain": n_count,
        "segments_in_brain": s_count,
        "paths_in_brain": p_count,
        "compound_fraction": compound_fraction,
        "relations_pool": _RELATIONS_POOL,
        "concepts": concepts,
        "triples": triples,
        "source_label": source_label,
    }

    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "db_path": str(out.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        **{k: v for k, v in manifest.items() if k != "triples" and k != "concepts"},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="Output .db path")
    ap.add_argument("--concepts", type=int, default=30,
                    help="Number of distinct concept labels (default: 30)")
    ap.add_argument("--triples", type=int, default=80,
                    help="Number of triples (default: 80)")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (default: time-based)")
    ap.add_argument("--compound-fraction", type=float, default=0.5,
                    help="Fraction of compound (multi-word) concepts (default: 0.5)")
    args = ap.parse_args()

    info = generate_synthetic_substrate(
        out_path=args.out,
        num_concepts=args.concepts,
        num_triples=args.triples,
        seed=args.seed,
        compound_fraction=args.compound_fraction,
    )

    print("Synthetic substrate generated.")
    for k, v in info.items():
        if isinstance(v, list) and len(v) > 6:
            print(f"  {k}: <{len(v)} entries; see manifest>")
        else:
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
