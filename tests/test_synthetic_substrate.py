"""Tests for the synthetic-substrate generator.

The generator's job is to produce training-orthogonal substrates for
instrument validation. The labels must be unlikely to be real words,
the same seed must produce the same substrate (reproducibility), and
the resulting brain must contain the expected triples.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

# Add the generator script path to sys.path so we can import it.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent
                       / "papers" / "instrument_validation"))
from generate_synthetic_substrate import generate_synthetic_substrate, _random_word

import random


def test_random_word_is_pronounceable_and_not_real():
    """A handful of generated words shouldn't collide with common English."""
    rng = random.Random(0)
    common = {
        "the", "and", "but", "for", "you", "all", "any", "are", "can", "had",
        "has", "her", "him", "his", "not", "now", "one", "our", "out", "she",
        "two", "use", "way", "who", "why", "yes", "had", "way", "let", "set",
        "rake", "vine", "pine", "lemon", "alpha", "beta", "delta",
    }
    words = [_random_word(rng) for _ in range(50)]
    collisions = [w for w in words if w in common]
    assert collisions == [], f"Random words collided with common English: {collisions}"


def test_generator_produces_expected_counts():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "test.db"
        info = generate_synthetic_substrate(
            out_path=str(out),
            num_concepts=10,
            num_triples=20,
            seed=12345,
        )
        assert info["num_concepts"] == 10
        assert info["num_triples"] == 20
        # Generator may create more neurons than concepts because
        # _link_sub_concepts decomposes multi-word labels into bare-word
        # neurons too.
        assert info["neurons_in_brain"] >= 10
        assert info["paths_in_brain"] == 20


def test_generator_writes_manifest():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "test.db"
        generate_synthetic_substrate(
            out_path=str(out),
            num_concepts=8,
            num_triples=15,
            seed=99,
        )
        manifest_path = Path(str(out) + ".manifest.json")
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["substrate_type"] == "synthetic"
        assert manifest["seed"] == 99
        assert len(manifest["concepts"]) == 8
        assert len(manifest["triples"]) == 15


def test_generator_is_reproducible_with_same_seed():
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        out1 = Path(td1) / "a.db"
        out2 = Path(td2) / "b.db"
        info1 = generate_synthetic_substrate(out_path=str(out1), num_concepts=12,
                                             num_triples=30, seed=42)
        info2 = generate_synthetic_substrate(out_path=str(out2), num_concepts=12,
                                             num_triples=30, seed=42)
        # Read both manifests and compare triples
        m1 = json.loads(Path(str(out1) + ".manifest.json").read_text())
        m2 = json.loads(Path(str(out2) + ".manifest.json").read_text())
        assert m1["concepts"] == m2["concepts"]
        assert m1["triples"] == m2["triples"]


def test_generator_refuses_to_overwrite():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "test.db"
        generate_synthetic_substrate(out_path=str(out), num_concepts=5,
                                     num_triples=10, seed=1)
        try:
            generate_synthetic_substrate(out_path=str(out), num_concepts=5,
                                         num_triples=10, seed=1)
        except FileExistsError:
            return
        assert False, "Should have refused to overwrite existing .db"
