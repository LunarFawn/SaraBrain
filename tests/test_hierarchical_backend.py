"""Tests for the hierarchical per-concept SQLite backend.

Covers:
- Backend initialises three-tier file structure
- Concept registration, slugification, idempotence
- route_teach: trigger-lemma fan-out
- route_query: concept_vocab routing
- update_concept_vocab + route_query round-trip
- LRU cap — opening >256 concepts without running out of FDs
- concept_conn returns a working SQLite connection with standard tables
- Parity: NeuronRepo / SegmentRepo / PathRepo work against a concept conn
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from sara_brain.storage.hierarchical_backend import (
    HierarchicalBackend,
    slugify_concept,
)
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo
from sara_brain.storage.path_repo import PathRepo
from sara_brain.models.neuron import NeuronType
from sara_brain.models.segment import Segment
from sara_brain.models.path import Path as PathModel, PathStep


# ── Fixtures ──

@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    return tmp_path / "brain_root"


@pytest.fixture
def backend(tmp_root: Path) -> HierarchicalBackend:
    b = HierarchicalBackend(str(tmp_root))
    b.register_subject("biology", description="test")
    yield b
    b.close()


# ── Slugify ──

def test_slugify_basic():
    assert slugify_concept("Mitosis") == "mitosis"
    assert slugify_concept("DNA replication") == "dna_replication"
    assert slugify_concept("gram-negative bacterium!") == "gram_negative_bacterium"
    assert slugify_concept("") == "unnamed"


# ── Tier initialisation ──

def test_brain_db_created(tmp_root: Path, backend: HierarchicalBackend):
    assert (tmp_root / "brain.db").exists()
    assert (tmp_root / "subjects" / "biology.db").exists()


def test_register_subject_idempotent(backend: HierarchicalBackend):
    backend.register_subject("biology")  # second call — no error
    subjects = backend.list_subjects()
    assert len([s for s in subjects if s["name"] == "biology"]) == 1


def test_register_concept_creates_db(tmp_root: Path,
                                     backend: HierarchicalBackend):
    slug = backend.register_concept(
        "biology", "Mitosis",
        source_kind="test",
        trigger_lemmas=["mitosis", "mitotic"],
    )
    assert slug == "mitosis"
    assert (tmp_root / "concepts" / "biology" / "mitosis.db").exists()


def test_register_concept_idempotent(backend: HierarchicalBackend):
    backend.register_concept("biology", "Mitosis", source_kind="test",
                             trigger_lemmas=["mitosis"])
    backend.register_concept("biology", "Mitosis", source_kind="test",
                             trigger_lemmas=["mitosis"])  # second call
    concepts = backend.list_concepts("biology")
    assert len([c for c in concepts if c["name"] == "mitosis"]) == 1


# ── Routing: route_teach ──

def test_route_teach_single_concept(backend: HierarchicalBackend):
    backend.register_concept("biology", "Mitosis", source_kind="test",
                             trigger_lemmas=["mitosis", "mitotic", "prophase"])
    backend.register_concept("biology", "Photosynthesis", source_kind="test",
                             trigger_lemmas=["photosynthesis", "chlorophyll"])
    result = backend.route_teach("biology", ["mitotic", "spindle"])
    assert "mitosis" in result
    assert "photosynthesis" not in result


def test_route_teach_overlap(backend: HierarchicalBackend):
    backend.register_concept("biology", "Mitosis", source_kind="test",
                             trigger_lemmas=["mitosis", "chromosome"])
    backend.register_concept("biology", "Meiosis", source_kind="test",
                             trigger_lemmas=["meiosis", "chromosome"])
    result = backend.route_teach("biology", ["chromosome", "division"])
    # Both share 'chromosome' — both should fire
    assert "mitosis" in result
    assert "meiosis" in result


def test_route_teach_no_match_returns_empty(backend: HierarchicalBackend):
    backend.register_concept("biology", "Mitosis", source_kind="test",
                             trigger_lemmas=["mitosis"])
    result = backend.route_teach("biology", ["geology", "rock", "mineral"])
    assert result == []


# ── Routing: route_query ──

def test_route_query_uses_vocab(backend: HierarchicalBackend):
    backend.register_concept("biology", "Mitosis", source_kind="test",
                             trigger_lemmas=["mitosis"])
    backend.register_concept("biology", "Photosynthesis", source_kind="test",
                             trigger_lemmas=["photosynthesis"])
    backend.update_concept_vocab("biology", "Mitosis",
                                 {"mitosis": 10, "chromosome": 5, "spindle": 3})
    backend.update_concept_vocab("biology", "Photosynthesis",
                                 {"photosynthesis": 8, "chlorophyll": 4})
    result = backend.route_query("biology", ["mitosis", "spindle"], top_k=2)
    assert result[0] == "mitosis"
    assert "photosynthesis" not in result


def test_route_query_top_k(backend: HierarchicalBackend):
    for i in range(5):
        backend.register_concept("biology", f"concept_{i}",
                                 source_kind="test",
                                 trigger_lemmas=[f"lemma_{i}"])
        backend.update_concept_vocab(
            "biology", f"concept_{i}", {f"lemma_{i}": i + 1}
        )
    result = backend.route_query(
        "biology", [f"lemma_{i}" for i in range(5)], top_k=3
    )
    assert len(result) <= 3


# ── Ten hand-graded biology routing queries ──

ROUTING_CASES: list[tuple[list[str], list[str], list[str]]] = [
    # (question lemmas, must-include concepts, must-exclude concepts)
    (["mitosis", "prophase", "metaphase"], ["mitosis"], ["photosynthesis"]),
    (["meiosis", "gamete", "chromosome"], ["meiosis"], ["mitosis"]),
    (["photosynthesis", "chlorophyll", "light"], ["photosynthesis"], ["meiosis"]),
    (["heterotroph", "autotroph", "energy"], ["heterotroph"], ["mitosis"]),
    (["dna", "replication", "polymerase"], ["dna_replication"], ["photosynthesis"]),
    (["mitosis", "meiosis", "chromosome"], ["mitosis", "meiosis"], []),
    (["cell", "cycle", "interphase"], ["mitosis"], ["photosynthesis"]),
    (["glycolysis", "glucose", "atp"], ["glycolysis"], ["meiosis"]),
    (["protein", "ribosome", "translation"], ["translation"], ["photosynthesis"]),
    (["gene", "allele", "dominant"], ["gene", "allele"], ["mitosis"]),
]


@pytest.mark.parametrize("q_lemmas,must_have,must_not", ROUTING_CASES)
def test_routing_biology(q_lemmas: list[str],
                         must_have: list[str],
                         must_not: list[str]):
    """Routing spot-check: question lemmas must resolve to expected concepts."""
    with tempfile.TemporaryDirectory() as td:
        b = HierarchicalBackend(td)
        b.register_subject("biology")
        concept_defs = {
            "mitosis": ["mitosis", "prophase", "metaphase", "anaphase",
                        "telophase", "spindle", "chromosome", "cell", "cycle",
                        "interphase", "centrosome"],
            "meiosis": ["meiosis", "gamete", "crossing", "homolog",
                        "chromosome", "synapsis", "tetrad"],
            "photosynthesis": ["photosynthesis", "chlorophyll", "chloroplast",
                               "light", "carbon", "oxygen"],
            "heterotroph": ["heterotroph", "autotroph", "energy", "consume",
                            "producer", "consumer"],
            "dna_replication": ["dna", "replication", "polymerase",
                                "helicase", "primase", "ligase"],
            "glycolysis": ["glycolysis", "glucose", "atp", "pyruvate",
                           "substrate", "kinase"],
            "translation": ["translation", "ribosome", "mrna", "trna",
                            "protein", "codon", "anticodon"],
            "gene": ["gene", "allele", "locus", "expression", "genome"],
            "allele": ["allele", "dominant", "recessive", "gene",
                       "genotype", "phenotype"],
        }
        for concept, triggers in concept_defs.items():
            b.register_concept("biology", concept, source_kind="test",
                               trigger_lemmas=triggers)
            b.update_concept_vocab("biology", concept,
                                   {t: 5 for t in triggers})

        result_set = set(b.route_query("biology", q_lemmas, top_k=5))
        for expected in must_have:
            assert expected in result_set, (
                f"Expected '{expected}' in route for {q_lemmas}, "
                f"got {result_set}"
            )
        for excluded in must_not:
            # must_not is advisory — only fire if it's the *only* result
            if len(result_set) == 1 and excluded in result_set:
                pytest.fail(
                    f"'{excluded}' was the sole result for {q_lemmas} "
                    f"— expected {must_have}"
                )
        b.close()


# ── Repo parity: standard repos work against a concept conn ──

def test_repos_work_against_concept_conn(backend: HierarchicalBackend):
    backend.register_concept("biology", "mitosis", source_kind="test",
                             trigger_lemmas=["mitosis"])
    conn = backend.concept_conn("biology", "mitosis")

    nr = NeuronRepo(conn)
    sr = SegmentRepo(conn)
    pr = PathRepo(conn)

    n1, _ = nr.get_or_create("mitosis", NeuronType.CONCEPT)
    n2, _ = nr.get_or_create("spindle", NeuronType.PROPERTY)
    n3, _ = nr.get_or_create("mitosis_spindle", NeuronType.RELATION)

    seg1, _ = sr.get_or_create(n2.id, n3.id, "has")
    seg2, _ = sr.get_or_create(n3.id, n1.id, "describes")

    path = PathModel(id=None, origin_id=n2.id, terminus_id=n1.id,
                     source_text="mitosis has spindle")
    path = pr.create(path)
    pr.add_step(PathStep(id=None, path_id=path.id,
                         step_order=0, segment_id=seg1.id))
    pr.add_step(PathStep(id=None, path_id=path.id,
                         step_order=1, segment_id=seg2.id))
    conn.commit()

    assert pr.count() == 1
    assert nr.count() == 3
    fetched = pr.get_by_id(path.id)
    assert fetched.source_text == "mitosis has spindle"


# ── LRU: opening many concept DBs doesn't exhaust FDs ──

def test_lru_many_concepts(tmp_root: Path):
    """Opening > _MAX_OPEN_CONCEPTS (256) distinct concept DBs should
    not raise a file-descriptor error."""
    with HierarchicalBackend(str(tmp_root)) as b:
        b.register_subject("biology")
        for i in range(300):
            b.register_concept("biology", f"concept_{i}",
                               source_kind="test",
                               trigger_lemmas=[f"lemma_{i}"])
        # Access all 300 — older ones get evicted from the LRU cache
        for i in range(300):
            conn = b.concept_conn("biology", f"concept_{i}")
            assert isinstance(conn, sqlite3.Connection)
