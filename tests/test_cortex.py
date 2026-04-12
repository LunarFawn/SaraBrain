"""Tests for the Sara Cortex — language layer over Sara Brain."""

from __future__ import annotations

import pytest

from sara_brain.cortex import Cortex, EnhancedParser, TurnKind
from sara_brain.cortex.generator import TemplateGenerator
from sara_brain.cortex.training.synthesize import synthesize


# ── Parser tests ──


def test_parser_detects_questions():
    p = EnhancedParser()
    for q in [
        "what is the edubba",
        "what is the edubba?",
        "who was sumerian",
        "tell me about apples",
        "describe rna",
        "is the apple red?",
    ]:
        result = p.parse(q)
        assert result.kind == TurnKind.QUESTION, f"failed for {q!r}"
        assert result.topics, f"no topics extracted for {q!r}"


def test_parser_detects_statements():
    p = EnhancedParser()
    for s in [
        "apples are red",
        "the edubba was a sumerian school",
        "rna is a molecule",
    ]:
        result = p.parse(s)
        assert result.kind == TurnKind.STATEMENT, f"failed for {s!r}"
        assert len(result.facts) == 1
        assert not result.facts[0].negated


def test_parser_detects_negation():
    p = EnhancedParser()
    for s in [
        "the edubba was not akkadian",
        "apples are not blue",
        "rna is not protein",
    ]:
        result = p.parse(s)
        assert result.kind == TurnKind.NEGATION, f"failed for {s!r}"
        assert len(result.facts) == 1
        assert result.facts[0].negated


def test_parser_handles_compound_statements():
    p = EnhancedParser()
    result = p.parse("apples are red and oranges are orange")
    assert result.kind == TurnKind.STATEMENT
    # Both halves should parse
    assert len(result.facts) >= 1  # at least one half


def test_parser_handles_greetings():
    p = EnhancedParser()
    for g in ["hello", "hi there", "good morning"]:
        result = p.parse(g)
        assert result.kind == TurnKind.GREETING


def test_parser_extracts_topics_from_questions():
    p = EnhancedParser()
    result = p.parse("what is the edubba")
    assert "edubba" in result.topics


def test_parser_extracts_source():
    p = EnhancedParser()
    src, body = p._extract_source("according to wikipedia, the edubba was a school")
    assert src == "wikipedia"
    assert "edubba" in body.lower()


def test_parser_handles_quantifiers():
    p = EnhancedParser()
    high = p._extract_quantifier("apples are definitely red")
    low = p._extract_quantifier("apples are maybe red")
    assert high == 1.0
    assert low == 0.5


# ── Cortex / Brain integration tests ──


def test_cortex_handles_question_with_no_knowledge(brain):
    cortex = Cortex(brain)
    response = cortex.process("what is the edubba")
    assert "no knowledge" in response.text.lower() or "don't know" in response.text.lower() or "tell me" in response.text.lower()
    assert not response.delegate  # honest no-knowledge is high confidence
    assert response.confidence >= 0.5


def test_cortex_teaches_statement(brain):
    cortex = Cortex(brain)
    response = cortex.process("apples are red")
    assert "learned" in response.text.lower()
    # Verify Sara actually has the fact
    red = brain.neuron_repo.get_by_label("red")
    assert red is not None


def test_cortex_refutes_negation(brain):
    cortex = Cortex(brain)
    cortex.process("apples are blue")  # teach a wrong fact
    response = cortex.process("apples are not blue")  # refute it
    assert "false" in response.text.lower() or "refuted" in response.text.lower() or "marked" in response.text.lower()


def test_cortex_query_after_teach(brain):
    cortex = Cortex(brain)
    cortex.process("the edubba was a school")
    response = cortex.process("what is the edubba")
    assert "school" in response.text.lower() or "edubba" in response.text.lower()
    assert not response.delegate  # cortex handled it directly


def test_cortex_full_correction_flow(brain):
    """End-to-end: teach wrong, refute, teach right, query."""
    cortex = Cortex(brain)
    cortex.process("the edubba was for akkadian")
    cortex.process("the edubba was not for akkadian")
    cortex.process("the edubba was for sumerian")
    response = cortex.process("what is the edubba")
    assert "edubba" in response.text.lower() or "sumerian" in response.text.lower()


def test_cortex_greeting(brain):
    cortex = Cortex(brain)
    response = cortex.process("hello")
    assert response.confidence == 1.0
    assert not response.delegate


def test_cortex_records_operations(brain):
    cortex = Cortex(brain)
    response = cortex.process("apples are red")
    assert len(response.operations) >= 1
    assert response.operations[0].op == "teach"
    assert response.operations[0].success


# ── Generator tests ──


def test_generator_renders_no_knowledge():
    g = TemplateGenerator()
    text = g.no_knowledge("edubba")
    assert "edubba" in text
    assert "no knowledge" in text.lower() or "tell me" in text.lower()


def test_generator_confirm_taught():
    g = TemplateGenerator()
    text = g.confirm_taught("apples are red")
    assert "learned" in text.lower() or "apples are red" in text.lower()


# ── Synthesize tests ──


def test_synthesize_generates_examples():
    examples = synthesize(count=20)
    assert len(examples) == 20
    for ex in examples:
        assert "input" in ex
        assert "kind" in ex
        assert ex["kind"] in ("teach", "refute", "question")


def test_synthesize_balanced_kinds():
    examples = synthesize(count=200, seed=1)
    kinds = [ex["kind"] for ex in examples]
    teach_count = kinds.count("teach")
    # Should be roughly 60% teach
    assert 100 < teach_count < 160


# ── Disambiguation tests ──


def test_disambiguation_fires_on_close_typo(brain):
    """Teaching 'childen' when Sara knows 'children' should require disambiguation."""
    # Pre-populate brain with a well-established neuron
    cortex = Cortex(brain)
    cortex.process("children are young")
    cortex.process("children are humans")
    cortex.process("children are small")

    # Now try to teach a close-spelled (1 edit) typo
    response = cortex.process("childen are happy")

    # Should require disambiguation because of edit-distance match
    assert response.requires_disambiguation
    assert len(response.ambiguities) > 0
    assert "childen" in response.text.lower() or "children" in response.text.lower()


def test_disambiguation_ignored_for_exact_match(brain):
    """Teaching the SAME term as an existing neuron is fine — no ambiguity."""
    cortex = Cortex(brain)
    cortex.process("apples are red")
    response = cortex.process("apples are sweet")
    # 'apples' and 'red' both already exist or are simple — no ambiguity needed
    # 'sweet' is new but has no close match in the small brain
    assert not response.requires_disambiguation


def test_disambiguation_skipped_for_unrelated_terms(brain):
    """Teaching a totally new concept with no close matches just commits."""
    cortex = Cortex(brain)
    response = cortex.process("xylophones are musical instruments")
    assert not response.requires_disambiguation


# ── Cluster / association tests ──


def test_bare_word_triggers_association(brain):
    """A single word with no verb should be classified as ASSOCIATION."""
    from sara_brain.cortex.parser import EnhancedParser, TurnKind
    p = EnhancedParser()
    r = p.parse("edubba")
    assert r.kind == TurnKind.ASSOCIATION
    assert "edubba" in r.topics


def test_explicit_association_phrase(brain):
    """'what is associated with X' triggers ASSOCIATION."""
    from sara_brain.cortex.parser import EnhancedParser, TurnKind
    p = EnhancedParser()
    r = p.parse("what is associated with sumerians")
    assert r.kind == TurnKind.ASSOCIATION
    assert "sumerians" in r.topics


def test_question_does_not_trigger_association(brain):
    """'what is the edubba' is a question, not an association request."""
    from sara_brain.cortex.parser import EnhancedParser, TurnKind
    p = EnhancedParser()
    r = p.parse("what is the edubba")
    assert r.kind == TurnKind.QUESTION


def test_brain_cluster_around(brain):
    """cluster_around returns connected neurons ranked by edge count."""
    brain.teach("apples are red")
    brain.teach("apples are sweet")
    brain.teach("apples are round")
    cluster = brain.cluster_around("apple")
    assert len(cluster) > 0
    labels = [c["label"] for c in cluster]
    assert any("red" in l or "sweet" in l or "round" in l or "apple" in l for l in labels)


def test_cortex_association_returns_cluster(brain):
    """Bare-word input through the cortex returns cluster output."""
    cortex = Cortex(brain)
    cortex.process("apples are red")
    cortex.process("apples are sweet")
    cortex.process("apples are round")

    response = cortex.process("apple")
    assert "associated" in response.text.lower() or "connected" in response.text.lower()
    assert not response.delegate


def test_disambiguation_does_not_commit_when_required(brain):
    """When disambiguation is required, no path should be created yet."""
    cortex = Cortex(brain)
    cortex.process("children are young")
    cortex.process("children are humans")
    cortex.process("children are small")
    initial_paths = brain.path_repo.count()

    response = cortex.process("childen are happy")
    if response.requires_disambiguation:
        # No new path should have been committed
        new_count = brain.path_repo.count()
        assert new_count == initial_paths
        # And no childen neuron should exist yet
        assert brain.neuron_repo.get_by_label("childen") is None
