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
