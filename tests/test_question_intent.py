"""Unit tests for the question-intent classifier."""
from __future__ import annotations

from sara_brain.cortex.question_intent import (
    QuestionIntent,
    apply_bias_to_edge_text,
    classify,
    rank_edges_by_bias,
    shape_for,
)


# ── classify() ────────────────────────────────────────────────────────


def test_classify_what_is_definitional():
    assert classify("what is a fulcrum") == QuestionIntent.DEFINITIONAL
    assert classify("what is the edubba") == QuestionIntent.DEFINITIONAL
    assert classify("who is einstein") == QuestionIntent.DEFINITIONAL
    assert classify("where is rome") == QuestionIntent.DEFINITIONAL


def test_classify_how_does_x_work_mechanism():
    # Verbatim fulcrum case — the bug that motivated the patch.
    assert classify("how does the fulcrum work") == QuestionIntent.MECHANISM
    assert classify("how does it work") == QuestionIntent.MECHANISM
    assert classify("how do enzymes function") == QuestionIntent.MECHANISM


def test_classify_why_mechanism():
    assert classify("why does this happen") == QuestionIntent.MECHANISM
    assert classify("why is rna folded") == QuestionIntent.MECHANISM


def test_classify_how_big_not_mechanism():
    # Manner-of-being guard — "how big" isn't asking for mechanism.
    assert classify("how big is the molecule") != QuestionIntent.MECHANISM
    assert classify("how tall is mount everest") != QuestionIntent.MECHANISM


def test_classify_tell_me_everything_associative():
    assert classify("tell me everything about rna") == QuestionIntent.ASSOCIATIVE
    assert classify("give me the complete picture of fulcrums") == QuestionIntent.ASSOCIATIVE
    assert classify("everything you know about aptamers") == QuestionIntent.ASSOCIATIVE


def test_classify_compound_takes_strongest():
    # MECHANISM beats DEFINITIONAL when both clauses are present.
    assert (
        classify("what is a fulcrum and how does it work")
        == QuestionIntent.MECHANISM
    )


def test_classify_no_wh_word_with_mechanism_noun():
    assert classify("fulcrum mechanism please") == QuestionIntent.MECHANISM
    assert classify("the rna folding process") == QuestionIntent.MECHANISM


def test_classify_unknown_fallback():
    # No WH-word, no mechanism-noun, no comprehensive phrase → UNKNOWN.
    assert classify("") == QuestionIntent.UNKNOWN
    assert classify("hmm") == QuestionIntent.UNKNOWN


def test_classify_relational():
    assert classify("which one is correct") == QuestionIntent.RELATIONAL
    assert classify("when did it happen") == QuestionIntent.RELATIONAL


# ── shape_for() ───────────────────────────────────────────────────────


def test_shape_for_mechanism_depth_2():
    s = shape_for(QuestionIntent.MECHANISM)
    assert s.depth == 2
    assert s.bidirectional is True
    # Substrate-observed mechanism relations must be in the bias set.
    assert "act_as" in s.relation_bias
    assert "causes" in s.relation_bias
    assert "enables" in s.relation_bias


def test_shape_for_definitional_depth_1():
    s = shape_for(QuestionIntent.DEFINITIONAL)
    assert s.depth == 1
    assert "is_a" in s.relation_bias
    assert "is" in s.relation_bias


def test_shape_for_associative_depth_3_no_bias():
    s = shape_for(QuestionIntent.ASSOCIATIVE)
    assert s.depth == 3
    assert s.relation_bias == frozenset()


def test_shape_for_unknown_is_default():
    s = shape_for(QuestionIntent.UNKNOWN)
    assert s.depth == 1
    assert s.relation_bias == frozenset()
    assert s.bidirectional is True


# ── rank_edges_by_bias() ──────────────────────────────────────────────


def test_rank_edges_by_bias_stable():
    edges = [
        {"relation": "is_a", "id": 1},
        {"relation": "act_as", "id": 2},
        {"relation": "is_a", "id": 3},
        {"relation": "causes", "id": 4},
    ]
    bias = frozenset({"act_as", "causes"})
    ranked = rank_edges_by_bias(edges, bias)
    # Bias-matched first, original-order preserved within each bucket.
    assert [e["id"] for e in ranked] == [2, 4, 1, 3]


def test_rank_edges_empty_bias_is_identity():
    edges = [
        {"relation": "is_a", "id": 1},
        {"relation": "act_as", "id": 2},
    ]
    assert rank_edges_by_bias(edges, frozenset()) == edges


def test_rank_edges_missing_relation_key_treated_as_non_biased():
    edges = [
        {"id": 1},  # no relation key
        {"relation": "act_as", "id": 2},
    ]
    bias = frozenset({"act_as"})
    ranked = rank_edges_by_bias(edges, bias)
    assert [e["id"] for e in ranked] == [2, 1]


# ── apply_bias_to_edge_text() ─────────────────────────────────────────


def test_apply_bias_to_edge_text_promotes_mechanism_edges():
    text = """Neighborhood of 'fulcrum'  (depth=2)

Edges (source --[relation]--> target):

  Discovered at depth 1:
    'fulcrum' --[is_a]--> 'thing'
    'fulcrum' --[part_of]--> 'lever'
    'static loops' --[act_as]--> 'fulcrum'
    'fulcrum' --[provides]--> 'pivot point'
"""
    bias = frozenset({"act_as", "provides"})
    out = apply_bias_to_edge_text(text, bias)
    # Pull only real edge lines (start with whitespace + a quoted source).
    lines = out.splitlines()
    edge_lines = [l for l in lines if l.startswith("    '")]
    assert "act_as" in edge_lines[0]
    assert "provides" in edge_lines[1]
    # is_a / part_of fall to the back, in original order.
    assert "is_a" in edge_lines[2]
    assert "part_of" in edge_lines[3]


def test_apply_bias_to_edge_text_empty_bias_is_identity():
    text = "  'a' --[is_a]--> 'b'\n  'c' --[causes]--> 'd'\n"
    assert apply_bias_to_edge_text(text, frozenset()) == text


def test_apply_bias_to_edge_text_handles_no_edges():
    text = "Sara doesn't have 'foo' as a neuron."
    assert apply_bias_to_edge_text(text, frozenset({"causes"})) == text


def test_apply_bias_preserves_section_headers():
    text = """Edges:

  Discovered at depth 1:
    'a' --[is_a]--> 'b'
    'a' --[causes]--> 'c'

  Discovered at depth 2:
    'd' --[is_a]--> 'e'
"""
    out = apply_bias_to_edge_text(text, frozenset({"causes"}))
    assert "Discovered at depth 1:" in out
    assert "Discovered at depth 2:" in out
    # Within depth-1 block, causes comes first.
    lines = out.splitlines()
    d1_idx = next(i for i, l in enumerate(lines) if "depth 1" in l)
    # First edge after the depth-1 header should be the causes edge.
    first_edge_after = next(
        l for l in lines[d1_idx:] if "--[" in l
    )
    assert "causes" in first_edge_after
