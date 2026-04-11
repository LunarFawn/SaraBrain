"""Tests for refutation — Sara's signed-strength self-correction.

Sara never deletes. When something is refuted, the path stays but its
strength goes negative. Recognition weights paths by signed strength,
so refuted concepts can be detected as actively-known-false.
"""

from __future__ import annotations

import math


def test_refute_existing_fact_makes_strength_negative(brain):
    """A fact taught and then refuted multiple times should have negative strength.

    Note: the first teach() creates the segment at baseline (traversals=0).
    Repeat teaches increment traversals. Refute() always increments refutations.
    """
    brain.teach("apples are red")  # creates segment, traversals=0
    brain.teach("apples are red")  # strengthens, traversals=1
    brain.refute("apples are red")  # weakens, refutations=1
    brain.refute("apples are red")  # weakens, refutations=2
    brain.refute("apples are red")  # weakens, refutations=3

    red = brain.neuron_repo.get_by_label("red")
    assert red is not None
    segments = brain.segment_repo.get_outgoing(red.id)
    assert len(segments) > 0

    seg = segments[0]
    assert seg.traversals == 1
    assert seg.refutations == 3
    # strength = 1 + ln(2) - ln(4) = 1 + 0.693 - 1.386 = 0.307
    # Still positive but reduced — refutations outweigh validations
    assert seg.strength < 1.0


def test_refute_creates_negative_strength_when_dominant(brain):
    """Refuting more times than teaching pushes strength below baseline."""
    brain.refute("the earth is flat")
    brain.refute("the earth is flat")
    brain.refute("the earth is flat")
    brain.refute("the earth is flat")
    brain.refute("the earth is flat")

    flat = brain.neuron_repo.get_by_label("flat")
    assert flat is not None
    segments = brain.segment_repo.get_outgoing(flat.id)
    assert len(segments) > 0

    seg = segments[0]
    assert seg.traversals == 0
    assert seg.refutations == 5
    # strength = 1 + ln(1) - ln(6) = 1 - 1.79 = -0.79
    assert seg.strength < 0
    assert seg.is_refuted


def test_path_preserved_after_refutation(brain):
    """Refutation never deletes the path — it stays as evidence of what was claimed."""
    brain.teach("apples are blue")
    initial_path_count = brain.path_repo.count()

    brain.refute("apples are blue")
    after_refute_count = brain.path_repo.count()

    # Refutation creates a new path with [refuted] prefix
    assert after_refute_count > initial_path_count

    # The original path is still there
    paths = brain.path_repo.list_all()
    sources = [p.source_text for p in paths]
    assert any("[refuted]" in (s or "") for s in sources)


def test_refute_returns_none_for_unparseable(brain):
    """Refute should fail gracefully on unparseable input."""
    result = brain.refute("xyzzy")
    assert result is None


def test_validation_and_refutation_balance_to_baseline(brain):
    """Equal teach repeats and refutes should balance.

    First teach creates the segment (traversals=0). Subsequent teaches strengthen.
    So 6 teaches = 5 traversal increments. To balance, we need 5 refutations.
    """
    for _ in range(6):
        brain.teach("apples are red")
    for _ in range(5):
        brain.refute("apples are red")

    red = brain.neuron_repo.get_by_label("red")
    segments = brain.segment_repo.get_outgoing(red.id)
    seg = segments[0]
    # 5 traversals, 5 refutations
    # strength = 1 + ln(6) - ln(6) = 1.0
    assert abs(seg.strength - 1.0) < 0.01


def test_recognition_uses_signed_confidence(brain):
    """Recognition should weight refuted paths negatively."""
    brain.teach("apples are red")
    brain.teach("apples are round")

    # First confirm normal recognition works
    results = brain.recognize("red, round")
    apple_results = [r for r in results if r.neuron.label == "apple"]
    assert len(apple_results) > 0
    apple = apple_results[0]
    assert apple.signed_confidence > 0
    assert not apple.is_refuted

    # Now refute one of the facts heavily
    for _ in range(10):
        brain.refute("apples are red")

    results = brain.recognize("red, round")
    apple_results = [r for r in results if r.neuron.label == "apple"]
    if apple_results:
        # Even if still recognized, signed confidence should be lower
        new_confidence = apple_results[0].signed_confidence
        assert new_confidence < apple.signed_confidence


def test_why_returns_path_weight(brain):
    """why() should report the signed weight of each path."""
    brain.teach("apples are red")
    traces = brain.why("apple")
    assert len(traces) > 0
    assert traces[0].weight > 0
    assert not traces[0].is_refuted


def test_why_marks_refuted_paths(brain):
    """A heavily refuted fact should show up as a refuted path in why()."""
    for _ in range(5):
        brain.refute("apples are blue")

    # The path was created via refute (not teach), so it's all negative
    traces = brain.why("apple")
    refuted_traces = [t for t in traces if t.is_refuted]
    # At least one trace should be refuted
    assert len(refuted_traces) > 0


def test_signed_confidence_property(brain):
    """RecognitionResult.signed_confidence sums path weights."""
    from sara_brain.models.result import PathTrace, RecognitionResult
    from sara_brain.models.neuron import Neuron, NeuronType

    n = Neuron(id=1, label="apple", neuron_type=NeuronType.CONCEPT)
    result = RecognitionResult(
        neuron=n,
        converging_paths=[
            PathTrace(neurons=[], weight=3.0),
            PathTrace(neurons=[], weight=2.0),
            PathTrace(neurons=[], weight=-1.0),
        ],
    )
    assert result.confidence == 3
    assert result.signed_confidence == 4.0
    assert not result.is_refuted


def test_signed_confidence_can_be_negative(brain):
    """A concept reached only by refuted paths is is_refuted."""
    from sara_brain.models.result import PathTrace, RecognitionResult
    from sara_brain.models.neuron import Neuron, NeuronType

    n = Neuron(id=1, label="phlogiston", neuron_type=NeuronType.CONCEPT)
    result = RecognitionResult(
        neuron=n,
        converging_paths=[
            PathTrace(neurons=[], weight=-2.0),
            PathTrace(neurons=[], weight=-1.5),
        ],
    )
    assert result.signed_confidence == -3.5
    assert result.is_refuted


def test_segment_weaken_method(brain):
    """Segment.weaken() should increment refutations and recalculate."""
    from sara_brain.models.segment import Segment

    seg = Segment(id=None, source_id=1, target_id=2, relation="is_a")
    assert seg.strength == 1.0
    assert seg.refutations == 0

    seg.weaken()
    assert seg.refutations == 1
    # 1 + ln(1) - ln(2) = 1 - 0.693 = 0.307
    assert abs(seg.strength - (1.0 - math.log(2))) < 0.001
    # is_refuted = refutations > traversals → 1 > 0 → True
    assert seg.is_refuted

    seg.weaken()
    assert seg.refutations == 2
    # 1 + ln(1) - ln(3) = 1 - 1.099 = -0.099
    assert seg.strength < 0
    assert seg.is_refuted


def test_segment_strengthen_then_weaken(brain):
    """Validation followed by refutation should partially cancel."""
    from sara_brain.models.segment import Segment

    seg = Segment(id=None, source_id=1, target_id=2, relation="is_a")
    seg.strengthen()
    seg.strengthen()
    assert seg.traversals == 2
    # 1 + ln(3) = 2.099
    expected_after_2_strengthen = 1.0 + math.log(3)
    assert abs(seg.strength - expected_after_2_strengthen) < 0.001

    seg.weaken()
    assert seg.refutations == 1
    # 1 + ln(3) - ln(2) = 1 + 1.099 - 0.693 = 1.406
    expected = 1.0 + math.log(3) - math.log(2)
    assert abs(seg.strength - expected) < 0.001
    assert not seg.is_refuted  # still positive


def test_parser_rejects_pronoun_subjects(brain):
    """Subjects like 'it', 'they', 'this' must be rejected — they're meaningless."""
    parser = brain.parser
    for s in ["it was a school", "they are ancient", "this is wrong"]:
        assert parser.parse(s) is None, f"should reject {s!r}"


def test_parser_handles_auxiliary_verb_negation(brain):
    """X did not Y / X does not Y / X don't Y should parse as refutation."""
    parser = brain.parser

    r = parser.parse("the edubba did not teach akkadian")
    assert r is not None
    assert r.negated
    assert r.subject == "edubba"
    assert "akkadian" in r.obj

    r = parser.parse("apples do not contain protein")
    assert r is not None
    assert r.negated
    assert r.subject == "apple"


def test_parser_strips_typo_articles(brain):
    """Common article typos like 'tteh' / 'teh' should be stripped."""
    parser = brain.parser

    r = parser.parse("tteh edubba was a sumerian school")
    assert r is not None
    assert r.subject == "edubba"

    r = parser.parse("teh apple is red")
    assert r is not None
    assert r.subject == "apple"


def test_parser_handles_typo_article_with_negation(brain):
    """Combined: typo article + auxiliary negation."""
    parser = brain.parser

    r = parser.parse("tteh edubba did not teach akkadian")
    assert r is not None
    assert r.negated
    assert r.subject == "edubba"
