"""Tests for the learning system."""


class TestLearner:
    def test_teach_creates_path(self, brain):
        result = brain.teach("apples are red")
        assert result is not None
        assert "red" in result.path_label
        assert "apple" in result.path_label
        assert result.path_id is not None

    def test_teach_creates_neurons(self, brain):
        brain.teach("apples are red")
        assert brain.neuron_repo.get_by_label("red") is not None
        assert brain.neuron_repo.get_by_label("apple") is not None
        assert brain.neuron_repo.get_by_label("apple_color") is not None

    def test_teach_creates_segments(self, brain):
        brain.teach("apples are red")
        red = brain.neuron_repo.get_by_label("red")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        apple = brain.neuron_repo.get_by_label("apple")

        seg1 = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        assert seg1 is not None
        seg2 = brain.segment_repo.find(apple_color.id, apple.id, "describes")
        assert seg2 is not None

    def test_teach_reuses_neurons(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        # "apple" should exist once
        assert brain.neuron_repo.count() == 5  # red, round, apple, apple_color, apple_shape

    def test_teach_strengthens_existing_segments(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are red")  # same thing again
        red = brain.neuron_repo.get_by_label("red")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        seg = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        assert seg.traversals == 1  # strengthened once (second teach)
        assert seg.strength > 1.0

    def test_teach_unparseable(self, brain):
        result = brain.teach("hello")
        assert result is None

    def test_teach_records_path_steps(self, brain):
        result = brain.teach("apples are red")
        steps = brain.path_repo.get_steps(result.path_id)
        assert len(steps) == 2
        assert steps[0].step_order == 0
        assert steps[1].step_order == 1

    # ── First-class negation ──────────────────────────────────────────
    # The parser populates ParsedStatement.negated for "X is not Y" and
    # "X does not Y" patterns. learn() routes negated facts through the
    # same refutation storage path that brain.refute()/unlearn() use.

    def test_teach_negation_copula_refutes(self, brain):
        """Teaching "X is not Y" must refute, not affirm."""
        brain.teach("apples are red")
        brain.teach("apples are not red")
        red = brain.neuron_repo.get_by_label("red")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        seg = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        assert seg is not None
        assert seg.refutations > 0, (
            "parsed.negated must flow into _build_chain(refute=True)"
        )

    def test_teach_negation_does_not_duplicate_segment(self, brain):
        """Teach positive then negative — segment is weakened, not duplicated."""
        brain.teach("apples are red")
        brain.teach("apples are not red")
        red = brain.neuron_repo.get_by_label("red")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        # Still exactly one segment between them for the same relation;
        # negation modifies counters, doesn't create a parallel path.
        seg = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        assert seg is not None
        # And its strength is lower than an uncontested affirmation.
        assert seg.strength < 1.0 + 1e-9

    def test_teach_negation_fresh_fact_still_refutes(self, brain):
        """Teaching "X is not Y" with no prior positive still writes a
        refutation — creates segments and weakens them, so the epistemic
        state is 'refuted' rather than 'unknown'."""
        brain.teach("apples are not blue")
        blue = brain.neuron_repo.get_by_label("blue")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        seg = brain.segment_repo.find(blue.id, apple_color.id, "has_color")
        assert seg is not None
        assert seg.refutations > 0
