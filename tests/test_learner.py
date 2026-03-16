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
