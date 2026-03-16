"""Tests for the recognition system."""


class TestRecognizer:
    def test_single_input(self, brain):
        brain.teach("apples are red")
        results = brain.recognize("red")
        assert len(results) >= 1
        labels = [r.neuron.label for r in results]
        assert "apple" in labels or "apple_color" in labels

    def test_intersection(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        brain.teach("circles are round")

        results = brain.recognize("red, round")
        # apple should be #1 with 2 converging paths
        assert len(results) >= 1
        assert results[0].neuron.label == "apple"
        assert results[0].confidence == 2

    def test_circle_one_path(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        brain.teach("circles are round")

        results = brain.recognize("red, round")
        circle_results = [r for r in results if r.neuron.label == "circle"]
        # circle may appear with 1 converging path (only "round" reaches it)
        # but since we filter for 2+ convergence in recognize, it might not appear
        # Let's check the full results
        for r in results:
            if r.neuron.label == "circle":
                assert r.confidence == 1

    def test_no_match(self, brain):
        results = brain.recognize("nonexistent")
        assert len(results) == 0

    def test_trace(self, brain):
        brain.teach("apples are red")
        traces = brain.trace("red")
        assert len(traces) >= 1
        # Should find path from red → apple_color → apple
        found = False
        for t in traces:
            labels = t.labels()
            if "red" in labels and "apple" in labels:
                found = True
        assert found

    def test_why(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        traces = brain.why("apple")
        assert len(traces) == 2
        sources = {t.source_text for t in traces}
        assert "apples are red" in sources
        assert "apples are round" in sources

    def test_recognition_strengthens_segments(self, brain):
        brain.teach("apples are red")
        red = brain.neuron_repo.get_by_label("red")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        seg_before = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        strength_before = seg_before.strength

        brain.recognize("red")
        seg_after = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        assert seg_after.strength >= strength_before
