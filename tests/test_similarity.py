"""Tests for path similarity detection."""


class TestSimilarity:
    def test_analyze_finds_shared_paths(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        # red and round both reach apple (via different relation neurons)
        links = brain.analyze_similarity()
        # Should find similarity between red and round (both reach apple)
        assert len(links) >= 1
        labels = {(l.neuron_a_label, l.neuron_b_label) for l in links}
        # Order may vary
        assert ("red", "round") in labels or ("round", "red") in labels

    def test_get_similar(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        brain.analyze_similarity()

        similar = brain.get_similar("red")
        assert len(similar) >= 1
        other_labels = {l.neuron_b_label for l in similar}
        assert "round" in other_labels

    def test_no_similarity_unrelated(self, brain):
        brain.teach("apples are red")
        brain.teach("circles are round")
        # These don't share any downstream paths
        links = brain.analyze_similarity()
        # red reaches apple, round reaches circle — no shared endpoints
        shared_pairs = [(l.neuron_a_label, l.neuron_b_label) for l in links]
        assert ("red", "round") not in shared_pairs
        assert ("round", "red") not in shared_pairs
