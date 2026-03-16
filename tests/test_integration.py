"""End-to-end integration tests: teach → recognize → trace → similarity."""

import os
import tempfile

from sara_brain.core.brain import Brain
from sara_brain.visualization.text_tree import render_paths_from, render_graph_dot


class TestEndToEnd:
    def test_apple_vs_circle(self, brain):
        """The canonical example from the spec."""
        brain.teach("apples are red")
        brain.teach("apples are round")
        brain.teach("circles are round")

        results = brain.recognize("red, round")

        # Apple should win with 2 converging paths
        assert results[0].neuron.label == "apple"
        assert results[0].confidence == 2

        # Verify path traces
        trace_labels = [str(t) for t in results[0].converging_paths]
        assert any("red" in t and "apple" in t for t in trace_labels)
        assert any("round" in t and "apple" in t for t in trace_labels)

    def test_why_shows_provenance(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        brain.teach("apples are sweet")

        traces = brain.why("apple")
        assert len(traces) == 3
        source_texts = {t.source_text for t in traces}
        assert "apples are red" in source_texts
        assert "apples are round" in source_texts
        assert "apples are sweet" in source_texts

    def test_persistence(self):
        """Teach, close, reopen — everything should still be there."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Teach
            with Brain(db_path) as b:
                b.teach("apples are red")
                b.teach("apples are round")

            # Reopen and recognize
            with Brain(db_path) as b:
                results = b.recognize("red, round")
                assert len(results) >= 1
                assert results[0].neuron.label == "apple"
                assert results[0].confidence == 2

                # Why should still work
                traces = b.why("apple")
                assert len(traces) == 2
        finally:
            os.unlink(db_path)

    def test_teach_then_similarity(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        brain.teach("apples are sweet")

        links = brain.analyze_similarity()
        # red, round, sweet all reach apple — they should be similar
        assert len(links) >= 1

    def test_stats(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")
        stats = brain.stats()
        assert stats["neurons"] == 5  # red, round, apple, apple_color, apple_shape
        assert stats["segments"] == 4
        assert stats["paths"] == 2

    def test_visualization(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are round")

        tree = render_paths_from(brain, "red")
        assert "red" in tree
        assert "apple" in tree

        dot = render_graph_dot(brain)
        assert "digraph" in dot
        assert "apple" in dot

    def test_many_teachings(self, brain):
        """Teach many facts and verify recognition still works."""
        teachings = [
            "apples are red",
            "apples are round",
            "apples are sweet",
            "bananas are yellow",
            "bananas are sweet",
            "lemons are yellow",
            "lemons are sour",
            "circles are round",
            "dogs are soft",
            "dogs are warm",
        ]
        for t in teachings:
            brain.teach(t)

        # "yellow, sweet" should recognize banana
        results = brain.recognize("yellow, sweet")
        assert results[0].neuron.label == "banana"
        assert results[0].confidence == 2

        # "yellow, sour" should recognize lemon
        results = brain.recognize("yellow, sour")
        assert results[0].neuron.label == "lemon"
        assert results[0].confidence == 2

    def test_repeated_teaching_strengthens(self, brain):
        brain.teach("apples are red")
        brain.teach("apples are red")
        brain.teach("apples are red")

        red = brain.neuron_repo.get_by_label("red")
        apple_color = brain.neuron_repo.get_by_label("apple_color")
        seg = brain.segment_repo.find(red.id, apple_color.id, "has_color")
        # Should have been strengthened twice (2nd and 3rd teach)
        assert seg.traversals == 2
        assert seg.strength > 1.0
