"""Tests for the sensory shell module.

Tests the empty transformer shell — tokenizer, renderer, shell
processing, and the contract that all output is traceable to
specific taught facts.
"""

from __future__ import annotations

import pytest

from sara_brain.core.brain import Brain
from sara_brain.sensory.tokenizer import Tokenizer, Token, _STOP
from sara_brain.sensory.renderer import Renderer, SourcedLine
from sara_brain.sensory.shell import SensoryShell, ShellResponse
from sara_brain.sensory.session import Session


@pytest.fixture
def brain(tmp_path):
    """Create a fresh brain for testing."""
    db_path = str(tmp_path / "test_sensory.db")
    return Brain(db_path)


@pytest.fixture
def taught_brain(brain):
    """Brain with a few facts taught."""
    brain.teach("apple is red")
    brain.teach("apple is round")
    brain.teach("apple is fruit")
    brain.teach("banana is yellow")
    brain.teach("banana is fruit")
    brain.teach("methane has one carbon atom")
    brain.teach("methane has four hydrogen atoms")
    brain.teach("methane is a chemical compound")
    return brain


# ── Tokenizer tests ──


class TestTokenizer:

    def test_splits_words(self, brain):
        tok = Tokenizer(brain)
        tokens = tok.tokenize("hello world")
        labels = [t.label for t in tokens]
        assert "hello" in labels
        assert "world" in labels

    def test_strips_stopwords(self, brain):
        tok = Tokenizer(brain)
        tokens = tok.tokenize("what is the apple")
        labels = [t.label for t in tokens]
        # "what", "is", "the" are stopwords
        assert "what" not in labels
        assert "is" not in labels
        assert "the" not in labels
        assert "apple" in labels

    def test_lowercases(self, brain):
        tok = Tokenizer(brain)
        tokens = tok.tokenize("Apple Is RED")
        labels = [t.label for t in tokens]
        assert "apple" in labels
        assert "red" in labels
        assert "Apple" not in labels

    def test_finds_known_neurons(self, taught_brain):
        tok = Tokenizer(taught_brain)
        tokens = tok.tokenize("apple red")
        apple_tokens = [t for t in tokens if t.label == "apple"]
        assert len(apple_tokens) == 1
        assert apple_tokens[0].neuron_id is not None

    def test_unknown_words_have_no_neuron_id(self, brain):
        tok = Tokenizer(brain)
        tokens = tok.tokenize("quark")
        assert len(tokens) == 1
        assert tokens[0].label == "quark"
        assert tokens[0].neuron_id is None

    def test_empty_input(self, brain):
        tok = Tokenizer(brain)
        tokens = tok.tokenize("")
        assert tokens == []

    def test_all_stopwords(self, brain):
        tok = Tokenizer(brain)
        tokens = tok.tokenize("is the a an")
        assert tokens == []

    def test_seed_labels_convenience(self, taught_brain):
        tok = Tokenizer(taught_brain)
        labels = tok.seed_labels("apple red fruit")
        assert "apple" in labels
        assert "red" in labels
        assert "fruit" in labels

    def test_multi_word_phrase_match(self, taught_brain):
        # Teach a multi-word concept
        taught_brain.teach("electron transport chain is a process")
        tok = Tokenizer(taught_brain)
        tokens = tok.tokenize("the electron transport chain produces ATP")
        labels = [t.label for t in tokens]
        assert "electron transport chain" in labels
        phrase_tokens = [t for t in tokens if t.label == "electron transport chain"]
        assert phrase_tokens[0].is_phrase is True


# ── Renderer tests ──


class TestRenderer:

    def test_no_paths_says_dont_know(self, brain):
        renderer = Renderer(brain)
        lines = renderer.render_recognition([])
        assert len(lines) == 1
        assert "don't know" in lines[0].text.lower()

    def test_sourced_line_str(self):
        line = SourcedLine(
            text="apple is red.",
            path_id=42,
            source_text="apple is red",
        )
        s = str(line)
        assert "path #42" in s
        assert "apple is red" in s

    def test_format_output_no_lines(self):
        assert Renderer.format_output([]) == "I don't know."


# ── Shell tests ──


class TestSensoryShell:

    def test_empty_brain_says_dont_know(self, brain):
        shell = SensoryShell(brain)
        response = shell.process("what is a quark")
        assert "don't know" in response.text.lower() or len(response.gaps) > 0

    def test_query_taught_fact(self, taught_brain):
        shell = SensoryShell(taught_brain)
        response = shell.query("apple")
        # Should find paths about apple
        assert response.confidence > 0 or len(response.sources) > 0

    def test_query_unknown_topic(self, brain):
        shell = SensoryShell(brain)
        response = shell.query("quark")
        assert "don't know" in response.text.lower()

    def test_process_returns_shell_response(self, taught_brain):
        shell = SensoryShell(taught_brain)
        response = shell.process("red round fruit")
        assert isinstance(response, ShellResponse)
        assert isinstance(response.tokens, list)
        assert isinstance(response.sources, list)
        assert isinstance(response.gaps, list)

    def test_process_with_taught_facts_finds_signal(self, taught_brain):
        shell = SensoryShell(taught_brain)
        response = shell.process("red round fruit")
        # With apple taught as red, round, fruit — should find something
        has_output = (
            response.confidence > 0
            or len(response.sources) > 0
            or "apple" in response.text.lower()
        )
        assert has_output

    def test_teach_then_query(self, brain):
        shell = SensoryShell(brain)
        # Empty — knows nothing
        response = shell.query("dog")
        assert "don't know" in response.text.lower()

        # Teach
        brain.teach("dog is loyal")
        brain.teach("dog is animal")

        # Now it knows
        response = shell.query("dog")
        assert "don't know" not in response.text.lower()

    def test_gaps_reported_for_unknown_tokens(self, brain):
        shell = SensoryShell(brain)
        response = shell.process("quark gluon boson")
        # All tokens unknown — should appear in gaps
        assert len(response.gaps) > 0

    def test_empty_input(self, brain):
        shell = SensoryShell(brain)
        response = shell.process("")
        assert "don't know" in response.text.lower()


# ── Session tests ──


class TestSession:

    def test_add_turn_tracks_topics(self):
        session = Session()
        session.add_turn(["apple", "red"])
        assert "apple" in session.context_seeds()
        assert "red" in session.context_seeds()

    def test_clear_resets(self):
        session = Session()
        session.add_turn(["apple"])
        session.clear()
        assert session.context_seeds() == []
        assert session.turn_count == 0

    def test_dedup_topics(self):
        session = Session()
        session.add_turn(["apple", "red"])
        session.add_turn(["apple", "green"])
        seeds = session.context_seeds()
        assert seeds.count("apple") == 1

    def test_max_history(self):
        session = Session(max_history=3)
        session.add_turn(["a"])
        session.add_turn(["b"])
        session.add_turn(["c"])
        session.add_turn(["d"])
        seeds = session.context_seeds()
        assert "a" not in seeds  # pushed out
        assert "d" in seeds
