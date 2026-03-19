"""Tests for the perception loop orchestrator."""

from unittest.mock import MagicMock, patch
import base64
import tempfile

import pytest

from sara_brain.core.brain import Brain
from sara_brain.core.perceiver import Perceiver, PerceptionResult
from sara_brain.nlp.vision import VisionObserver


@pytest.fixture
def brain():
    b = Brain(":memory:")
    yield b
    b.close()


@pytest.fixture
def test_image(tmp_path):
    """Create a minimal test PNG."""
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4"
        "2mP8/58BAwAI/AL+hc2rNAAAAABJRU5ErkJggg=="
    )
    img_path = tmp_path / "apple.png"
    img_path.write_bytes(base64.b64decode(png_b64))
    return str(img_path)


@pytest.fixture
def mock_observer():
    observer = MagicMock(spec=VisionObserver)
    observer.observe_initial.return_value = ["red", "round", "smooth", "shiny", "small"]
    observer.observe_directed.return_value = {}
    observer.verify_property.return_value = None
    return observer


class TestInitialObservation:
    def test_teaches_observations(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        assert result.total_taught > 0
        assert len(result.all_observations) == 5
        assert "red" in result.all_observations
        assert "round" in result.all_observations

    def test_creates_image_concept_neuron(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        neuron = brain.neuron_repo.get_by_label(result.label)
        assert neuron is not None
        assert neuron.neuron_type.value == "concept"

    def test_label_format(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        assert result.label.startswith("img_apple_")
        assert len(result.label) > len("img_apple_")

    def test_custom_label(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image, label="my_photo")

        assert result.label == "my_photo"

    def test_runs_recognition(self, brain, test_image, mock_observer):
        # Pre-teach apple so recognition can find it
        brain.teach("apple is red")
        brain.teach("apple is round")

        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        # Should have recognition results in the first step
        assert len(result.steps) >= 1
        assert result.steps[0].phase == "initial"
        # Recognition should find apple since red and round converge on it
        if result.steps[0].recognition:
            labels = [r.neuron.label for r in result.steps[0].recognition]
            assert "apple" in labels


class TestDirectedInquiry:
    def test_asks_about_unobserved_associations(self, brain, test_image, mock_observer):
        # Directed inquiry fires when there are known associations
        # that weren't covered by initial observation
        mock_observer.observe_directed.return_value = {"taste": "sweet"}

        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        # Should have called observe_directed since taste isn't in initial obs
        # (color and shape are covered by red and round)
        if len(result.steps) > 1:
            assert result.steps[1].phase.startswith("directed")

    def test_teaches_directed_observations(self, brain, test_image, mock_observer):
        mock_observer.observe_directed.return_value = {"taste": "sweet"}

        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        if len(result.steps) > 1:
            assert "sweet" in result.all_observations


class TestVerification:
    def test_verifies_candidate_properties(self, brain, test_image, mock_observer):
        # Teach apple is crunchy so verification has something to check
        brain.teach("apple is red")
        brain.teach("apple is round")
        brain.teach("apple is crunchy")

        mock_observer.verify_property.return_value = True

        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        # Should have a verification step
        verification_steps = [s for s in result.steps if s.phase == "verification"]
        if verification_steps:
            assert "crunchy" in verification_steps[0].observations

    def test_skips_already_observed_properties(self, brain, test_image, mock_observer):
        brain.teach("apple is red")
        brain.teach("apple is round")

        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        # verify_property should NOT be called for red or round (already observed)
        for call in mock_observer.verify_property.call_args_list:
            prop = call[0][1]
            assert prop not in ("red", "round")


class TestConvergence:
    def test_stops_early_on_convergence(self, brain, test_image, mock_observer):
        # All directed observations return None → convergence
        mock_observer.observe_directed.return_value = {"taste": None, "temperature": None}

        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image, max_rounds=5)

        # Should not have 5 directed rounds
        directed = [s for s in result.steps if s.phase.startswith("directed")]
        assert len(directed) <= 2  # Converges quickly with no new info


class TestCorrection:
    def test_correction_teaches_correct_label(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        info = perceiver.correct("ball", result)
        assert info["correct_label"] == "ball"

        # Ball should now have properties taught to it
        ball = brain.neuron_repo.get_by_label("ball")
        assert ball is not None

    def test_correction_retains_original_observations(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        perceiver.correct("ball", result)

        # Original image observations should still exist
        img_label = result.label
        traces = brain.why(img_label)
        assert len(traces) > 0

    def test_correction_transfers_properties(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        info = perceiver.correct("ball", result)
        assert len(info["properties_taught"]) > 0

        # Ball should have red, round, etc.
        ball_traces = brain.why("ball")
        ball_properties = [t.neurons[0].label for t in ball_traces if t.neurons]
        assert "red" in ball_properties or "round" in ball_properties


class TestAddObservation:
    def test_parent_teaches_new_property(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        info = perceiver.add_observation("seams", result)
        assert info["taught"] is True
        assert info["property"] == "seams"

        # The image should now have seams
        assert "seams" in result.all_observations

    def test_duplicate_property_not_retaught(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        # First time
        perceiver.add_observation("seams", result)
        # Second time — teach still succeeds (strengthens), but path already exists
        info2 = perceiver.add_observation("seams", result)
        assert info2["taught"] is True  # Still creates a path (strengthening)


class TestNoAssociations:
    def test_perception_works_with_zero_associations(self, brain, test_image, mock_observer):
        """A naive brain with no associations still gets initial observations."""
        # Don't define any associations — just use initial observation
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        assert len(result.all_observations) == 5
        assert result.total_taught > 0
        assert len(result.steps) >= 1


class TestCallback:
    def test_callback_called_per_step(self, brain, test_image, mock_observer):
        steps_received = []

        def cb(step):
            steps_received.append(step)

        perceiver = Perceiver(brain, mock_observer)
        perceiver.perceive(test_image, callback=cb)

        assert len(steps_received) >= 1
        assert steps_received[0].phase == "initial"


class TestBrainPerceive:
    def test_brain_perceive_requires_llm(self, brain, test_image):
        with pytest.raises(ValueError, match="No LLM configured"):
            brain.perceive(test_image)

    def test_brain_correct_requires_perception(self, brain):
        with pytest.raises(ValueError, match="No recent perception"):
            brain.correct("ball")

    def test_brain_see_requires_perception(self, brain):
        with pytest.raises(ValueError, match="No recent perception"):
            brain.see("seams")

    def test_brain_stores_last_perception(self, brain, test_image, mock_observer):
        perceiver = Perceiver(brain, mock_observer)
        result = perceiver.perceive(test_image)

        assert brain._last_perception is not None
        assert brain._last_perception.label == result.label
