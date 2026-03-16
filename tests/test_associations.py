"""Tests for dynamic associations."""

import os
import tempfile

import pytest

from sara_brain.core.brain import Brain
from sara_brain.models.neuron import NeuronType


@pytest.fixture
def brain():
    b = Brain(":memory:")
    yield b
    b.close()


class TestDefineAssociation:
    def test_define_creates_association_neuron(self, brain):
        neuron = brain.define_association("mood")
        assert neuron.label == "mood"
        assert neuron.neuron_type == NeuronType.ASSOCIATION

    def test_define_idempotent(self, brain):
        n1 = brain.define_association("mood")
        n2 = brain.define_association("mood")
        assert n1.id == n2.id


class TestDescribeAssociation:
    def test_describe_registers_properties(self, brain):
        brain.define_association("taste")
        registered = brain.describe_association("taste", ["sweet", "sour", "bitter"])
        assert registered == ["sweet", "sour", "bitter"]

    def test_describe_creates_property_to_association_paths(self, brain):
        brain.define_association("taste")
        brain.describe_association("taste", ["sweet", "sour"])

        # sweet should have an is_a segment to taste
        sweet = brain.neuron_repo.get_by_label("sweet")
        taste = brain.neuron_repo.get_by_label("taste")
        assert sweet is not None
        assert taste is not None
        assert sweet.neuron_type == NeuronType.PROPERTY
        assert taste.neuron_type == NeuronType.ASSOCIATION

        seg = brain.segment_repo.find(sweet.id, taste.id, "is_a")
        assert seg is not None

    def test_describe_updates_taxonomy(self, brain):
        brain.define_association("mood")
        brain.describe_association("mood", ["happy", "sad"])
        assert brain.taxonomy.property_type("happy") == "mood"
        assert brain.taxonomy.property_type("sad") == "mood"

    def test_describe_unknown_association_fails(self, brain):
        with pytest.raises(ValueError, match="Unknown association"):
            brain.describe_association("nonexistent", ["a", "b"])

    def test_teach_uses_dynamic_association(self, brain):
        brain.define_association("mood")
        brain.describe_association("mood", ["happy", "sad", "anxious"])

        result = brain.teach("dogs are happy")
        assert result is not None
        # Path should use "mood" as the property type → dog_mood intermediate
        assert "dog_mood" in result.path_label


class TestAssociationsList:
    def test_associations_list(self, brain):
        brain.define_association("mood")
        brain.describe_association("mood", ["happy", "sad"])
        brain.define_association("taste")
        brain.describe_association("taste", ["tangy"])

        assocs = brain.list_associations()
        assert "mood" in assocs
        assert "taste" in assocs
        assert sorted(assocs["mood"]) == ["happy", "sad"]
        assert assocs["taste"] == ["tangy"]

    def test_list_empty(self, brain):
        assocs = brain.list_associations()
        assert assocs == {}


class TestAssociationPersistence:
    def test_associations_persist_across_restart(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            # First session: define and describe
            with Brain(db_path) as b1:
                b1.define_association("mood")
                b1.describe_association("mood", ["happy", "sad"])

            # Second session: should load from DB
            with Brain(db_path) as b2:
                assert b2.taxonomy.property_type("happy") == "mood"
                assert b2.taxonomy.property_type("sad") == "mood"

                assocs = b2.list_associations()
                assert "mood" in assocs
                assert sorted(assocs["mood"]) == ["happy", "sad"]

                # teach should work with the persisted association
                result = b2.teach("dogs are happy")
                assert result is not None
                assert "dog_mood" in result.path_label
        finally:
            os.unlink(db_path)
