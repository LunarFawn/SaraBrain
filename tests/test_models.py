"""Tests for pure dataclasses."""

import math

from sara_brain.models.neuron import Neuron, NeuronType
from sara_brain.models.segment import Segment
from sara_brain.models.path import Path, PathStep
from sara_brain.models.result import PathTrace, RecognitionResult


class TestNeuron:
    def test_create(self):
        n = Neuron(id=1, label="apple", neuron_type=NeuronType.CONCEPT)
        assert n.label == "apple"
        assert n.neuron_type == NeuronType.CONCEPT

    def test_metadata_json(self):
        n = Neuron(id=1, label="x", neuron_type=NeuronType.PROPERTY, metadata={"key": "val"})
        assert '"key"' in n.metadata_json()

    def test_metadata_from_json(self):
        result = Neuron.metadata_from_json('{"a": 1}')
        assert result == {"a": 1}

    def test_metadata_none(self):
        assert Neuron.metadata_from_json(None) is None


class TestSegment:
    def test_strengthen(self):
        s = Segment(id=1, source_id=1, target_id=2, relation="test")
        assert s.strength == 1.0
        assert s.traversals == 0

        s.strengthen()
        assert s.traversals == 1
        assert s.strength == 1.0 + math.log(2)

        s.strengthen()
        assert s.traversals == 2
        assert s.strength == 1.0 + math.log(3)

    def test_strength_only_goes_up(self):
        s = Segment(id=1, source_id=1, target_id=2, relation="test")
        prev = s.strength
        for _ in range(100):
            s.strengthen()
            assert s.strength >= prev
            prev = s.strength


class TestPath:
    def test_create(self):
        p = Path(id=1, origin_id=1, terminus_id=3, source_text="apples are red")
        assert p.source_text == "apples are red"

    def test_step(self):
        ps = PathStep(id=1, path_id=1, step_order=0, segment_id=5)
        assert ps.step_order == 0


class TestResult:
    def test_path_trace(self):
        neurons = [
            Neuron(id=1, label="red", neuron_type=NeuronType.PROPERTY),
            Neuron(id=2, label="apple_color", neuron_type=NeuronType.RELATION),
            Neuron(id=3, label="apple", neuron_type=NeuronType.CONCEPT),
        ]
        trace = PathTrace(neurons=neurons)
        assert trace.labels() == ["red", "apple_color", "apple"]
        assert str(trace) == "red → apple_color → apple"

    def test_recognition_result(self):
        apple = Neuron(id=3, label="apple", neuron_type=NeuronType.CONCEPT)
        trace1 = PathTrace(neurons=[
            Neuron(id=1, label="red", neuron_type=NeuronType.PROPERTY),
            apple,
        ])
        trace2 = PathTrace(neurons=[
            Neuron(id=2, label="round", neuron_type=NeuronType.PROPERTY),
            apple,
        ])
        result = RecognitionResult(neuron=apple, converging_paths=[trace1, trace2])
        assert result.confidence == 2
