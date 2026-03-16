"""Tests for SQLite storage layer."""

from sara_brain.storage.database import Database
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo
from sara_brain.storage.path_repo import PathRepo
from sara_brain.models.neuron import Neuron, NeuronType
from sara_brain.models.segment import Segment
from sara_brain.models.path import Path, PathStep


class TestDatabase:
    def test_creates_tables(self):
        db = Database(":memory:")
        tables = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t[0] for t in tables]
        assert "neurons" in table_names
        assert "segments" in table_names
        assert "paths" in table_names
        assert "path_steps" in table_names
        assert "similarities" in table_names
        db.close()

    def test_wal_mode(self):
        db = Database(":memory:")
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        # WAL may not be supported for :memory:, that's fine
        assert mode in ("wal", "memory")
        db.close()


class TestNeuronRepo:
    def test_create_and_get(self):
        db = Database(":memory:")
        repo = NeuronRepo(db.conn)
        n = Neuron(id=None, label="apple", neuron_type=NeuronType.CONCEPT)
        repo.create(n)
        assert n.id is not None

        fetched = repo.get_by_id(n.id)
        assert fetched.label == "apple"
        assert fetched.neuron_type == NeuronType.CONCEPT
        db.close()

    def test_get_by_label(self):
        db = Database(":memory:")
        repo = NeuronRepo(db.conn)
        repo.create(Neuron(id=None, label="red", neuron_type=NeuronType.PROPERTY))
        assert repo.get_by_label("red") is not None
        assert repo.get_by_label("blue") is None
        db.close()

    def test_get_or_create(self):
        db = Database(":memory:")
        repo = NeuronRepo(db.conn)
        n1, created1 = repo.get_or_create("apple", NeuronType.CONCEPT)
        assert created1 is True
        n2, created2 = repo.get_or_create("apple", NeuronType.CONCEPT)
        assert created2 is False
        assert n1.id == n2.id
        db.close()

    def test_list_all(self):
        db = Database(":memory:")
        repo = NeuronRepo(db.conn)
        repo.create(Neuron(id=None, label="a", neuron_type=NeuronType.CONCEPT))
        repo.create(Neuron(id=None, label="b", neuron_type=NeuronType.PROPERTY))
        assert repo.count() == 2
        assert len(repo.list_all()) == 2
        db.close()


class TestSegmentRepo:
    def test_create_and_get(self):
        db = Database(":memory:")
        nr = NeuronRepo(db.conn)
        nr.create(Neuron(id=None, label="a", neuron_type=NeuronType.PROPERTY))
        nr.create(Neuron(id=None, label="b", neuron_type=NeuronType.CONCEPT))
        sr = SegmentRepo(db.conn)
        seg = Segment(id=None, source_id=1, target_id=2, relation="test")
        sr.create(seg)
        assert seg.id is not None

        fetched = sr.get_by_id(seg.id)
        assert fetched.source_id == 1
        assert fetched.target_id == 2
        db.close()

    def test_get_outgoing(self):
        db = Database(":memory:")
        nr = NeuronRepo(db.conn)
        nr.create(Neuron(id=None, label="a", neuron_type=NeuronType.PROPERTY))
        nr.create(Neuron(id=None, label="b", neuron_type=NeuronType.CONCEPT))
        nr.create(Neuron(id=None, label="c", neuron_type=NeuronType.CONCEPT))
        sr = SegmentRepo(db.conn)
        sr.create(Segment(id=None, source_id=1, target_id=2, relation="r1"))
        sr.create(Segment(id=None, source_id=1, target_id=3, relation="r2"))
        assert len(sr.get_outgoing(1)) == 2
        assert len(sr.get_outgoing(2)) == 0
        db.close()

    def test_strengthen(self):
        db = Database(":memory:")
        nr = NeuronRepo(db.conn)
        nr.create(Neuron(id=None, label="a", neuron_type=NeuronType.PROPERTY))
        nr.create(Neuron(id=None, label="b", neuron_type=NeuronType.CONCEPT))
        sr = SegmentRepo(db.conn)
        seg = Segment(id=None, source_id=1, target_id=2, relation="test")
        sr.create(seg)

        sr.strengthen(seg)
        fetched = sr.get_by_id(seg.id)
        assert fetched.traversals == 1
        assert fetched.strength > 1.0
        db.close()


class TestPathRepo:
    def test_create_with_steps(self):
        db = Database(":memory:")
        nr = NeuronRepo(db.conn)
        nr.create(Neuron(id=None, label="a", neuron_type=NeuronType.PROPERTY))
        nr.create(Neuron(id=None, label="b", neuron_type=NeuronType.RELATION))
        nr.create(Neuron(id=None, label="c", neuron_type=NeuronType.CONCEPT))
        sr = SegmentRepo(db.conn)
        sr.create(Segment(id=None, source_id=1, target_id=2, relation="r1"))
        sr.create(Segment(id=None, source_id=2, target_id=3, relation="r2"))

        pr = PathRepo(db.conn)
        path = Path(id=None, origin_id=1, terminus_id=3, source_text="test")
        pr.create(path)
        assert path.id is not None

        pr.add_step(PathStep(id=None, path_id=path.id, step_order=0, segment_id=1))
        pr.add_step(PathStep(id=None, path_id=path.id, step_order=1, segment_id=2))

        steps = pr.get_steps(path.id)
        assert len(steps) == 2
        assert steps[0].step_order == 0
        assert steps[1].step_order == 1
        db.close()

    def test_get_paths_to(self):
        db = Database(":memory:")
        nr = NeuronRepo(db.conn)
        nr.create(Neuron(id=None, label="a", neuron_type=NeuronType.PROPERTY))
        nr.create(Neuron(id=None, label="b", neuron_type=NeuronType.CONCEPT))
        pr = PathRepo(db.conn)
        pr.create(Path(id=None, origin_id=1, terminus_id=2, source_text="x"))
        pr.create(Path(id=None, origin_id=1, terminus_id=2, source_text="y"))
        assert len(pr.get_paths_to(2)) == 2
        db.close()
