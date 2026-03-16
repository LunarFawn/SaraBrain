"""Tests for category functionality."""

import os
import tempfile

import pytest

from sara_brain.core.brain import Brain


@pytest.fixture
def brain():
    b = Brain(":memory:")
    yield b
    b.close()


class TestCategorize:
    def test_categorize_sets_category(self, brain):
        brain.categorize("apple", "item")
        assert brain.get_category("apple") == "item"

    def test_categorize_overrides_builtin(self, brain):
        # apple is "fruit" by default in taxonomy
        assert brain.get_category("apple") == "fruit"
        brain.categorize("apple", "item")
        assert brain.get_category("apple") == "item"

    def test_categorize_new_concept(self, brain):
        brain.categorize("running", "action")
        assert brain.get_category("running") == "action"

    def test_default_category_is_thing(self, brain):
        assert brain.get_category("unknown_xyz") == "thing"


class TestListCategories:
    def test_list_includes_builtins(self, brain):
        cats = brain.list_categories()
        assert "fruit" in cats
        assert "apple" in cats["fruit"]

    def test_list_includes_dynamic(self, brain):
        brain.categorize("running", "action")
        cats = brain.list_categories()
        assert "action" in cats
        assert "running" in cats["action"]

    def test_list_sorted(self, brain):
        brain.categorize("zucchini", "vegetable")
        brain.categorize("carrot", "vegetable")
        cats = brain.list_categories()
        assert cats["vegetable"] == ["carrot", "zucchini"]


class TestCategoryPersistence:
    def test_categories_persist_across_restart(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with Brain(db_path) as b1:
                b1.categorize("apple", "item")
                b1.categorize("running", "action")

            with Brain(db_path) as b2:
                assert b2.get_category("apple") == "item"
                assert b2.get_category("running") == "action"
                cats = b2.list_categories()
                assert "item" in cats
                assert "apple" in cats["item"]
        finally:
            os.unlink(db_path)

    def test_category_persists_and_loads(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with Brain(db_path) as b1:
                b1.categorize("pizza", "food")

            with Brain(db_path) as b2:
                # pizza should now be categorized as "food" after restart
                assert b2.get_category("pizza") == "food"
                # Teaching still works with the persisted category
                result = b2.teach("pizza is hot")
                assert result is not None
                assert "pizza_temperature" in result.path_label
        finally:
            os.unlink(db_path)


class TestCategoryRepo:
    def test_set_and_get(self, brain):
        brain.category_repo.set_category("test", "demo")
        brain.conn.commit()
        assert brain.category_repo.get_category("test") == "demo"

    def test_get_nonexistent(self, brain):
        assert brain.category_repo.get_category("nope") is None

    def test_list_by_category(self, brain):
        brain.category_repo.set_category("a", "cat1")
        brain.category_repo.set_category("b", "cat1")
        brain.category_repo.set_category("c", "cat2")
        brain.conn.commit()
        assert brain.category_repo.list_by_category("cat1") == ["a", "b"]
        assert brain.category_repo.list_by_category("cat2") == ["c"]

    def test_list_categories(self, brain):
        brain.category_repo.set_category("x", "type1")
        brain.category_repo.set_category("y", "type1")
        brain.category_repo.set_category("z", "type2")
        brain.conn.commit()
        cats = brain.category_repo.list_categories()
        assert cats == {"type1": ["x", "y"], "type2": ["z"]}

    def test_replace_category(self, brain):
        brain.category_repo.set_category("item", "cat1")
        brain.conn.commit()
        brain.category_repo.set_category("item", "cat2")
        brain.conn.commit()
        assert brain.category_repo.get_category("item") == "cat2"
