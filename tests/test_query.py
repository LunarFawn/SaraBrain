"""Tests for query and question word functionality."""

import os
import tempfile

import pytest

from sara_brain.core.brain import Brain, _BUILTIN_QUESTION_WORDS


@pytest.fixture
def brain():
    b = Brain(":memory:")
    yield b
    b.close()


class TestQueryAssociation:
    def test_query_builtin_taste(self, brain):
        brain.teach("apples are sweet")
        results = brain.query_association("apple", "taste")
        assert "sweet" in results

    def test_query_builtin_color(self, brain):
        brain.teach("apples are red")
        results = brain.query_association("apple", "color")
        assert "red" in results

    def test_query_multiple_properties(self, brain):
        brain.teach("apples are sweet")
        brain.teach("apples are sour")
        results = brain.query_association("apple", "taste")
        assert "sweet" in results
        assert "sour" in results

    def test_query_unknown_concept(self, brain):
        results = brain.query_association("nonexistent", "taste")
        assert results == []

    def test_query_unknown_association(self, brain):
        brain.teach("apples are red")
        results = brain.query_association("apple", "nonexistent")
        assert results == []

    def test_query_with_dynamic_association(self, brain):
        brain.define_association("mood", "how")
        brain.describe_association("mood", ["happy", "sad"])
        brain.teach("dogs are happy")
        results = brain.query_association("dog", "mood")
        assert "happy" in results

    def test_query_returns_sorted(self, brain):
        brain.teach("apples are sweet")
        brain.teach("apples are sour")
        results = brain.query_association("apple", "taste")
        assert results == sorted(results)


class TestDefineWithQuestionWord:
    def test_define_stores_question_word(self, brain):
        brain.define_association("taste", "how")
        qw = brain.association_repo.get_question_word("taste")
        assert qw == "how"

    def test_define_without_question_word(self, brain):
        brain.define_association("mood")
        qw = brain.association_repo.get_question_word("mood")
        assert qw is None

    def test_get_by_question_word(self, brain):
        brain.define_association("taste", "how")
        brain.define_association("texture", "how")
        assocs = brain.association_repo.get_by_question_word("how")
        assert "taste" in assocs
        assert "texture" in assocs

    def test_get_by_question_word_not_found(self, brain):
        assocs = brain.association_repo.get_by_question_word("zorp")
        assert assocs == []


class TestListQuestionWords:
    def test_list_includes_builtins(self, brain):
        qwords = brain.list_question_words()
        assert "what" in qwords
        assert "how" in qwords
        assert "color" in qwords["what"]
        assert "taste" in qwords["how"]

    def test_list_includes_dynamic(self, brain):
        brain.define_association("location", "where")
        qwords = brain.list_question_words()
        assert "where" in qwords
        assert "location" in qwords["where"]

    def test_list_combines_builtin_and_dynamic(self, brain):
        brain.define_association("weight", "what")
        qwords = brain.list_question_words()
        # "what" should have both builtins and dynamic
        assert "color" in qwords["what"]
        assert "weight" in qwords["what"]


class TestResolveQuestionWord:
    def test_resolve_builtin(self, brain):
        assocs = brain.resolve_question_word("what")
        assert "color" in assocs
        assert "shape" in assocs
        assert "size" in assocs

    def test_resolve_dynamic(self, brain):
        brain.define_association("location", "where")
        assocs = brain.resolve_question_word("where")
        assert "location" in assocs

    def test_resolve_unknown(self, brain):
        assocs = brain.resolve_question_word("zorp")
        assert assocs == []


class TestQuestionWordPersistence:
    def test_question_words_persist_across_restart(self):
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            with Brain(db_path) as b1:
                b1.define_association("location", "where")

            with Brain(db_path) as b2:
                qw = b2.association_repo.get_question_word("location")
                assert qw == "where"
                assocs = b2.resolve_question_word("where")
                assert "location" in assocs
        finally:
            os.unlink(db_path)
