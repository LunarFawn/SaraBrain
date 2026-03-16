"""Tests for statement parser and taxonomy."""

from sara_brain.parsing.taxonomy import Taxonomy
from sara_brain.parsing.statement_parser import StatementParser


class TestTaxonomy:
    def test_property_type(self):
        t = Taxonomy()
        assert t.property_type("red") == "color"
        assert t.property_type("round") == "shape"
        assert t.property_type("sweet") == "taste"
        assert t.property_type("unknown") == "attribute"

    def test_subject_category(self):
        t = Taxonomy()
        assert t.subject_category("apple") == "fruit"
        assert t.subject_category("circle") == "geometric"
        assert t.subject_category("dog") == "animal"
        assert t.subject_category("unknown") == "thing"

    def test_relation_label(self):
        t = Taxonomy()
        assert t.relation_label("apple", "red") == "apple_color"
        assert t.relation_label("apple", "round") == "apple_shape"
        assert t.relation_label("circle", "round") == "circle_shape"
        assert t.relation_label("dog", "soft") == "dog_texture"

    def test_register(self):
        t = Taxonomy()
        t.register_property("sparkly", "visual")
        assert t.property_type("sparkly") == "visual"
        t.register_category("diamond", "gem")
        assert t.subject_category("diamond") == "gem"


class TestStatementParser:
    def test_parse_simple(self):
        t = Taxonomy()
        p = StatementParser(t)
        result = p.parse("apples are red")
        assert result is not None
        assert result.subject == "apple"
        assert result.obj == "red"
        assert result.relation == "has_color"

    def test_parse_singular(self):
        t = Taxonomy()
        p = StatementParser(t)
        result = p.parse("an apple is red")
        assert result is not None
        assert result.subject == "apple"

    def test_parse_shape(self):
        t = Taxonomy()
        p = StatementParser(t)
        result = p.parse("circles are round")
        assert result.subject == "circle"
        assert result.relation == "has_shape"
        assert result.obj == "round"

    def test_parse_is_a(self):
        t = Taxonomy()
        p = StatementParser(t)
        result = p.parse("a dog is a pet")
        assert result is not None
        assert result.subject == "dog"
        assert result.relation == "is_a"
        assert result.obj == "pet"

    def test_parse_empty(self):
        t = Taxonomy()
        p = StatementParser(t)
        assert p.parse("") is None
        assert p.parse("hello") is None

    def test_parse_preserves_original(self):
        t = Taxonomy()
        p = StatementParser(t)
        result = p.parse("Apples are Red")
        assert result.original == "Apples are Red"
