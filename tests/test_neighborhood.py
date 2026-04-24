"""Tests for Brain.neighborhood — BFS graph walk from a seed label.

Covers the core behavior expected by the brain_explore MCP tool:
- Seed found / not found
- Depth 1 returns direct neighbors only
- Depth 2 includes adjacent neurons' neighbors
- Case-insensitive seed resolution
- Truncation honors max_edges
"""
from __future__ import annotations

from sara_brain.core.brain import Brain


def _make_brain() -> Brain:
    b = Brain(":memory:")
    # Build a small traceable graph:
    #   creed 2 is_a jkd creed
    #   creed 2 builds_on creed 1
    #   creed 2 taught_by jennifer
    #   jennifer authored the paper
    #   the paper discusses creed 2
    b.teach_triple("creed 2", "is_a", "jkd creed")
    b.teach_triple("creed 2", "builds_on", "creed 1")
    b.teach_triple("creed 2", "taught_by", "jennifer")
    b.teach_triple("jennifer", "authored", "the paper")
    b.teach_triple("the paper", "discusses", "creed 2")
    return b


def test_neighborhood_seed_not_found() -> None:
    b = _make_brain()
    result = b.neighborhood("does not exist")
    assert result["seed_found"] is False
    assert result["total_neurons"] == 0
    assert result["edges"] == []


def test_neighborhood_depth_1_finds_direct_neighbors() -> None:
    b = _make_brain()
    result = b.neighborhood("creed 2", depth=1)
    assert result["seed_found"] is True
    reachable = set()
    for labels in result["neurons_by_depth"].values():
        reachable.update(labels)
    # Direct semantic neighbors (ignore attribute plumbing for this check)
    non_attr = {l for l in reachable if not l.endswith("_attribute")}
    assert "creed 2" in non_attr
    assert "jkd creed" in non_attr
    assert "creed 1" in non_attr
    assert "jennifer" in non_attr


def test_neighborhood_depth_2_expands_to_adjacents_adjacents() -> None:
    b = _make_brain()
    result = b.neighborhood("creed 2", depth=2)
    reachable = set()
    for labels in result["neurons_by_depth"].values():
        reachable.update(labels)
    non_attr = {l for l in reachable if not l.endswith("_attribute")}
    # At depth 2 we should reach "the paper" via jennifer
    assert "the paper" in non_attr


def test_neighborhood_case_insensitive_seed() -> None:
    b = _make_brain()
    r_upper = b.neighborhood("CREED 2")
    r_lower = b.neighborhood("creed 2")
    assert r_upper["seed_found"] is True
    assert r_lower["seed_found"] is True
    # Same graph, same edge count
    assert r_upper["total_edges"] == r_lower["total_edges"]


def test_neighborhood_truncation_respects_max_edges() -> None:
    b = _make_brain()
    full = b.neighborhood("creed 2", depth=3, max_edges=10_000)
    truncated = b.neighborhood("creed 2", depth=3, max_edges=2)
    assert truncated["truncated"] is True
    # Truncation must stop BFS growth; truncated run has strictly fewer
    # edges than an unbounded one on the same graph.
    assert truncated["total_edges"] < full["total_edges"]


def test_neighborhood_edges_have_expected_shape() -> None:
    b = _make_brain()
    result = b.neighborhood("creed 2", depth=1)
    for e in result["edges"]:
        assert set(e.keys()) >= {"source", "relation", "target", "strength", "depth_seen"}
        assert isinstance(e["source"], str)
        assert isinstance(e["target"], str)
        assert e["depth_seen"] >= 1
