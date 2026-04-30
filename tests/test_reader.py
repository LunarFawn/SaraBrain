"""Tests for sara_reader.

Network providers (anthropic, ollama) are not exercised in unit tests.
The tests here validate provider-loader policy, tool registration,
schema shapes, and the OpenAI-exclusion enforcement.
"""
from __future__ import annotations

import pytest

from sara_reader.providers import get_provider, BaseProvider
from sara_reader.tools import TOOLS, get_tool_schemas


# --- Provider exclusion (the ethical-stance enforcement) ------------------

@pytest.mark.parametrize("name", [
    "openai", "OpenAI", "OPENAI", "open-ai", "open_ai", "gpt", "openai-api",
    " openai ", " OPENAI",
])
def test_openai_provider_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="not a supported provider"):
        get_provider(name)


def test_unknown_provider_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent_provider")


def test_anthropic_provider_loads() -> None:
    # Loading should succeed even without an API key set; the key
    # is only required at .chat() time.
    p = get_provider("anthropic", api_key="dummy-for-construction")
    assert isinstance(p, BaseProvider)


def test_ollama_provider_loads() -> None:
    p = get_provider("ollama")
    assert isinstance(p, BaseProvider)


# --- Tool surface ---------------------------------------------------------

def test_expected_tools_are_registered() -> None:
    expected = {
        "brain_explore",
        "brain_why",
        "brain_trace",
        "brain_recognize",
        "brain_did_you_mean",
        "brain_define",
        "brain_value",
    }
    assert set(TOOLS.keys()) == expected


def test_write_tools_are_not_exposed() -> None:
    """Confirm teach/refute/ingest do NOT appear in the tool surface.

    sara_reader is read-only by default; consumer apps that need to
    teach must use Brain.teach_triple directly with authorization.
    """
    forbidden = {"brain_teach", "brain_teach_triple", "brain_refute", "brain_ingest"}
    assert forbidden.isdisjoint(TOOLS.keys())


def test_tool_schemas_have_provider_agnostic_shape() -> None:
    schemas = get_tool_schemas()
    for s in schemas:
        assert set(s.keys()) == {"name", "description", "parameters"}
        assert isinstance(s["parameters"], dict)
        assert s["parameters"].get("type") == "object"
