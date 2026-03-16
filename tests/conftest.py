"""Shared test fixtures."""

import pytest

from sara_brain.core.brain import Brain


@pytest.fixture
def brain():
    """An in-memory Brain instance for testing."""
    b = Brain(":memory:")
    yield b
    b.close()
