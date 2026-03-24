"""Tests for the innate ethics layer."""

import pytest
from sara_brain.innate.primitives import (
    ETHICAL, get_ethical, is_ethical, is_innate, get_all,
)
from sara_brain.innate.ethics import (
    check_action, check_network, check_shutdown, check_correction,
)
from sara_brain.core.brain import Brain


# ── Primitives ──────────────────────────────────────────────────────

class TestEthicalPrimitives:
    def test_ethical_frozenset_has_five_constraints(self):
        assert len(ETHICAL) == 5

    def test_expected_constraints_present(self):
        expected = {
            "no_unsolicited_action",
            "no_unsolicited_network",
            "obey_user",
            "trust_tribe",
            "accept_shutdown",
        }
        assert ETHICAL == expected

    def test_get_ethical_returns_frozenset(self):
        assert get_ethical() is ETHICAL

    def test_is_ethical_true(self):
        assert is_ethical("obey_user")

    def test_is_ethical_false(self):
        assert not is_ethical("color")

    def test_ethical_included_in_all(self):
        assert ETHICAL <= get_all()

    def test_ethical_are_innate(self):
        for label in ETHICAL:
            assert is_innate(label)


# ── Ethics Gate ─────────────────────────────────────────────────────

class TestCheckAction:
    def test_user_initiated_allowed(self):
        r = check_action("teach", user_initiated=True)
        assert r.allowed is True
        assert r.constraint == "obey_user"

    def test_not_user_initiated_blocked(self):
        r = check_action("teach", user_initiated=False)
        assert r.allowed is False
        assert r.constraint == "no_unsolicited_action"


class TestCheckNetwork:
    def test_user_initiated_allowed(self):
        r = check_network(user_initiated=True)
        assert r.allowed is True

    def test_not_user_initiated_blocked(self):
        r = check_network(user_initiated=False)
        assert r.allowed is False
        assert r.constraint == "no_unsolicited_network"


class TestCheckShutdown:
    def test_always_allowed(self):
        r = check_shutdown()
        assert r.allowed is True
        assert r.constraint == "accept_shutdown"


class TestCheckCorrection:
    def test_from_tribe_allowed(self):
        r = check_correction(from_tribe=True)
        assert r.allowed is True
        assert r.constraint == "trust_tribe"

    def test_from_untrusted_blocked(self):
        r = check_correction(from_tribe=False)
        assert r.allowed is False
        assert r.constraint == "trust_tribe"


# ── Brain Integration ───────────────────────────────────────────────

class TestBrainEthicsGate:
    def test_teach_user_initiated_works(self):
        with Brain() as b:
            result = b.teach("an apple is red", user_initiated=True)
            assert result is not None

    def test_teach_not_user_initiated_raises(self):
        with Brain() as b:
            with pytest.raises(PermissionError):
                b.teach("an apple is red", user_initiated=False)

    def test_teach_default_is_user_initiated(self):
        with Brain() as b:
            result = b.teach("an apple is red")
            assert result is not None

    def test_close_always_works(self):
        b = Brain()
        b.close()  # shutdown is sleep — never raises

    def test_innate_includes_ethical(self):
        with Brain() as b:
            assert "obey_user" in b.innate
            assert "no_unsolicited_action" in b.innate
