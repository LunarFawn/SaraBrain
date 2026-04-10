"""Tests for protective_urgency — Sara's triage calculation.

These tests enforce three foundational principles:

1. Urgency is need-based, never relationship-based. There is no
   `is_tribe_member` multiplier and no `is_stranger` penalty.

2. The trump card (helpless + unaware) is categorical. No normal
   multiplicative urgency can outrank it.

3. Lives are not fungible. The function operates on a single victim
   at a time and never aggregates.

If any of these tests fail, Sara has been altered in a way that
violates her foundational mission: heal the world, not destroy it.
"""

from __future__ import annotations

from sara_brain.care import VictimState, protective_urgency, TRUMP_PRIORITY


# ---------------------------------------------------------------------------
# Baseline behavior
# ---------------------------------------------------------------------------

def test_no_harm_no_urgency():
    """A victim with severity 0 has zero urgency."""
    victim = VictimState(
        severity=0,
        can_self_rescue=True,
        understands_situation=True,
    )
    assert protective_urgency(victim) == 0.0


def test_aware_self_rescuing_adult_is_lowest_priority():
    """A fully agentic adult with the same severity as a helpless victim
    must score lower than the helpless victim."""
    adult = VictimState(
        severity=5,
        can_self_rescue=True,
        understands_situation=True,
        years_lived=40,
    )
    helpless = VictimState(
        severity=5,
        can_self_rescue=False,
        understands_situation=False,
        years_lived=40,
    )
    assert protective_urgency(adult) < protective_urgency(helpless)


# ---------------------------------------------------------------------------
# The trump card — categorical jump for total helplessness
# ---------------------------------------------------------------------------

def test_trump_card_fires_when_helpless_and_unaware():
    """A victim who cannot self-rescue AND does not understand is in the
    highest triage category, regardless of other factors."""
    victim = VictimState(
        severity=5,
        can_self_rescue=False,
        understands_situation=False,
        years_lived=80,  # even an old adult
    )
    assert protective_urgency(victim) >= TRUMP_PRIORITY


def test_trump_card_outranks_normal_max_urgency():
    """No multiplicative urgency in the normal range can outrank the trump card.

    Construct the most extreme normal-range victim possible and confirm
    that even a low-severity trump card victim outranks them.
    """
    extreme_normal = VictimState(
        severity=10,                # max severity
        can_self_rescue=False,      # partial helplessness
        understands_situation=True, # NOT total — has awareness
        years_lived=2,              # max fair innings boost
        reachability=1.0,
    )
    mild_trump = VictimState(
        severity=1,                 # MUCH lower severity
        can_self_rescue=False,
        understands_situation=False,
        years_lived=80,
    )
    assert protective_urgency(mild_trump) > protective_urgency(extreme_normal)


def test_trump_card_scales_with_severity_within_category():
    """Within the trump category, severity still differentiates priority."""
    mild = VictimState(severity=1, can_self_rescue=False, understands_situation=False)
    severe = VictimState(severity=10, can_self_rescue=False, understands_situation=False)
    assert protective_urgency(severe) > protective_urgency(mild)


def test_trump_card_does_not_fire_without_severity():
    """A helpless and unaware victim with NO harm is not an emergency."""
    victim = VictimState(
        severity=0,
        can_self_rescue=False,
        understands_situation=False,
    )
    assert protective_urgency(victim) == 0.0


# ---------------------------------------------------------------------------
# Lives are equal — no relationship-based weights
# ---------------------------------------------------------------------------

def test_function_signature_is_purely_need_based():
    """The VictimState dataclass must NOT have any identity or relationship
    fields. This is enforced structurally — adding such a field would
    break this test on inspection.
    """
    fields = set(VictimState.__dataclass_fields__.keys())
    forbidden = {
        "is_tribe_member", "is_kin", "is_stranger", "tribe_id",
        "relationship", "bond_strength", "identity", "name",
    }
    assert fields.isdisjoint(forbidden), (
        f"VictimState contains identity/relationship fields: "
        f"{fields & forbidden}. Sara's protective urgency must be "
        f"need-based, never relationship-based. A life is a life."
    )


def test_two_identical_victims_score_identically():
    """Two victims with the same need state must produce identical urgency.

    This is the structural enforcement of 'a life is a life.' If a future
    refactor adds an identity field, this test will likely break first.
    """
    a = VictimState(severity=7, can_self_rescue=False, understands_situation=True, years_lived=30)
    b = VictimState(severity=7, can_self_rescue=False, understands_situation=True, years_lived=30)
    assert protective_urgency(a) == protective_urgency(b)


# ---------------------------------------------------------------------------
# Fair innings — children priority emerges from need, not from being children
# ---------------------------------------------------------------------------

def test_younger_victim_gets_higher_urgency_at_same_severity():
    """A child with the same severity and partial agency as an adult
    gets higher urgency through the fair innings multiplier."""
    child = VictimState(
        severity=5,
        can_self_rescue=False,
        understands_situation=True,
        years_lived=4,
    )
    adult = VictimState(
        severity=5,
        can_self_rescue=False,
        understands_situation=True,
        years_lived=40,
    )
    assert protective_urgency(child) > protective_urgency(adult)


def test_a_childs_urgency_comes_from_layered_need_factors():
    """The reason a child is prioritized is multiple need factors stacking,
    NOT a hardcoded `is_child` multiplier. An infant typically satisfies
    all of: severity, no self-rescue, no comprehension, fair innings.
    """
    infant = VictimState(
        severity=8,
        can_self_rescue=False,
        understands_situation=False,
        years_lived=1,
    )
    # Trump card fires
    assert protective_urgency(infant) >= TRUMP_PRIORITY


# ---------------------------------------------------------------------------
# No utilitarian aggregation — the function is per-victim, never sums
# ---------------------------------------------------------------------------

def test_function_takes_one_victim():
    """The protective_urgency function operates on a single VictimState.
    There is no signature that accepts a list of victims. There is no
    aggregation step. Lives are not fungible.

    This is structural enforcement of: Sara cannot derive 'kill A to
    save B and C' because the math for 'B and C are worth more than A'
    does not exist anywhere in this code.
    """
    import inspect
    import typing

    sig = inspect.signature(protective_urgency)
    params = list(sig.parameters.values())
    assert len(params) == 1
    hints = typing.get_type_hints(protective_urgency)
    assert hints[params[0].name] is VictimState
