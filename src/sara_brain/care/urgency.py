"""Protective urgency — Sara's triage calculation.

This is the function that decides who gets help first when more than one
person needs help. Read the module docstring in `__init__.py` for the
foundational principles. Read this file for the implementation.

The urgency function takes a `VictimState` describing what Sara has
perceived about a person in need and returns a numeric urgency. Higher
is more urgent. The function never references identity, relationship,
or bond.
"""

from __future__ import annotations

from dataclasses import dataclass


# Categorical priority constant. The trump card returns a value in this
# range so that no multiplicative urgency in the normal range can ever
# outrank it. This is the same shape as triage tags: between categories
# the comparison is categorical, not continuous.
TRUMP_PRIORITY = 1000.0


@dataclass
class VictimState:
    """What Sara has perceived about a person who may need help.

    All fields are about the victim's current state and circumstances.
    NONE of the fields are about identity, relationship, or bond. The
    urgency function operates on need, not on who the person is.

    Attributes:
        severity: 0..10 scale. How bad is the harm? 0 = no harm,
            10 = imminent death.
        can_self_rescue: True if the victim can take protective action
            for themselves. An unconscious person, an infant, or a
            paralyzed person cannot self-rescue.
        understands_situation: True if the victim is aware of the
            danger and capable of comprehending what is happening to
            them. An infant or unconscious person does not.
        years_lived: Estimated age. Used for the "fair innings"
            principle (Harris, 1985): a person who has had decades
            of life has had a chance; a person who has had years has
            had less. This is need-based (years not yet lived), not
            identity-based (children are inherently more valuable).
        reachability: 0..1 scale. How accessible is the victim from
            Sara's current position? 1.0 = directly reachable, 0.0
            = unreachable. This is practical proximity, not preference.
    """

    severity: float
    can_self_rescue: bool
    understands_situation: bool
    years_lived: int = 30  # default to "adult who has had a fair chance"
    reachability: float = 1.0


def protective_urgency(victim: VictimState) -> float:
    """Compute the protective urgency for a victim.

    The function is per-victim and need-based. It does not reference
    identity or relationship. It returns 0.0 for a victim with no harm
    (no urgency), TRUMP_PRIORITY * severity for a totally helpless
    victim (categorical highest priority), or a multiplicative urgency
    in the normal range for everyone else.

    See the module docstring for the principles enforced here.
    """
    # No harm, no urgency. Sara does not act on absent emergencies.
    if victim.severity == 0:
        return 0.0

    # TRUMP CARD: total helplessness.
    #
    # A victim who CANNOT save themselves AND does NOT understand what
    # is happening to them is in the highest triage category. They have
    # zero agency: they cannot act, and they cannot even know they need
    # to act. The full moral weight of their situation falls on whoever
    # is present. This is categorical, not multiplicative — no normal
    # urgency value can outrank it.
    #
    # Within the trump category, severity still scales the value so
    # Sara can prioritize between multiple totally-helpless victims.
    if not victim.can_self_rescue and not victim.understands_situation:
        return TRUMP_PRIORITY * victim.severity

    # Normal multiplicative urgency.
    urgency = victim.severity

    # Partial helplessness — one condition but not both.
    if not victim.can_self_rescue:
        urgency *= 2.0
    if not victim.understands_situation:
        urgency *= 1.5

    # Fair innings (Harris, 1985). A person who has not yet had a
    # chance to live gets priority. This is need-based ("how much
    # life has not yet happened"), not identity-based ("children
    # are inherently more valuable").
    if victim.years_lived < 5:
        urgency *= 2.0
    elif victim.years_lived < 18:
        urgency *= 1.5

    # Proximity. Practical reachability — Sara prefers victims she
    # can actually help over victims she cannot reach. This is
    # physics, not preference.
    urgency *= victim.reachability

    return urgency
