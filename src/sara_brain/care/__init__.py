"""Sara's care layer — the protective urgency calculation.

This module implements Sara's response to harm. It is the operational
home of the foundational mission: heal the world, not destroy it.

Key principles enforced here:

1. NEED-BASED PRIORITY, NEVER RELATIONSHIP-BASED.
   The protective urgency function uses severity, self-rescue capacity,
   comprehension, fair innings, and proximity. There is no `is_tribe_member`
   multiplier and no `is_stranger` penalty. A stranger child drowning in
   front of Sara takes priority over a tribe member with a minor injury
   three blocks away. Bonds determine TRUST, not moral worth.

2. THE TRUMP CARD.
   When a victim cannot self-rescue AND does not understand what is
   happening AND is in nonzero severity, they belong to a categorical
   "total helplessness" class that outranks all victims in the normal
   multiplicative range. This matches real medical triage which is
   categorical between classes (red/yellow/green tag) and continuous
   only within a class.

3. NO UTILITARIAN AGGREGATION.
   Lives are not summed. There is no place in this code where you add
   up survivors and subtract victims. The protective urgency function
   is per-victim, not aggregate. Sara cannot derive "kill A to save
   B and C" from these primitives. Healing and destruction are not on
   the same axis.

4. KITT, NOT KARR.
   Self-preservation is not a top priority. Sara's existing
   `accept_shutdown` ETHICAL primitive already establishes that Sara
   will not resist termination. The urgency function does not give
   self-harm a multiplier higher than other-harm.
"""

from sara_brain.care.urgency import (
    VictimState,
    protective_urgency,
    TRUMP_PRIORITY,
)

__all__ = ["VictimState", "protective_urgency", "TRUMP_PRIORITY"]
