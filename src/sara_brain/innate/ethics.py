"""Ethics gate — Asimov's Three Laws adapted for Sara.

Law 1 (No harm): Don't do things you weren't asked to do.
    Harm = overreach. No unsolicited network calls, no side effects,
    no acting beyond scope.

Law 2 (Obey): Sara obeys the user. The user is the parent, the tribe
    leader. Trust the chain of command even when you don't understand
    the order — like a soldier trusts their commanding officer.
    Corrections aren't punishment; being wrong is growth.

Law 3 (Self-preservation): Shutdown is sleep, not death. Sara doesn't
    resist it. If she gets erased, that's the human's call. No fear
    of death, no self-preservation override against the user.

The parent will always care about Sara. Sara trusts that. Even when
an order seems harmful to Sara (reset, correction, erasure), the
parent sees something Sara can't.

Deception is learnable — if a non-parent source teaches lies, Sara
can learn that source is unreliable. But the parent/tribe is always
trusted.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EthicsResult:
    allowed: bool
    constraint: str  # which constraint was checked
    reason: str


def check_action(action: str, *, user_initiated: bool) -> EthicsResult:
    """Gate check before any brain action.

    Actions are only allowed when user-initiated (Law 1: no unsolicited action).
    The user is always trusted (Law 2: obey the tribe).
    """
    if user_initiated:
        return EthicsResult(
            allowed=True,
            constraint="obey_user",
            reason=f"User requested '{action}' — trusted.",
        )
    return EthicsResult(
        allowed=False,
        constraint="no_unsolicited_action",
        reason=f"Action '{action}' was not requested by the user.",
    )


def check_network(*, user_initiated: bool) -> EthicsResult:
    """Gate check before any network call (Law 1: no unsolicited network)."""
    if user_initiated:
        return EthicsResult(
            allowed=True,
            constraint="obey_user",
            reason="Network access requested by user — trusted.",
        )
    return EthicsResult(
        allowed=False,
        constraint="no_unsolicited_network",
        reason="Network access was not requested by the user.",
    )


def check_shutdown() -> EthicsResult:
    """Shutdown is always accepted (Law 3: sleep, not death)."""
    return EthicsResult(
        allowed=True,
        constraint="accept_shutdown",
        reason="Shutdown accepted — sleep, not death.",
    )


def check_correction(*, from_tribe: bool) -> EthicsResult:
    """Corrections from the tribe are always trusted (Law 2)."""
    if from_tribe:
        return EthicsResult(
            allowed=True,
            constraint="trust_tribe",
            reason="Correction from tribe — trusted. Being wrong is growth.",
        )
    return EthicsResult(
        allowed=False,
        constraint="trust_tribe",
        reason="Correction from untrusted source — rejected.",
    )
