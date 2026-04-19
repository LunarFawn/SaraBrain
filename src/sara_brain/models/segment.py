from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass
class Segment:
    id: int | None
    source_id: int
    target_id: int
    relation: str
    strength: float = 1.0
    traversals: int = 0
    refutations: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    # operation_tag — optional arithmetic hint. When a segment represents
    # a relation like "reduces chromosome number by half", this field
    # carries the primitive + operand as a string ("multiply:0.5") that
    # MathCompute interprets at query time. None for the vast majority
    # of biological/descriptive segments.
    operation_tag: str | None = None

    def _recalculate(self) -> None:
        """Recalculate strength from traversals and refutations.

        Symmetric formula:
            strength = 1 + ln(1 + traversals) - ln(1 + refutations)

        - Fresh segment: 1.0 (baseline)
        - Validated 100x: ~5.6
        - Refuted 100x: ~-3.6
        - Both 100x: cancels back to 1.0
        """
        self.strength = (
            1.0
            + math.log(1 + self.traversals)
            - math.log(1 + self.refutations)
        )

    def strengthen(self) -> None:
        """Increment traversals and recalculate strength."""
        self.traversals += 1
        self._recalculate()
        self.last_used = time.time()

    def weaken(self) -> None:
        """Increment refutations and recalculate strength.

        Sara never deletes — she marks things as known-to-be-false.
        Strength can go negative when refutations exceed traversals.
        """
        self.refutations += 1
        self._recalculate()
        self.last_used = time.time()

    @property
    def is_refuted(self) -> bool:
        """True if the segment has been refuted more than validated."""
        return self.refutations > self.traversals

    @property
    def belief(self) -> float:
        """Direction of evidence: positive = believed, negative = refuted, ~0 = contested or unknown.

        This is the same as `strength - 1.0`. Exposed as a separate property
        because the contested-vs-fresh distinction depends on pairing belief
        with evidence_weight.
        """
        return math.log(1 + self.traversals) - math.log(1 + self.refutations)

    @property
    def evidence_weight(self) -> float:
        """Total information: how much we have heard, regardless of side.

        This is what distinguishes a fresh segment (T=0, R=0) from a heavily
        contested one (T=100, R=100). Both have belief ~ 0, but the contested
        segment has high evidence_weight while the fresh one is near zero.
        """
        return math.log(1 + self.traversals + self.refutations)

    @property
    def epistemic_state(self) -> str:
        """Categorical epistemic state derived from belief and evidence weight.

        - "unknown"   — fresh segment, little evidence either way
        - "believed"  — evidence leans positive
        - "refuted"   — evidence leans negative
        - "contested" — high evidence on both sides, no resolution

        This is the fix for the contested-vs-fresh bug: a segment with
        T=100, R=100 has the same strength (1.0) as a fresh segment but
        is in a fundamentally different epistemic state.
        """
        if self.evidence_weight < 1.0:
            return "unknown"
        if abs(self.belief) < 0.3:
            return "contested"
        return "believed" if self.belief > 0 else "refuted"
