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
