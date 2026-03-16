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
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    def strengthen(self) -> None:
        """Increment traversals and recalculate strength. Strength only goes up."""
        self.traversals += 1
        self.strength = 1.0 + math.log(1 + self.traversals)
        self.last_used = time.time()
