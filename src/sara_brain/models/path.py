from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Path:
    id: int | None
    origin_id: int
    terminus_id: int
    source_text: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class PathStep:
    id: int | None
    path_id: int
    step_order: int
    segment_id: int
