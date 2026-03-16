from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum


class NeuronType(str, Enum):
    CONCEPT = "concept"
    PROPERTY = "property"
    RELATION = "relation"
    ASSOCIATION = "association"


@dataclass
class Neuron:
    id: int | None
    label: str
    neuron_type: NeuronType
    created_at: float = field(default_factory=time.time)
    metadata: dict | None = None

    def metadata_json(self) -> str | None:
        if self.metadata is None:
            return None
        return json.dumps(self.metadata)

    @staticmethod
    def metadata_from_json(raw: str | None) -> dict | None:
        if raw is None:
            return None
        return json.loads(raw)
