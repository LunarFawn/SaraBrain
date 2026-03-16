from __future__ import annotations

from dataclasses import dataclass, field

from .neuron import Neuron


@dataclass
class PathTrace:
    """A single path of thought that led to a conclusion."""
    neurons: list[Neuron]
    source_text: str | None = None

    def labels(self) -> list[str]:
        return [n.label for n in self.neurons]

    def __str__(self) -> str:
        return " → ".join(self.labels())


@dataclass
class RecognitionResult:
    """A recognized concept with all the paths that led to it."""
    neuron: Neuron
    converging_paths: list[PathTrace] = field(default_factory=list)

    @property
    def confidence(self) -> int:
        """Number of independent paths converging on this neuron."""
        return len(self.converging_paths)

    def __str__(self) -> str:
        lines = [f"{self.neuron.label} ({self.confidence} converging paths)"]
        for trace in self.converging_paths:
            lines.append(f"  {trace}")
        return "\n".join(lines)
