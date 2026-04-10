from __future__ import annotations

from dataclasses import dataclass, field

from .neuron import Neuron


@dataclass
class PathTrace:
    """A single path of thought that led to a conclusion."""
    neurons: list[Neuron]
    source_text: str | None = None
    weight: float = 0.0  # signed sum of segment strengths along this path

    def labels(self) -> list[str]:
        return [n.label for n in self.neurons]

    @property
    def is_refuted(self) -> bool:
        """True if this path's signed weight is negative — Sara knows it's false."""
        return self.weight < 0

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

    @property
    def signed_confidence(self) -> float:
        """Sum of signed weights across all converging paths.

        Positive paths add confidence; refuted (negative) paths subtract it.
        A concept reached only by refuted paths has negative confidence —
        Sara knows it as actively false.
        """
        return sum(p.weight for p in self.converging_paths)

    @property
    def is_refuted(self) -> bool:
        """True if this concept is dominated by refuted paths."""
        return self.signed_confidence < 0

    def __str__(self) -> str:
        lines = [
            f"{self.neuron.label} "
            f"({self.confidence} paths, signed weight {self.signed_confidence:+.2f})"
        ]
        for trace in self.converging_paths:
            lines.append(f"  {trace}")
        return "\n".join(lines)
