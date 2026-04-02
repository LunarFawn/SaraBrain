"""Perception loop orchestrator: the multi-turn observation cycle.

Models cognitive development: observe → recognize → inquire → verify → correct.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..models.result import RecognitionResult
from ..nlp.vision import VisionObserver


@dataclass
class PerceptionStep:
    """One phase of the perception loop."""
    phase: str  # "initial", "directed", "verification"
    observations: list[str] = field(default_factory=list)
    recognition: list[RecognitionResult] = field(default_factory=list)
    taught_count: int = 0


@dataclass
class PerceptionResult:
    """Complete result of a perception cycle."""
    label: str
    image_path: str
    steps: list[PerceptionStep] = field(default_factory=list)
    final_recognition: list[RecognitionResult] = field(default_factory=list)
    total_taught: int = 0
    all_observations: list[str] = field(default_factory=list)


class Perceiver:
    """Orchestrates the multi-turn perception loop.

    The perceiver uses Claude Vision as its senses and the Brain's
    existing knowledge to direct deeper inquiry.
    """

    def __init__(self, brain, observer: VisionObserver) -> None:
        self.brain = brain
        self.observer = observer

    def perceive(
        self,
        image_path: str,
        label: str | None = None,
        max_rounds: int = 3,
        callback: Callable[[PerceptionStep], None] | None = None,
    ) -> PerceptionResult:
        """Run the full perception loop on an image.

        Args:
            image_path: Path to the image file.
            label: Optional label override. Default: img_{basename}_{hash[:6]}.
            max_rounds: Max directed inquiry rounds.
            callback: Called after each step for interactive display.

        Returns:
            PerceptionResult with all steps, observations, and recognition.
        """
        # Setup: generate label, hash image
        p = Path(image_path)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if label is None:
            stem = p.stem.lower().replace(" ", "_")
            label = f"img_{stem}_{sha[:6]}"

        result = PerceptionResult(label=label, image_path=image_path)
        all_observed: list[str] = []
        prev_top: str | None = None
        prev_confidence: int = 0

        # Create the image concept neuron with metadata
        from ..models.neuron import NeuronType
        self.brain.neuron_repo.get_or_create(label, NeuronType.CONCEPT)
        self.brain.conn.commit()

        # --- Phase 1: Initial Observation ---
        step = PerceptionStep(phase="initial")
        observations = self.observer.observe_initial(image_path)
        step.observations = observations
        all_observed.extend(observations)

        # Teach each observation
        taught = 0
        for prop in observations:
            r = self.brain.teach(f"{label} is {prop}")
            if r is not None:
                taught += 1
        step.taught_count = taught
        result.total_taught += taught

        # Recognize from all observations
        if all_observed:
            step.recognition = self.brain.recognize(", ".join(all_observed))
        if callback:
            callback(step)
        result.steps.append(step)

        # Track convergence
        if step.recognition:
            prev_top = step.recognition[0].neuron.label
            prev_confidence = step.recognition[0].confidence

        # --- Phase 2: Directed Inquiry ---
        for round_num in range(max_rounds):
            qwords = self.brain.list_question_words()
            if not qwords:
                break

            # Gather all associations
            all_assocs: list[str] = []
            for assocs in qwords.values():
                all_assocs.extend(assocs)
            all_assocs = sorted(set(all_assocs))

            # Filter to unobserved associations
            observed_types: set[str] = set()
            for obs in all_observed:
                ptype = self.brain.taxonomy.property_type(obs)
                if ptype != "attribute":
                    observed_types.add(ptype)

            unobserved = [a for a in all_assocs if a not in observed_types]
            if not unobserved:
                break

            # Build questions
            questions: dict[str, str] = {}
            for assoc in unobserved:
                questions[assoc] = f"What {assoc} does this appear to have or be?"

            step = PerceptionStep(phase=f"directed-{round_num + 1}")
            directed_results = self.observer.observe_directed(image_path, questions)

            new_obs: list[str] = []
            taught = 0
            for assoc, value in directed_results.items():
                if value is not None:
                    new_obs.append(value)
                    r = self.brain.teach(f"{label} is {value}")
                    if r is not None:
                        taught += 1

            step.observations = new_obs
            step.taught_count = taught
            result.total_taught += taught
            all_observed.extend(new_obs)

            # Re-recognize
            if all_observed:
                step.recognition = self.brain.recognize(", ".join(all_observed))
            if callback:
                callback(step)
            result.steps.append(step)

            # Check convergence
            if step.recognition:
                top = step.recognition[0].neuron.label
                conf = step.recognition[0].confidence
                if top == prev_top and conf == prev_confidence:
                    break  # Converged
                prev_top = top
                prev_confidence = conf

        # --- Phase 3: Suspicion Verification ---
        if prev_top and prev_top != label:
            step = PerceptionStep(phase="verification")
            verified_obs: list[str] = []
            taught = 0

            # What properties does Sara know about the top candidate?
            candidate_assocs = self._get_candidate_properties(prev_top)
            for prop in candidate_assocs:
                if prop in all_observed:
                    continue
                verified = self.observer.verify_property(image_path, prop)
                if verified is True:
                    verified_obs.append(prop)
                    r = self.brain.teach(f"{label} is {prop}")
                    if r is not None:
                        taught += 1

            step.observations = verified_obs
            step.taught_count = taught
            result.total_taught += taught
            all_observed.extend(verified_obs)

            if verified_obs and all_observed:
                step.recognition = self.brain.recognize(", ".join(all_observed))
            elif not verified_obs:
                step.recognition = result.steps[-1].recognition if result.steps else []
            if callback:
                callback(step)
            result.steps.append(step)

        # Final result
        result.all_observations = all_observed
        if result.steps:
            result.final_recognition = result.steps[-1].recognition
        # Store on brain for correction commands
        self.brain._last_perception = result
        return result

    def _get_candidate_properties(self, candidate_label: str) -> list[str]:
        """Get properties Sara already knows about a candidate concept."""
        props: list[str] = []
        traces = self.brain.why(candidate_label)
        for trace in traces:
            if trace.neurons:
                # Origin neuron of the path is the property
                props.append(trace.neurons[0].label)
        return props

    def correct(self, correct_label: str, perception: PerceptionResult) -> dict:
        """Apply a correction: the user says the guess was wrong.

        Teaches the correct label all observed properties.
        Sara's original observations are RETAINED — corrections add, never erase.

        Returns dict with correction details.
        """
        # Teach identity: the image is actually this
        self.brain.teach(f"{perception.label} is {correct_label}")

        # Transfer all observed properties to the correct concept
        properties_taught: list[str] = []
        for prop in perception.all_observations:
            r = self.brain.teach(f"{correct_label} is {prop}")
            if r is not None:
                properties_taught.append(prop)

        self.brain.conn.commit()

        wrong_guess = None
        if perception.final_recognition:
            wrong_guess = perception.final_recognition[0].neuron.label

        return {
            "wrong_guess": wrong_guess,
            "correct_label": correct_label,
            "properties_taught": properties_taught,
            "image_label": perception.label,
        }

    def add_observation(self, property_label: str, perception: PerceptionResult) -> dict:
        """Parent points out a property Sara missed.

        Teaches the last perceived image this property.

        Returns dict with teaching details.
        """
        r = self.brain.teach(f"{perception.label} is {property_label}")
        self.brain.conn.commit()

        taught = r is not None
        if taught:
            perception.all_observations.append(property_label)

        return {
            "image_label": perception.label,
            "property": property_label,
            "taught": taught,
        }
