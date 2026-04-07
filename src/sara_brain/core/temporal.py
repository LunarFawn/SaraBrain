"""Temporal system: time as first-class neurons in the knowledge graph.

Date neurons (e.g., day_2026_04_06) are CONCEPT neurons. Facts are linked
to them via 'happened_on' segments. Querying "what happened yesterday?"
is a standard wavefront from the date neuron.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta

from ..models.neuron import NeuronType
from ..models.path import Path, PathStep
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo


def _date_label(d: date) -> str:
    """Convert a date to a neuron label: day_2026_04_06."""
    return f"day_{d.year}_{d.month:02d}_{d.day:02d}"


def _period_label() -> str:
    """Return current time-of-day label: morning, afternoon, evening, night."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    elif hour < 21:
        return "evening"
    else:
        return "night"


class TemporalResolver:
    """Resolve natural language time references to date neuron labels."""

    _RELATIVE_WORDS = {
        "today": 0,
        "yesterday": -1,
        "day before yesterday": -2,
        "tomorrow": 1,
    }

    _PERIOD_WORDS = ("morning", "afternoon", "evening", "night")

    def resolve(self, text: str) -> tuple[str | None, str | None]:
        """Return (date_label, period_label) from a time reference.

        Returns (None, None) if the text doesn't contain a time reference.
        """
        text_lower = text.lower().strip()

        date_label = None
        period_label = None

        # Check relative day words
        for word, offset in self._RELATIVE_WORDS.items():
            if word in text_lower:
                target = date.today() + timedelta(days=offset)
                date_label = _date_label(target)
                break

        # Check day-of-week references
        if date_label is None:
            date_label = self._resolve_weekday(text_lower)

        # Check period references
        for period in self._PERIOD_WORDS:
            if period in text_lower:
                period_label = period
                break

        # "this morning" / "this afternoon" implies today
        if period_label and date_label is None:
            if "this" in text_lower or "last" not in text_lower:
                date_label = _date_label(date.today())

        return date_label, period_label

    def _resolve_weekday(self, text: str) -> str | None:
        """Resolve 'last monday', 'tuesday', etc. to a date neuron label."""
        days = ["monday", "tuesday", "wednesday", "thursday",
                "friday", "saturday", "sunday"]
        today = date.today()
        for i, day_name in enumerate(days):
            if day_name in text:
                # Calculate how many days back to that weekday
                current_weekday = today.weekday()
                diff = (current_weekday - i) % 7
                if diff == 0:
                    diff = 7  # "monday" means last monday if today is monday
                if "last" in text:
                    diff = (current_weekday - i) % 7
                    if diff == 0:
                        diff = 7
                target = today - timedelta(days=diff)
                return _date_label(target)
        return None


class TemporalLinker:
    """Creates temporal links between learned facts and date neurons."""

    def __init__(self, neuron_repo: NeuronRepo, segment_repo: SegmentRepo,
                 path_repo: PathRepo) -> None:
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.path_repo = path_repo

    def link_to_now(self, concept_neuron_id: int,
                    account_id: int | None = None,
                    trust_status: str | None = None) -> int:
        """Link a concept to today's date neuron and current time period.

        Creates: concept -> happened_on -> day_YYYY_MM_DD
        Optionally: concept -> happened_during -> morning/afternoon/etc.

        Returns the date path ID.
        """
        today_label = _date_label(date.today())
        period = _period_label()

        # Get or create date neuron
        date_neuron, _ = self.neuron_repo.get_or_create(
            today_label, NeuronType.CONCEPT
        )

        # Get or create period neuron
        period_neuron, _ = self.neuron_repo.get_or_create(
            period, NeuronType.PROPERTY
        )

        # Get the concept neuron
        concept_neuron = self.neuron_repo.get_by_id(concept_neuron_id)
        if concept_neuron is None:
            raise ValueError(f"Neuron {concept_neuron_id} not found")

        # Create segment: concept -> happened_on -> date
        seg_date, _ = self.segment_repo.get_or_create(
            concept_neuron.id, date_neuron.id, "happened_on"
        )

        # Create segment: concept -> happened_during -> period
        seg_period, _ = self.segment_repo.get_or_create(
            concept_neuron.id, period_neuron.id, "happened_during"
        )

        # Record the temporal path
        path = Path(
            id=None,
            origin_id=concept_neuron.id,
            terminus_id=date_neuron.id,
            source_text=f"[temporal] {concept_neuron.label} on {today_label}",
            account_id=account_id,
            trust_status=trust_status,
        )
        path = self.path_repo.create(path)
        self.path_repo.add_step(
            PathStep(id=None, path_id=path.id, step_order=0, segment_id=seg_date.id)
        )

        return path.id


def temporal_query(brain, time_reference: str) -> list:
    """Resolve a time reference and return all paths leading to that date neuron.

    Uses brain.why() to find everything linked to the date.
    """
    resolver = TemporalResolver()
    date_label, period_label = resolver.resolve(time_reference)

    results = []
    if date_label:
        traces = brain.why(date_label)
        results.extend(traces)

    if period_label and date_label:
        # Also check the period neuron for more specific results
        period_traces = brain.why(period_label)
        # Filter to only traces that also connect to the date
        for trace in period_traces:
            if trace not in results:
                results.append(trace)

    return results
