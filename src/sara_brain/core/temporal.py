"""Temporal system — time as paths grounding in TEMPORAL primitives.

Maps to hippocampal time cells + entorhinal cortex + SCN.

A baby understands "before" and "after" before it knows what a
"Tuesday" is. The capacity for temporal reasoning is innate. The
calendar is culture. This module bridges them: it detects time
references in natural language and creates paths that ground
learned temporal concepts (dates, periods, eras) in innate TEMPORAL
primitives (before, after, during, moment, era).

When Sara ingests "Akkadians conquered Sumer around 2334 BCE",
the temporal linker creates:

    2334 bce → is_a → moment    (TEMPORAL primitive)
    akkadian conquest → happened_during → 2334 bce

When Sara later ingests "the edubba taught Sumerian in the early
period", the linker creates:

    early sumerian period → is_a → era    (TEMPORAL primitive)
    edubba taught sumerian → happened_during → early sumerian period

Now "what happened before the Akkadians" follows the `before` links
and finds everything in the earlier era.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from ..models.neuron import NeuronType
from ..models.path import Path, PathStep
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo


# ── Date label helpers ──

def _date_label(d: date) -> str:
    """Convert a date to a neuron label: day_2026_04_06."""
    return f"day_{d.year}_{d.month:02d}_{d.day:02d}"


def _period_label() -> str:
    """Return current time-of-day label."""
    hour = datetime.now().hour
    if hour < 12:
        return "morning"
    elif hour < 17:
        return "afternoon"
    elif hour < 21:
        return "evening"
    else:
        return "night"


# ── Historical date patterns ──

# Matches patterns like "3500 BCE", "2334 BC", "around 1200 BCE",
# "approximately 4500-3100 BCE", "circa 2000 BC"
_YEAR_BCE = re.compile(
    r"(?:around|approximately|circa|~)?\s*(\d{3,4})\s*(?:BCE|BC|B\.C\.E?\.?)\b",
    re.IGNORECASE,
)

# Matches patterns like "1990 CE", "2026 AD", or just "in 2026"
_YEAR_CE = re.compile(
    r"(?:around|approximately|circa|~)?\s*(\d{3,4})\s*(?:CE|AD|A\.D\.?)?\b",
    re.IGNORECASE,
)

# Matches period/era references
_ERA_PATTERNS = [
    (re.compile(r"\b(early|late|middle)\s+([\w\s]+?)\s+period\b", re.IGNORECASE), "era"),
    (re.compile(r"\b(bronze|iron|stone|ice)\s+age\b", re.IGNORECASE), "era"),
    (re.compile(r"\b(\w+)\s+dynasty\b", re.IGNORECASE), "era"),
    (re.compile(r"\b(\w+)\s+empire\b", re.IGNORECASE), "era"),
    (re.compile(r"\b(\w+)\s+kingdom\b", re.IGNORECASE), "era"),
]

# Matches temporal relationship words
_TEMPORAL_RELATIONS = {
    "before": "before",
    "after": "after",
    "during": "during",
    "while": "during",
    "until": "until",
    "since": "since",
    "prior to": "before",
    "following": "after",
    "preceded": "before",
    "succeeded": "after",
}


class TemporalResolver:
    """Resolve time references in text to temporal neuron labels and relations."""

    def extract_dates(self, text: str) -> list[dict]:
        """Extract all date/time references from text.

        Returns list of dicts with keys:
            label:    neuron label for the date (e.g., "3500_bce")
            type:     "moment", "era", "period"
            original: the matched text
        """
        results = []
        seen = set()
        bce_years = set()  # track BCE years to avoid CE false positives

        # BCE dates first
        for m in _YEAR_BCE.finditer(text):
            year = m.group(1)
            label = f"{year}_bce"
            bce_years.add(year)
            if label not in seen:
                seen.add(label)
                results.append({
                    "label": label,
                    "type": "moment",
                    "original": m.group(0).strip(),
                })

        # CE dates — skip years already caught as BCE
        for m in _YEAR_CE.finditer(text):
            year = m.group(1)
            if year in bce_years:
                continue  # already matched as BCE
            if len(year) == 4 and int(year) > 1000:
                label = f"{year}_ce"
                if label not in seen:
                    seen.add(label)
                    results.append({
                        "label": label,
                        "type": "moment",
                        "original": m.group(0).strip(),
                    })

        # Era/period references
        for pattern, temporal_type in _ERA_PATTERNS:
            for m in pattern.finditer(text):
                label = m.group(0).lower().replace(" ", "_")
                if label not in seen:
                    seen.add(label)
                    results.append({
                        "label": label,
                        "type": temporal_type,
                        "original": m.group(0).strip(),
                    })

        # Relative day references
        text_lower = text.lower()
        for word, offset in [("today", 0), ("yesterday", -1), ("tomorrow", 1)]:
            if word in text_lower:
                target = date.today() + timedelta(days=offset)
                label = _date_label(target)
                if label not in seen:
                    seen.add(label)
                    results.append({
                        "label": label,
                        "type": "moment",
                        "original": word,
                    })

        return results

    def extract_temporal_relations(self, text: str) -> list[dict]:
        """Extract temporal relationship phrases from text.

        Returns list of dicts with keys:
            relation: the TEMPORAL primitive (before, after, during, etc.)
            original: the matched text
        """
        results = []
        text_lower = text.lower()
        for phrase, relation in _TEMPORAL_RELATIONS.items():
            if phrase in text_lower:
                results.append({
                    "relation": relation,
                    "original": phrase,
                })
        return results


class TemporalLinker:
    """Creates temporal paths grounding learned dates in TEMPORAL primitives."""

    def __init__(self, neuron_repo: NeuronRepo, segment_repo: SegmentRepo,
                 path_repo: PathRepo) -> None:
        self.neuron_repo = neuron_repo
        self.segment_repo = segment_repo
        self.path_repo = path_repo
        self.resolver = TemporalResolver()

    def link_fact_to_time(
        self,
        fact_neuron_id: int,
        text: str,
    ) -> list[int]:
        """Extract temporal references from text and link the fact to them.

        Creates paths grounding each date/era in the appropriate TEMPORAL
        primitive (moment, era, period) and linking the fact neuron to
        the date via happened_during/happened_before/etc.

        Returns list of created path IDs.
        """
        dates = self.resolver.extract_dates(text)
        relations = self.resolver.extract_temporal_relations(text)
        path_ids = []

        for date_ref in dates:
            # Get or create the date neuron
            date_neuron, _ = self.neuron_repo.get_or_create(
                date_ref["label"], NeuronType.CONCEPT
            )

            # Ground the date in its TEMPORAL primitive type
            type_neuron, _ = self.neuron_repo.get_or_create(
                date_ref["type"], NeuronType.PROPERTY
            )
            seg_type, _ = self.segment_repo.get_or_create(
                date_neuron.id, type_neuron.id, "is_a"
            )

            # Link the fact to the date
            relation = "happened_during"  # default
            if relations:
                relation = f"happened_{relations[0]['relation']}"

            seg_link, _ = self.segment_repo.get_or_create(
                fact_neuron_id, date_neuron.id, relation
            )

            # Create the temporal path
            path = Path(
                id=None,
                origin_id=fact_neuron_id,
                terminus_id=date_neuron.id,
                source_text=f"[temporal] linked to {date_ref['label']} from: {text[:100]}",
            )
            path = self.path_repo.create(path)
            self.path_repo.add_step(
                PathStep(id=None, path_id=path.id, step_order=0, segment_id=seg_link.id)
            )
            path_ids.append(path.id)

        return path_ids

    def link_to_now(self, concept_neuron_id: int) -> int:
        """Link a concept to today's date and current time period.

        Creates: concept → happened_on → day_YYYY_MM_DD

        Returns the path ID.
        """
        today_label = _date_label(date.today())
        period = _period_label()

        date_neuron, _ = self.neuron_repo.get_or_create(
            today_label, NeuronType.CONCEPT
        )
        period_neuron, _ = self.neuron_repo.get_or_create(
            period, NeuronType.PROPERTY
        )

        # Ground today in TEMPORAL primitive
        moment_neuron, _ = self.neuron_repo.get_or_create(
            "moment", NeuronType.PROPERTY
        )
        self.segment_repo.get_or_create(
            date_neuron.id, moment_neuron.id, "is_a"
        )

        # Link concept to date
        seg_date, _ = self.segment_repo.get_or_create(
            concept_neuron_id, date_neuron.id, "happened_on"
        )

        path = Path(
            id=None,
            origin_id=concept_neuron_id,
            terminus_id=date_neuron.id,
            source_text=f"[temporal] linked to {today_label}",
        )
        path = self.path_repo.create(path)
        self.path_repo.add_step(
            PathStep(id=None, path_id=path.id, step_order=0, segment_id=seg_date.id)
        )
        return path.id
