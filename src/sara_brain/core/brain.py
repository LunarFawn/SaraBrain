"""Brain: main orchestrator and public API."""

from __future__ import annotations

from ..models.neuron import NeuronType
from ..models.result import RecognitionResult, PathTrace
from ..parsing.statement_parser import StatementParser
from ..parsing.taxonomy import Taxonomy
from ..storage.database import Database
from ..storage.neuron_repo import NeuronRepo
from ..storage.segment_repo import SegmentRepo
from ..storage.path_repo import PathRepo
from ..storage.association_repo import AssociationRepo
from ..storage.category_repo import CategoryRepo
from ..storage.settings_repo import SettingsRepo
from ..storage.account_repo import AccountRepo
from ..storage.interaction_repo import InteractionRepo
from .learner import Learner, LearnResult
from .recognizer import Recognizer
from .similarity import SimilarityAnalyzer, SimilarityLink


# Maps account role to trust status for paths
_ROLE_TRUST_MAP: dict[str, str] = {
    "reader": "observed",
    "teacher": "taught",
    "doctor": "verified",
}


# Default question words for built-in taxonomy property types
_BUILTIN_QUESTION_WORDS: dict[str, str] = {
    "color": "what",
    "taste": "how",
    "shape": "what",
    "texture": "how",
    "size": "what",
    "temperature": "how",
}


class Brain:
    """The main entry point for Sara Brain.

    Every mutation writes to SQLite immediately. On restart, full state is recovered.
    """

    _last_perception = None  # Set by Perceiver after perceive()

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db = Database(db_path)
        self.conn = self.db.conn

        # Repos
        self.neuron_repo = NeuronRepo(self.conn)
        self.segment_repo = SegmentRepo(self.conn)
        self.path_repo = PathRepo(self.conn)
        self.association_repo = AssociationRepo(self.conn)
        self.category_repo = CategoryRepo(self.conn)
        self.settings_repo = SettingsRepo(self.conn)
        self.account_repo = AccountRepo(self.conn)
        self.interaction_repo = InteractionRepo(self.conn)

        # Innate layer — hardwired, survives reset
        from ..innate.primitives import get_all
        self.innate = get_all()

        # Ethics gate — Asimov's Laws adapted for Sara
        from ..innate import ethics as _ethics
        self._ethics = _ethics

        # Taxonomy & parser
        self.taxonomy = Taxonomy()
        self.parser = StatementParser(self.taxonomy)

        # Core algorithms
        self.learner = Learner(self.parser, self.neuron_repo, self.segment_repo, self.path_repo)
        self.recognizer = Recognizer(self.neuron_repo, self.segment_repo)
        self.similarity = SimilarityAnalyzer(self.neuron_repo, self.segment_repo, self.conn)

        # Load dynamic associations and categories from DB
        self._load_dynamic_associations()
        self._load_categories()

    def teach(self, statement: str, *, user_initiated: bool = True,
              account_id: int | None = None) -> LearnResult | None:
        """Teach a fact. Returns None if unparseable.

        If account_id is provided, trust_status is set automatically
        based on the account's role:
          reader  -> 'observed'
          teacher -> 'taught'
          doctor  -> 'verified'
        """
        gate = self._ethics.check_action("teach", user_initiated=user_initiated)
        if not gate.allowed:
            raise PermissionError(gate.reason)

        trust_status = None
        if account_id is not None:
            account = self.account_repo.get_by_id(account_id)
            if account is not None:
                trust_status = _ROLE_TRUST_MAP.get(account.role)

        result = self.learner.learn(statement, account_id=account_id,
                                    trust_status=trust_status)
        if result is not None:
            self.conn.commit()
        return result

    def recognize(self, inputs: str) -> list[RecognitionResult]:
        """Recognize from comma-separated input labels."""
        labels = [l.strip() for l in inputs.split(",") if l.strip()]
        results = self.recognizer.recognize(labels)
        self.conn.commit()
        return results

    def trace(self, label: str) -> list[PathTrace]:
        """Trace all outgoing paths from a neuron."""
        return self.recognizer.trace(label)

    def why(self, label: str) -> list[PathTrace]:
        """Show all paths that lead TO a neuron (reverse lookup)."""
        neuron = self.neuron_repo.get_by_label(label.strip().lower())
        if neuron is None:
            return []

        paths = self.path_repo.get_paths_to(neuron.id)
        traces: list[PathTrace] = []
        for p in paths:
            steps = self.path_repo.get_steps(p.id)
            neurons = []
            # Walk the segments to reconstruct the neuron chain
            for step in steps:
                seg = self.segment_repo.get_by_id(step.segment_id)
                if seg is None:
                    continue
                if not neurons:
                    source = self.neuron_repo.get_by_id(seg.source_id)
                    if source:
                        neurons.append(source)
                target = self.neuron_repo.get_by_id(seg.target_id)
                if target:
                    neurons.append(target)
            traces.append(PathTrace(neurons=neurons, source_text=p.source_text))

        return traces

    def analyze_similarity(self) -> list[SimilarityLink]:
        """Scan for path similarities across all property neurons."""
        return self.similarity.analyze()

    def get_similar(self, label: str) -> list[SimilarityLink]:
        """Get neurons that share downstream paths with the given neuron."""
        return self.similarity.get_similar(label)

    def stats(self) -> dict:
        """Return brain statistics."""
        neurons = self.neuron_repo.count()
        segments = self.segment_repo.count()
        paths = self.path_repo.count()

        strongest = None
        all_segs = self.segment_repo.list_all()
        if all_segs:
            s = max(all_segs, key=lambda s: s.strength)
            src = self.neuron_repo.get_by_id(s.source_id)
            tgt = self.neuron_repo.get_by_id(s.target_id)
            if src and tgt:
                strongest = f"{src.label} → {tgt.label} (strength: {s.strength:.2f})"

        return {
            "neurons": neurons,
            "segments": segments,
            "paths": paths,
            "strongest_segment": strongest,
        }

    def _load_dynamic_associations(self) -> None:
        """Reload dynamic associations from DB into taxonomy."""
        for assoc, prop_label in self.association_repo.list_all():
            self.taxonomy.register_property(prop_label, assoc)

    def define_association(self, name: str, question_word: str | None = None):
        """Create an ASSOCIATION neuron with an optional question word."""
        name = name.strip().lower()
        neuron, _ = self.neuron_repo.get_or_create(name, NeuronType.ASSOCIATION)
        if question_word:
            question_word = question_word.strip().lower()
            self.association_repo.set_question_word(name, question_word)
        self.conn.commit()
        return neuron

    def describe_association(self, name: str, properties: list[str]) -> list[str]:
        """Register properties under an association, creating neurons and segments."""
        name = name.strip().lower()

        # Ensure the association neuron exists
        assoc_neuron = self.neuron_repo.get_by_label(name)
        if assoc_neuron is None:
            raise ValueError(f"Unknown association: {name}. Use 'define {name}' first.")

        registered = []
        for prop_label in properties:
            prop_label = prop_label.strip().lower()
            if not prop_label:
                continue

            # Get or create PROPERTY neuron
            prop_neuron, _ = self.neuron_repo.get_or_create(prop_label, NeuronType.PROPERTY)

            # Create segment: property → association (relation: "is_a")
            self.segment_repo.get_or_create(prop_neuron.id, assoc_neuron.id, "is_a")

            # Register in taxonomy
            self.taxonomy.register_property(prop_label, name)

            # Persist to associations table
            self.association_repo.create(name, prop_label, assoc_neuron.id)

            registered.append(prop_label)

        self.conn.commit()
        return registered

    def query_association(self, subject: str, association: str) -> list[str]:
        """Find properties of <subject> under <association>.

        E.g., query_association("apple", "taste") -> ["sweet"]
        """
        concept = self.neuron_repo.get_by_label(subject.strip().lower())
        if concept is None:
            return []

        # Gather all properties registered under this association
        assoc_properties = set(self.association_repo.get_properties(association))
        # Also check built-in taxonomy
        for label, ptype in self.taxonomy._properties.items():
            if ptype == association:
                assoc_properties.add(label)

        # Find paths ending at this concept, filter by matching properties
        paths = self.path_repo.get_paths_to(concept.id)
        results = []
        for p in paths:
            origin = self.neuron_repo.get_by_id(p.origin_id)
            if origin and origin.label in assoc_properties:
                results.append(origin.label)
        return sorted(set(results))

    def list_question_words(self) -> dict[str, list[str]]:
        """Return {question_word: [association_names]} for all registered question words."""
        result: dict[str, list[str]] = {}
        # Built-in defaults
        for assoc, qword in _BUILTIN_QUESTION_WORDS.items():
            result.setdefault(qword, []).append(assoc)
        # Dynamic (from DB) — overrides/extends
        for assoc, qword in self.association_repo.list_question_words():
            result.setdefault(qword, []).append(assoc)
        return result

    def resolve_question_word(self, word: str) -> list[str]:
        """Return association names for a question word. Checks DB then builtins."""
        associations = self.association_repo.get_by_question_word(word)
        if associations:
            return associations
        # Check builtins
        result = []
        for assoc, qword in _BUILTIN_QUESTION_WORDS.items():
            if qword == word:
                result.append(assoc)
        return result

    def categorize(self, label: str, category: str) -> None:
        """Tag a concept: categorize apple item"""
        label = label.strip().lower()
        category = category.strip().lower()
        self.taxonomy.register_category(label, category)
        self.category_repo.set_category(label, category)
        self.conn.commit()

    def get_category(self, label: str) -> str:
        """Returns category for a concept, or 'thing' as default."""
        return self.taxonomy.subject_category(label)

    def list_categories(self) -> dict[str, list[str]]:
        """Return {category: [labels]} combining taxonomy and DB."""
        result: dict[str, list[str]] = {}
        # From taxonomy (builtins + loaded)
        for label, cat in self.taxonomy._categories.items():
            result.setdefault(cat, []).append(label)
        return {k: sorted(v) for k, v in sorted(result.items())}

    def _load_categories(self) -> None:
        """Reload categories from DB into taxonomy."""
        for cat, labels in self.category_repo.list_categories().items():
            for label in labels:
                self.taxonomy.register_category(label, cat)

    def list_associations(self) -> dict[str, list[str]]:
        """Return dict of {association: [properties]}."""
        result: dict[str, list[str]] = {}
        for assoc, prop_label in self.association_repo.list_all():
            result.setdefault(assoc, []).append(prop_label)
        return result

    def _make_provider(self):
        """Build LLM provider from settings."""
        from ..nlp.provider import get_provider
        provider_name = self.settings_repo.get("llm_provider") or "anthropic"
        return get_provider(provider_name)

    def _make_observer(self):
        """Build a VisionObserver from current settings."""
        from ..nlp.vision import VisionObserver
        from ..nlp.provider import DEFAULT_URLS

        provider = self._make_provider()
        url = self.settings_repo.get("llm_api_url") or DEFAULT_URLS.get(provider.name, "")
        key = self.settings_repo.get("llm_api_key") or ""
        model = self.settings_repo.get("llm_model") or ""
        return VisionObserver(url, key, model, provider=provider)

    def _make_reader(self):
        """Build a DocumentReader from current settings."""
        from ..nlp.reader import DocumentReader
        from ..nlp.provider import DEFAULT_URLS

        provider = self._make_provider()
        url = self.settings_repo.get("llm_api_url") or DEFAULT_URLS.get(provider.name, "")
        key = self.settings_repo.get("llm_api_key") or ""
        model = self.settings_repo.get("llm_model") or ""
        return DocumentReader(url, key, model, provider=provider)

    def perceive(self, image_path: str, label: str | None = None,
                 max_rounds: int = 3, callback=None, *,
                 user_initiated: bool = True):
        """Run the perception loop on an image.

        Requires LLM configured (same as 'ask'). Uses LLM Vision
        as Sara's senses: observe, recognize, inquire, verify.

        Returns a PerceptionResult.
        """
        gate = self._ethics.check_action("perceive", user_initiated=user_initiated)
        if not gate.allowed:
            raise PermissionError(gate.reason)
        from .perceiver import Perceiver

        model = self.settings_repo.get("llm_model")
        provider = self._make_provider()
        if provider.needs_api_key() and not self.settings_repo.get("llm_api_key"):
            raise ValueError("No LLM configured. Use: llm set <api_key> [model]")
        if not model:
            raise ValueError("No LLM configured. Use: llm set <api_key> [model]")

        observer = self._make_observer()
        perceiver = Perceiver(self, observer)
        result = perceiver.perceive(image_path, label=label,
                                    max_rounds=max_rounds, callback=callback)
        self.conn.commit()
        return result

    def correct(self, correct_label: str, *, from_tribe: bool = True):
        """Correct the last perception: the guess was wrong, this is actually <correct_label>.

        Returns correction details dict, or raises ValueError if no perception to correct.
        """
        gate = self._ethics.check_correction(from_tribe=from_tribe)
        if not gate.allowed:
            raise PermissionError(gate.reason)
        from .perceiver import Perceiver

        if self._last_perception is None:
            raise ValueError("No recent perception to correct.")

        observer = self._make_observer()
        perceiver = Perceiver(self, observer)
        result = perceiver.correct(correct_label, self._last_perception)
        self.conn.commit()
        return result

    def see(self, property_label: str):
        """Parent points out a property Sara missed on the last perceived image.

        Returns teaching details dict, or raises ValueError if no perception.
        """
        from .perceiver import Perceiver

        if self._last_perception is None:
            raise ValueError("No recent perception to add observations to.")

        observer = self._make_observer()
        perceiver = Perceiver(self, observer)
        result = perceiver.add_observation(property_label, self._last_perception)
        self.conn.commit()
        return result

    def ingest(self, text: str, source: str = "text", callback=None, *,
               user_initiated: bool = True):
        """Ingest a document through the LLM cortex.

        The LLM reads the document, extracts facts, Sara learns them.
        Then Sara reports what she understood and asks about unknowns.

        Returns a DigestionResult.
        """
        gate = self._ethics.check_action("ingest", user_initiated=user_initiated)
        if not gate.allowed:
            raise PermissionError(gate.reason)
        from .digester import Digester

        model = self.settings_repo.get("llm_model")
        provider = self._make_provider()
        if provider.needs_api_key() and not self.settings_repo.get("llm_api_key"):
            raise ValueError("No LLM configured. Use: llm set <provider> <model>")
        if not model:
            raise ValueError("No LLM configured. Use: llm set <provider> <model>")

        reader = self._make_reader()
        digester = Digester(self, reader)
        result = digester.ingest(text, source=source, callback=callback)
        self.conn.commit()
        return result

    def close(self) -> None:
        gate = self._ethics.check_shutdown()
        # Always allowed — shutdown is sleep, not death
        self.db.close()

    def __enter__(self) -> Brain:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
