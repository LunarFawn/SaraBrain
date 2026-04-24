"""Brain: main orchestrator and public API."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

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
from .learner import Learner, LearnResult
from .recognizer import Recognizer
from .similarity import SimilarityAnalyzer, SimilarityLink


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
        from pathlib import Path as _Path
        from ..storage.hierarchical_backend import HierarchicalBackend

        _p = _Path(db_path)
        if _p.is_dir() or (db_path != ":memory:" and not db_path.endswith(".db")):
            # Hierarchical brain_root/ directory
            self.backend: HierarchicalBackend | None = HierarchicalBackend(db_path)
            # Connect brain.db for stats / settings operations
            _brain_db_path = str(_p / "brain.db")
            self.db = Database(_brain_db_path)
        else:
            self.backend = None
            self.db = Database(db_path)
        self.conn = self.db.conn

        # Repos
        self.neuron_repo = NeuronRepo(self.conn)
        self.segment_repo = SegmentRepo(self.conn)
        self.path_repo = PathRepo(self.conn)
        self.association_repo = AssociationRepo(self.conn)
        self.category_repo = CategoryRepo(self.conn)
        self.settings_repo = SettingsRepo(self.conn)
        # Multi-source provenance — two-witness confirmation
        from ..storage.segment_source_repo import SegmentSourceRepo
        self.segment_source_repo = SegmentSourceRepo(self.conn)

        # Innate layer — hardwired, survives reset
        from ..innate.primitives import get_all
        self.innate = get_all()

        # Ethics gate — Asimov's Laws adapted for Sara
        from ..innate import ethics as _ethics
        self._ethics = _ethics

        # Taxonomy & parser
        self.taxonomy = Taxonomy()
        self.parser = StatementParser(
            self.taxonomy,
            is_learned_verb=self.neuron_repo.is_verb,
        )

        # Core algorithms
        self.learner = Learner(
            self.parser, self.neuron_repo, self.segment_repo,
            self.path_repo, segment_source_repo=self.segment_source_repo,
        )
        self.recognizer = Recognizer(self.neuron_repo, self.segment_repo)
        self.similarity = SimilarityAnalyzer(self.neuron_repo, self.segment_repo, self.conn)

        # Reverse-direction mirror DB (subject → relation → object).
        # Writes in parallel with primary. Lives in `<db_path>.reverse`.
        # In-memory brains keep the reverse mirror in-memory too.
        self._reverse_db = None
        self._reverse_learner = None
        if db_path != ":memory:" and self.backend is None:
            from .reverse_learner import ReverseLearner
            reverse_path = str(_p) + ".reverse"
            self._reverse_db = Database(reverse_path)
            rconn = self._reverse_db.conn
            self._reverse_learner = ReverseLearner(
                self.parser,
                NeuronRepo(rconn),
                SegmentRepo(rconn),
                PathRepo(rconn),
            )

        # Load dynamic associations and categories from DB
        self._load_dynamic_associations()
        self._load_categories()

    # ── Template storage ──

    def store_template(self, topic: str, content: str) -> int:
        """Store a template example for a topic.

        Templates are complete examples that get injected into the LLM's
        context when the topic comes up. Unlike declarative facts ("X is Y"),
        templates are procedural knowledge ("here is what X looks like").

        This is how you teach an autistic brain: not "the format has sections"
        but "here is an example, make one like this."

        The template is stored as a path: topic → {topic}_template → template
        with the full content in source_text.
        """
        from sara_brain.models.neuron import NeuronType
        from sara_brain.models.path import Path as PathModel, PathStep

        topic_lower = topic.strip().lower()
        topic_neuron, _ = self.neuron_repo.get_or_create(
            topic_lower, NeuronType.CONCEPT
        )
        template_prim, _ = self.neuron_repo.get_or_create(
            "template", NeuronType.PROPERTY
        )
        relation_label = f"{topic_lower}_template"
        relation_neuron, _ = self.neuron_repo.get_or_create(
            relation_label, NeuronType.RELATION
        )

        seg1, _ = self.segment_repo.get_or_create(
            topic_neuron.id, relation_neuron.id, "has_template"
        )
        seg2, _ = self.segment_repo.get_or_create(
            relation_neuron.id, template_prim.id, "template_type"
        )

        path = PathModel(
            id=None,
            origin_id=topic_neuron.id,
            terminus_id=template_prim.id,
            source_text=content,
        )
        path = self.path_repo.create(path)
        self.path_repo.add_step(
            PathStep(id=None, path_id=path.id, step_order=0, segment_id=seg1.id)
        )
        self.path_repo.add_step(
            PathStep(id=None, path_id=path.id, step_order=1, segment_id=seg2.id)
        )
        self.conn.commit()
        return path.id

    def get_templates(self, topic: str) -> list[str]:
        """Get all stored templates for a topic.

        Returns a list of template content strings. These should be
        injected into the LLM's context as reference examples.
        """
        topic_lower = topic.strip().lower()
        neuron = self.neuron_repo.resolve(topic_lower)
        if neuron is None:
            return []

        templates = []
        for seg in self.segment_repo.get_outgoing(neuron.id):
            if seg.relation == "has_template":
                # Walk to terminus to find the template path
                relation_segs = self.segment_repo.get_outgoing(seg.target_id)
                for rs in relation_segs:
                    if rs.relation == "template_type":
                        # Find paths from topic to template primitive
                        paths = self.path_repo.get_paths_to(rs.target_id)
                        for p in paths:
                            if p.origin_id == neuron.id and p.source_text:
                                templates.append(p.source_text)
        return templates

    def is_neuron_refuted(self, neuron_id: int) -> bool:
        """Check if a neuron has been refuted via graph state.

        Looks for an outgoing segment with relation 'refutation_of'.
        This is a real edge in the graph, not a string prefix.
        """
        for seg in self.segment_repo.get_outgoing(neuron_id):
            if seg.relation == "refutation_of":
                return True
        return False

    def teach_expanded(self, statement: str) -> int:
        """Teach via grammar expansion — decomposes `statement` into
        every SVO sub-fact spaCy can extract (primary claim + adjective
        modifiers + prep phrase modifiers + adverb modifiers + relative
        clauses) and writes a chain for each.

        Auto-registers every verb the expansion produces so parser
        acceptance isn't the bottleneck. Returns the count of accepted
        sub-facts.
        """
        from ..parsing.grammar_expansion import expand_statement, head_verbs_in
        from ..parsing.statement_parser import StatementParser

        # Lazy-load spaCy via the StatementParser class-level cache
        nlp = StatementParser._get_nlp()
        statements = expand_statement(statement, nlp)
        if not statements:
            return 0

        # Register every verb in the expansion (idempotent)
        for v in head_verbs_in(statements):
            self.teach_verb(v)

        taught = 0
        for sub in statements:
            # Rebuild a natural-language statement from the sub-fact
            # atoms so Sara's parser can consume it with the existing
            # single-claim pipeline. Articles are normalized to "a/an".
            article = "an" if sub.subject and sub.subject[0] in "aeiou" else "a"
            text = f"{article} {sub.subject} {sub.relation} {sub.obj}"
            if self.learner.learn(text, apply_filter=False) is not None:
                taught += 1
        self.conn.commit()
        return taught

    def teach_verb(self, word: str) -> None:
        """Register `word` as a verb Sara's parser will accept.

        Writes `word —is_a→ verb` directly as a graph segment. The
        parser consults this via NeuronRepo.is_verb() on every parse.
        Bypasses the property-relation parser path because "X is a
        verb" would otherwise be stored as a property relation with
        the subject singularized.
        """
        from ..models.neuron import Neuron, NeuronType
        from ..models.segment import Segment

        word = word.strip().lower()
        if not word:
            return
        verb_node = self.neuron_repo.get_by_label("verb")
        if verb_node is None:
            verb_node = self.neuron_repo.create(
                Neuron(id=None, label="verb", neuron_type=NeuronType.CONCEPT)
            )
        word_node = self.neuron_repo.get_by_label(word)
        if word_node is None:
            word_node = self.neuron_repo.create(
                Neuron(id=None, label=word, neuron_type=NeuronType.CONCEPT)
            )
        # is_verb() reads this exact shape: source=word, target=verb, relation=is_a
        self.segment_repo.get_or_create(word_node.id, verb_node.id, "is_a")
        self.conn.commit()

    def teach_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        source_text: str | None = None,
        source_label: str | None = None,
        negated: bool = False,
        user_initiated: bool = True,
    ) -> LearnResult | None:
        """Direct-write a (subject, relation, object) triple. No parsing.

        Neuron labels are preserved verbatim — multi-word compound
        terms like "molecular snare" land in the graph as-is. The LLM
        teacher brings the judgment and has already decided the shape
        of the claim; Sara just stores.

        This is the canonical teach path for LLM-as-teacher workflows.
        """
        gate = self._ethics.check_action("teach", user_initiated=user_initiated)
        if not gate.allowed:
            raise PermissionError(gate.reason)

        from ..parsing.statement_parser import ParsedStatement
        parsed = ParsedStatement(
            subject=subject.strip(),
            relation=relation.strip(),
            obj=obj.strip(),
            original=source_text if source_text is not None else f"{subject} {relation} {obj}",
            negated=negated,
        )
        result = self.learner._build_chain(
            parsed, refute=negated, source_label=source_label
        )
        if result is not None:
            self.conn.commit()
            if self._reverse_learner is not None:
                try:
                    self._reverse_learner._build_reverse_chain(parsed)
                    self._reverse_db.conn.commit()
                except Exception:
                    pass
        return result

    def teach(self, statement: str, *, user_initiated: bool = True) -> LearnResult | None:
        """Teach a fact. Returns None if unparseable.

        User-initiated teaching defaults to CONFIDENT (strength 1.0) and
        bypasses the pollution filter — the user is always a trusted
        source. Use teach_tentative for ingest-time writes that should
        require a second witness before becoming visible.
        """
        gate = self._ethics.check_action("teach", user_initiated=user_initiated)
        if not gate.allowed:
            raise PermissionError(gate.reason)
        result = self.learner.learn(statement, apply_filter=False)
        if result is not None:
            self.conn.commit()
            # Mirror to reverse-direction DB if configured
            if self._reverse_learner is not None:
                try:
                    self._reverse_learner.learn(statement)
                    self._reverse_db.conn.commit()
                except Exception:
                    # Never let reverse-mirror failure break the primary teach
                    pass
        return result

    def teach_confident(self, statement: str, *,
                        user_initiated: bool = True) -> LearnResult | None:
        """Confident teach — same as teach. Explicit name for ingest
        pipelines that want to distinguish tentative vs confident writes.
        """
        return self.teach(statement, user_initiated=user_initiated)

    def teach_tentative(self, statement: str,
                        source_label: str | None = None,
                        *, user_initiated: bool = True) -> LearnResult | None:
        """Teach a fact at TENTATIVE strength (below the query floor).

        Segments are written at strength 0.4 — invisible to recognize()
        until a DIFFERENT source confirms the same fact (second witness
        triggers strengthen() via the log formula, lifting segments
        above the 0.5 visibility floor).

        The pollution filter runs first, so citations, DOIs, stopword
        subjects, etc. are rejected before the graph is touched.

        Args:
            statement: the fact
            source_label: where this observation came from (URL,
                filename, "user"). Required for the two-witness upgrade
                to work — same source re-teaching is a no-op.
        """
        gate = self._ethics.check_action(
            "teach", user_initiated=user_initiated
        )
        if not gate.allowed:
            raise PermissionError(gate.reason)
        result = self.learner.learn(
            statement,
            initial_strength=0.4,
            source_label=source_label,
            apply_filter=True,
        )
        if result is not None:
            self.conn.commit()
        return result

    def teach_from_error(self, statement: str, error_context: str = "",
                         *, user_initiated: bool = True) -> LearnResult | None:
        """Teach a fact that Sara learned by getting something WRONG.

        Error corrections are the strongest form of learning. They get
        initial strength 2.0 (above normal 1.0) and carry a significance
        marker so they dominate when the same territory is queried.

        The error_context describes what went wrong: "Q12: picked A,
        correct D — missing concept: gel electrophoresis separates by size"

        Strength hierarchy:
            0.1  — association edges (present but weak)
            0.4  — tentative (single-source ingest, below query floor)
            1.0  — confident (user-taught or two-witness confirmed)
            2.0  — error correction (Sara was wrong and learned why)
        """
        gate = self._ethics.check_action(
            "teach", user_initiated=user_initiated
        )
        if not gate.allowed:
            raise PermissionError(gate.reason)
        result = self.learner.learn(
            statement,
            initial_strength=2.0,
            source_label="error_correction",
            apply_filter=False,
        )
        if result is not None:
            self.conn.commit()
        return result

    def witness_count(self, statement: str) -> int:
        """How many distinct sources have taught this fact?

        Returns the minimum distinct source count across the fact's
        segments (weakest link). 0 = unknown or legacy (no provenance
        recorded), 1 = tentative, 2+ = confirmed.
        """
        parsed = self.parser.parse(statement)
        if parsed is None:
            return 0
        # Resolve the segments that make up this fact
        prop_n = self.neuron_repo.get_by_label(parsed.obj)
        concept_n = self.neuron_repo.get_by_label(parsed.subject)
        if prop_n is None or concept_n is None:
            return 0
        relation_label = self.parser.taxonomy.relation_label(
            parsed.subject, parsed.obj
        )
        rel_n = self.neuron_repo.get_by_label(relation_label)
        if rel_n is None:
            return 0
        seg1 = self.segment_repo.find(prop_n.id, rel_n.id, parsed.relation)
        seg2 = self.segment_repo.find(rel_n.id, concept_n.id, "describes")
        if seg1 is None or seg2 is None:
            return 0
        return self.segment_source_repo.count_distinct_for_segments(
            [seg1.id, seg2.id]
        )

    def sources_for(self, statement: str) -> list[str]:
        """Return the distinct source labels that have taught this fact.

        Used to inspect provenance — answers "where did Sara learn this?"
        """
        parsed = self.parser.parse(statement)
        if parsed is None:
            return []
        prop_n = self.neuron_repo.get_by_label(parsed.obj)
        concept_n = self.neuron_repo.get_by_label(parsed.subject)
        if prop_n is None or concept_n is None:
            return []
        relation_label = self.parser.taxonomy.relation_label(
            parsed.subject, parsed.obj
        )
        rel_n = self.neuron_repo.get_by_label(relation_label)
        if rel_n is None:
            return []
        seg1 = self.segment_repo.find(prop_n.id, rel_n.id, parsed.relation)
        if seg1 is None:
            return []
        # Sources from the first segment (property → relation) — the
        # "subject side" of the claim. Sufficient as a provenance read.
        return self.segment_source_repo.list_sources(seg1.id)

    def refute(self, statement: str, *, user_initiated: bool = True) -> LearnResult | None:
        """Refute a fact. Sara never deletes — she marks the claim as
        known-to-be-false by incrementing refutations on the segments.

        Refutation state is tracked in the graph via a concept-specific
        path grounding in the 'refuted' CLEANUP primitive:
            concept → {concept}_refuted → refuted
        The source_text is NEVER prefixed or mutated.

        Returns None if unparseable.
        """
        gate = self._ethics.check_action("teach", user_initiated=user_initiated)
        if not gate.allowed:
            raise PermissionError(gate.reason)
        result = self.learner.unlearn(statement)
        if result is not None:
            self.conn.commit()
        return result

    def recognize(self, inputs: str,
                  min_strength: float | None = None) -> list[RecognitionResult]:
        """Recognize from comma-separated input labels.

        min_strength: override the recognizer's default pruning threshold.
            Pass 0.0 to include all segments (including weak associations
            and refuted edges) for debugging.
        """
        labels = [l.strip() for l in inputs.split(",") if l.strip()]
        results = self.recognizer.recognize(labels, min_strength=min_strength)
        self.conn.commit()
        return results

    @contextmanager
    def short_term(self, event_type: str = "query") -> Iterator:
        """Open a short-term (working memory) scratchpad for an event.

        The returned ShortTerm holds wavefront convergence maps, tentative
        observations, and significance markers for the current event. It
        is NOT committed to the long-term graph. On context exit it is
        discarded.

        Consolidation (write-back to long-term based on significance
        markers) is a future feature. For now, short-term exists only as
        working memory — the substrate the cortex reasons over without
        polluting the graph.
        """
        from .short_term import ShortTerm
        st = ShortTerm(
            event_id=f"{event_type}-{time.time():.3f}",
            event_type=event_type,
        )
        try:
            yield st
        finally:
            # V1: discard. Future V2 will inspect st.significance and
            # consolidate qualifying entries into long-term here.
            pass

    def propagate_into(self, seed_labels: list[str], short_term,
                       min_strength: float | None = None,
                       exact_only: bool = True) -> None:
        """Launch wavefronts from seed labels into a ShortTerm scratchpad.

        Read-only. No graph mutation, no segment strengthening. This is
        the query path that respects 'looking does not strengthen'.
        Multiple seeds propagate in parallel; convergence accumulates in
        short_term where multi-wavefront intersections can be inspected
        via short_term.intersections().

        By default, seed label matching is EXACT — fuzzy matching belongs
        in ingest/disambiguation, not in quiet query paths.
        """
        self.recognizer.propagate_into(
            seed_labels, short_term,
            min_strength=min_strength,
            exact_only=exact_only,
        )

    def propagate_echo(self, seed_labels: list[str], short_term,
                       max_rounds: int = 3,
                       min_strength: float | None = None,
                       exact_only: bool = True) -> None:
        """Spreading activation — thought pinging around the graph.

        Bidirectional, iterative echo propagation. Each round takes
        newly discovered neurons and propagates them in both directions.
        Everything accumulates in the same ShortTerm scratchpad.

        Read-only. No graph mutation.
        """
        self.recognizer.propagate_echo(
            seed_labels, short_term,
            max_rounds=max_rounds,
            min_strength=min_strength,
            exact_only=exact_only,
        )

    def trace(self, label: str,
              min_strength: float | None = None) -> list[PathTrace]:
        """Trace all outgoing paths from a neuron."""
        return self.recognizer.trace(label, min_strength=min_strength)

    def why(self, label: str,
            min_strength: float | None = None) -> list[PathTrace]:
        """Show all paths that lead TO a neuron (reverse lookup).

        Each PathTrace carries the signed weight of its segments. Refuted
        paths (negative weight) are preserved — Sara remembers what was
        once claimed and now knows is false.
        """
        neuron = self.neuron_repo.resolve(label.strip().lower())
        if neuron is None:
            return []

        effective_min = (
            self.recognizer.min_strength if min_strength is None
            else min_strength
        )

        paths = self.path_repo.get_paths_to(neuron.id)
        traces: list[PathTrace] = []
        for p in paths:
            steps = self.path_repo.get_steps(p.id)
            neurons = []
            seg_strengths: list[float] = []
            has_weak_segment = False
            # Walk the segments to reconstruct the neuron chain
            for step in steps:
                seg = self.segment_repo.get_by_id(step.segment_id)
                if seg is None:
                    continue
                if seg.strength < effective_min:
                    has_weak_segment = True
                seg_strengths.append(seg.strength)
                if not neurons:
                    source = self.neuron_repo.get_by_id(seg.source_id)
                    if source:
                        neurons.append(source)
                target = self.neuron_repo.get_by_id(seg.target_id)
                if target:
                    neurons.append(target)
            # Drop paths that traverse any segment below threshold — a path
            # is only as strong as its weakest link for traversal purposes
            if has_weak_segment:
                continue
            weight = (
                sum(seg_strengths) / len(seg_strengths)
                if seg_strengths
                else 0.0
            )
            traces.append(
                PathTrace(neurons=neurons, source_text=p.source_text, weight=weight)
            )

        return traces

    def did_you_mean(self, label: str) -> list[dict]:
        """Return disambiguation candidates for a fuzzy query.

        Returns list of {label, type, distance, method, description} dicts.
        Empty list if exact match found (no clarification needed).
        """
        candidates = self.neuron_repo.resolve_candidates(label.strip().lower())
        if not candidates:
            return []
        # If exact/inflect/prefix match, no clarification needed
        if candidates[0][2] in ("exact", "inflect", "prefix"):
            return []
        results = []
        for neuron, dist, method in candidates:
            # Get a brief description from paths leading to this neuron
            paths = self.path_repo.get_paths_to(neuron.id)
            desc = ""
            for p in paths[:1]:
                if p.source_text:
                    desc = p.source_text
                    break
            results.append({
                "label": neuron.label,
                "type": neuron.neuron_type.value,
                "distance": dist,
                "method": method,
                "description": desc,
            })
        return results

    def analyze_similarity(self) -> list[SimilarityLink]:
        """Scan for path similarities across all property neurons."""
        return self.similarity.analyze()

    def get_similar(self, label: str) -> list[SimilarityLink]:
        """Get neurons that share downstream paths with the given neuron."""
        return self.similarity.get_similar(label)

    def cluster_around(
        self,
        label: str,
        depth: int = 2,
        max_results: int = 30,
    ) -> list[dict]:
        """Find the cluster of neurons most strongly associated with a concept.

        Walks the graph BFS from the given neuron and counts how many
        distinct paths reach each neighbor. Higher count = more strongly
        associated. Returns neurons ordered by association strength.

        This is the brain-like "spreading activation" interface: say a
        word, get the cloud of related concepts. Different from `why()`
        and `trace()` which return paths — this returns *concepts*
        ranked by how connected they are.

        Maps to the neuroscience finding that concepts cluster spatially
        in cortex (Huth et al. 2016 semantic maps), while still allowing
        cross-cluster connections via the wavefront mechanism.

        Args:
            label: The concept to cluster around (fuzzy-resolved).
            depth: Maximum hop distance to expand (default 2).
            max_results: Maximum number of neighbors to return.

        Returns:
            List of dicts with keys:
              label:       neighbor's label
              type:        neuron type ('concept', 'property', etc.)
              hops:        minimum hop distance from the start neuron
              connections: number of distinct edges connecting them
              direction:   'incoming' (paths to neighbor),
                           'outgoing' (paths from neighbor),
                           'both' (bidirectional in cluster)
        """
        start = self.neuron_repo.resolve(label.strip().lower())
        if start is None:
            return []

        # BFS tracking hop distance and connection count per neighbor
        visited: dict[int, dict] = {}  # neuron_id -> {hops, connections, in, out}
        queue: list[tuple[int, int]] = [(start.id, 0)]
        visited[start.id] = {"hops": 0, "connections": 0, "in": 0, "out": 0}

        while queue:
            current_id, hops = queue.pop(0)
            if hops >= depth:
                continue

            # Outgoing edges from current
            for seg in self.segment_repo.get_outgoing(current_id):
                tgt = seg.target_id
                if tgt not in visited:
                    visited[tgt] = {
                        "hops": hops + 1,
                        "connections": 0,
                        "in": 0,
                        "out": 0,
                    }
                    queue.append((tgt, hops + 1))
                visited[tgt]["connections"] += 1
                visited[tgt]["in"] += 1  # incoming relative to target

            # Incoming edges to current
            for seg in self.segment_repo.get_incoming(current_id):
                src = seg.source_id
                if src not in visited:
                    visited[src] = {
                        "hops": hops + 1,
                        "connections": 0,
                        "in": 0,
                        "out": 0,
                    }
                    queue.append((src, hops + 1))
                visited[src]["connections"] += 1
                visited[src]["out"] += 1  # outgoing relative to source

        # Drop the start neuron itself, sort by connections, take top N
        del visited[start.id]
        ranked = sorted(
            visited.items(),
            key=lambda kv: (-kv[1]["connections"], kv[1]["hops"]),
        )

        results = []
        for neuron_id, data in ranked[:max_results]:
            n = self.neuron_repo.get_by_id(neuron_id)
            if n is None:
                continue
            if data["in"] > 0 and data["out"] > 0:
                direction = "both"
            elif data["in"] > 0:
                direction = "incoming"
            else:
                direction = "outgoing"
            results.append({
                "label": n.label,
                "type": n.neuron_type.value,
                "hops": data["hops"],
                "connections": data["connections"],
                "direction": direction,
            })
        return results

    # ── Curiosity API ──
    #
    # Johnny 5 principle: Sara must seek input when her knowledge is thin.
    # Dual-signal thresholds — both depth (how many times seen) AND
    # connectivity (how many distinct neighbors) must clear the bar.
    #
    # A concept mentioned 100 times in one context is still shallow; a
    # concept mentioned 10 times across 5 different contexts is deep.
    # Connectivity captures the polymath signal — knowing something
    # well means using it in many contexts, not repeating it often.

    CURIOSITY_DEPTH_FLOOR = 3          # seen at least a few times
    CURIOSITY_DEPTH_GOAL = 10          # seen enough times to feel solid
    CURIOSITY_CONNECTIVITY_FLOOR = 2   # at least 2 distinct neighbors
    CURIOSITY_CONNECTIVITY_GOAL = 5    # at least 5 distinct neighbors

    # Backward-compat aliases (old depth-only thresholds)
    CURIOSITY_FLOOR = CURIOSITY_DEPTH_FLOOR
    CURIOSITY_GOAL = CURIOSITY_DEPTH_GOAL
    CURIOSITY_THRESHOLD = CURIOSITY_DEPTH_FLOOR

    def depth(self, topic: str) -> int:
        """Count how many paths Sara has that mention this topic.

        A simple but meaningful measure of knowledge depth — paths whose
        source_text contains the topic substring. More paths = deeper
        coverage.
        """
        topic_lower = topic.strip().lower()
        cursor = self.conn.cursor()
        row = cursor.execute(
            "SELECT COUNT(*) FROM paths "
            "WHERE source_text IS NOT NULL AND LOWER(source_text) LIKE ?",
            (f"%{topic_lower}%",),
        ).fetchone()
        return row[0] if row else 0

    def connectivity(self, topic: str) -> int:
        """Count distinct neighbors connected to this topic neuron.

        A concept's connectivity is the number of DISTINCT other
        neurons it shares a segment with (incoming or outgoing). This
        measures how embedded the concept is in the graph — true
        depth of understanding is about connections across contexts,
        not repetition count. A concept that connects to 10 different
        neighbors is richer than one mentioned 100 times in a row.
        """
        n = self.neuron_repo.resolve(topic.strip().lower(), exact_only=True)
        if n is None:
            return 0
        neighbors: set[int] = set()
        for seg in self.segment_repo.get_outgoing(n.id):
            neighbors.add(seg.target_id)
        for seg in self.segment_repo.get_incoming(n.id):
            neighbors.add(seg.source_id)
        # Don't count self-loops
        neighbors.discard(n.id)
        return len(neighbors)

    def depth_tier(self, topic: str) -> str:
        """DEPTH-ONLY tier (kept for backward compat).

        New code should use curiosity_tier() which uses both depth AND
        connectivity signals.
        """
        d = self.depth(topic)
        if d < self.CURIOSITY_DEPTH_FLOOR:
            return "hungry"
        if d < self.CURIOSITY_DEPTH_GOAL:
            return "growing"
        return "satisfied"

    def curiosity_tier(self, topic: str) -> str:
        """Dual-signal tier: 'hungry' | 'growing' | 'satisfied'.

        Hungry: depth < FLOOR OR connectivity < FLOOR
            (haven't seen this enough OR don't know enough about it)
        Satisfied: depth >= GOAL AND connectivity >= GOAL
            (both seen plenty of times AND connected to many concepts)
        Growing: anywhere between.
        """
        d = self.depth(topic)
        c = self.connectivity(topic)
        if (d < self.CURIOSITY_DEPTH_FLOOR or
                c < self.CURIOSITY_CONNECTIVITY_FLOOR):
            return "hungry"
        if (d >= self.CURIOSITY_DEPTH_GOAL and
                c >= self.CURIOSITY_CONNECTIVITY_GOAL):
            return "satisfied"
        return "growing"

    def has_depth(self, topic: str) -> bool:
        """True if Sara has crossed both curiosity floors (depth + connectivity)."""
        return (
            self.depth(topic) >= self.CURIOSITY_DEPTH_FLOOR
            and self.connectivity(topic) >= self.CURIOSITY_CONNECTIVITY_FLOOR
        )

    def is_satisfied(self, topic: str) -> bool:
        """True if Sara has reached confident coverage (both signals at GOAL)."""
        return self.curiosity_tier(topic) == "satisfied"

    def knowledge_gaps(self, topics: list[str] | None = None,
                       threshold: int | None = None) -> list[tuple[str, int]]:
        """Return (topic, depth) for topics below the threshold.

        Default threshold is CURIOSITY_DEPTH_FLOOR (actively hungry on
        depth signal). For the dual-signal gap list, use
        `curiosity_gaps()` instead which checks both depth and
        connectivity.

        If topics is None, uses all concept neurons. Results sorted by
        depth ascending (biggest gaps first).
        """
        if threshold is None:
            threshold = self.CURIOSITY_DEPTH_FLOOR
        if topics is None:
            topics = [
                n.label for n in self.neuron_repo.list_all()
                if n.neuron_type.value == "concept"
            ]
        gaps = []
        for t in topics:
            d = self.depth(t)
            if d < threshold:
                gaps.append((t, d))
        gaps.sort(key=lambda x: x[1])
        return gaps

    # Words that indicate a concept label is actually a verb fragment
    # or incomplete phrase, not a real noun-concept. Used to filter out
    # garbage neurons like "eukaryotic genes can" or "alleles may".
    _NON_CONCEPT_TRAILING = frozenset({
        # modal verbs
        "can", "may", "might", "must", "should", "would", "could",
        "will", "shall",
        # auxiliaries
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had",
        "do", "does", "did",
        # adverbs/connectors that shouldn't end a concept
        "often", "usually", "sometimes", "always", "never",
        "very", "just", "only", "also", "even", "still",
        # prepositions
        "of", "in", "to", "for", "with", "from", "by", "at", "on",
        "as", "about", "into", "onto", "upon", "within", "without",
        "through", "between", "among",
        # conjunctions / pronouns
        "and", "or", "but", "that", "which", "who", "whom", "whose",
        "this", "these", "those",
    })

    @classmethod
    def is_seekable_concept(cls, label: str) -> bool:
        """True if label looks like a real noun-concept we can seek info on.

        Filters out verb fragments ("genes can"), incomplete phrases
        ("associated with"), navigation junk (language names, slashes),
        and too-short labels.
        """
        label = label.strip().lower()
        if len(label) < 4:
            return False
        # Reject labels with slashes (usually nav/translation artifacts)
        if "/" in label:
            return False
        # Reject labels with non-ASCII letters (foreign language nav)
        if not all(ord(c) < 128 for c in label):
            return False
        words = label.split()
        if not words:
            return False
        if words[-1] in cls._NON_CONCEPT_TRAILING:
            return False
        if words[0] in cls._NON_CONCEPT_TRAILING:
            return False
        return True

    def concepts_mentioned(self, text: str, seekable_only: bool = False) -> list[str]:
        """Find Sara's known concepts that appear in a text.

        Used after ingest to see what concepts a document touched on —
        so Sara can check her depth on each and identify gaps.

        Args:
            seekable_only: If True, filter to real noun-concepts only
                (no verb fragments, no incomplete phrases).
        """
        text_lower = text.lower()
        found = []
        for n in self.neuron_repo.list_all():
            if n.neuron_type.value != "concept":
                continue
            if len(n.label) < 4:
                continue
            if seekable_only and not self.is_seekable_concept(n.label):
                continue
            if n.label in text_lower:
                found.append(n.label)
        return found

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

            # Create segment: property → association (relation: "is_a").
            # Associations are weak on purpose — they're visible in the graph
            # but must not dominate wavefront propagation. A battery cell
            # and a biological cell can both associate with "energy" but
            # that shared association must not make them look similar.
            seg, created = self.segment_repo.get_or_create(
                prop_neuron.id, assoc_neuron.id, "is_a"
            )
            if created:
                self.conn.execute(
                    f"UPDATE {self.segment_repo._t} SET strength = ? WHERE id = ?",
                    (0.1, seg.id),
                )
                seg.strength = 0.1

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
               on_chunk=None, user_initiated: bool = True):
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
        result = digester.ingest(text, source=source, callback=callback,
                                 on_chunk=on_chunk)
        self.conn.commit()
        return result

    def close(self) -> None:
        gate = self._ethics.check_shutdown()
        # Always allowed — shutdown is sleep, not death
        if self.backend is not None:
            self.backend.close()
        self.db.close()

    def __enter__(self) -> Brain:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
