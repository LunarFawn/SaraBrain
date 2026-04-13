"""Agent bridge — Brain interface for the agent loop.

Two roles:
1. Query tools: LLM reads Sara's knowledge (read-only)
2. Observational learning: agent loop feeds outcomes to Sara

Follows the QBridge pattern (nlp/q_bridge.py) — every method returns str.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.brain import Brain
from ..cortex.cleanup import STOPWORD_SUBJECTS


class AgentBridge:
    """Brain interface for the agent. Read + observe."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    # ── Query tools (LLM reads Sara's knowledge) ──

    def query(self, topic: str | list) -> str:
        """What does Sara know about a topic? Uses why + trace."""
        if isinstance(topic, list):
            topic = " ".join(str(t) for t in topic)
        label = topic.strip().lower()
        traces = self.brain.why(label)
        forward = self.brain.trace(label)

        if not traces and not forward:
            return f"Sara doesn't know about '{topic}'."

        lines = []
        if traces:
            lines.append(f"Paths leading to '{label}':")
            for t in traces:
                src = f' (from: "{t.source_text}")' if t.source_text else ""
                lines.append(f"  {t}{src}")
        if forward:
            lines.append(f"Paths from '{label}':")
            for t in forward:
                src = f' (from: "{t.source_text}")' if t.source_text else ""
                lines.append(f"  {t}{src}")
        return "\n".join(lines)

    def recognize(self, inputs: str | list) -> str:
        """Given properties, what does Sara recognize?"""
        if isinstance(inputs, list):
            inputs = ", ".join(str(i) for i in inputs)
        results = self.brain.recognize(inputs)
        if not results:
            return "Sara doesn't recognize anything from those inputs."
        lines = []
        for r in results:
            lines.append(
                f"  {r.neuron.label} ({r.confidence} converging paths)"
            )
            for trace in r.converging_paths:
                lines.append(f"    path: {trace}")
        return "\n".join(lines)

    def context(self, keywords: str | list) -> str:
        """Search brain for knowledge relevant to keywords.

        Uses both exact and prefix matching so "planetary" finds "planet".
        """
        if isinstance(keywords, list):
            keywords = " ".join(str(k) for k in keywords)
        words = [
            w.strip().lower()
            for w in keywords.split()
            if len(w.strip()) > 2 and w.strip().lower() not in STOPWORD_SUBJECTS
        ]
        all_traces = []
        checked: set[int] = set()

        # Get all neurons once for prefix matching
        all_neurons = self.brain.neuron_repo.list_all()

        for kw in words:
            # Exact match first
            neuron = self.brain.neuron_repo.get_by_label(kw)
            if neuron and neuron.id not in checked:
                checked.add(neuron.id)
                for t in self.brain.why(kw):
                    all_traces.append((kw, t))
                for t in self.brain.trace(kw):
                    all_traces.append((kw, t))

            # Prefix/substring match — find neurons whose labels start with
            # or contain the keyword (e.g., "planetary" matches "planet")
            for n in all_neurons:
                if n.id in checked:
                    continue
                label = n.label
                # Match if: keyword starts with label, or label starts with keyword
                if (label.startswith(kw) or kw.startswith(label)) and len(label) > 2:
                    checked.add(n.id)
                    for t in self.brain.why(label):
                        all_traces.append((label, t))
                    for t in self.brain.trace(label):
                        all_traces.append((label, t))

        if not all_traces:
            return f"Sara has no knowledge about: {keywords}"

        # Strongest paths first so the most relevant facts survive truncation
        all_traces.sort(key=lambda lt: lt[1].weight, reverse=True)

        lines = [f"Sara knows {len(all_traces)} relevant fact(s):"]
        for label, t in all_traces:
            src = f' (from: "{t.source_text}")' if t.source_text else ""
            lines.append(f"  [{label}] {t}{src}")
        return "\n".join(lines)

    def summarize(self, topic: str | list) -> str:
        """Aggregate everything Sara knows about a topic."""
        if isinstance(topic, list):
            topic = " ".join(str(t) for t in topic)
        label = topic.strip().lower()
        traces = self.brain.why(label)
        similar = self.brain.get_similar(label)

        lines = []
        if traces:
            lines.append(f"Paths to '{label}':")
            for t in traces:
                lines.append(f"  {t}")
        if similar:
            lines.append(f"Similar to '{label}':")
            for s in similar:
                lines.append(
                    f"  {s.neuron_a_label} <-> {s.neuron_b_label}"
                    f" (overlap: {s.overlap_ratio:.0%})"
                )
        if not lines:
            return f"Sara knows nothing about '{topic}'."
        return "\n".join(lines)

    def stats(self) -> str:
        """Brain statistics."""
        s = self.brain.stats()
        return (
            f"Neurons: {s['neurons']}, Segments: {s['segments']}, "
            f"Paths: {s['paths']}"
        )

    def brain_summary(self, max_items: int = 20) -> str:
        """Compact summary of Sara's knowledge for system prompt injection."""
        neurons = self.brain.neuron_repo.list_all()
        if not neurons:
            return "Sara's brain is empty — no knowledge yet."

        concepts = [n for n in neurons if n.neuron_type.value == "concept"]
        properties = [n for n in neurons if n.neuron_type.value == "property"]

        lines = [f"Sara knows {len(neurons)} neurons ({len(concepts)} concepts, {len(properties)} properties)."]
        if concepts:
            labels = [n.label for n in concepts[:max_items]]
            lines.append(f"Concepts: {', '.join(labels)}")
            if len(concepts) > max_items:
                lines.append(f"  ... and {len(concepts) - max_items} more")
        return "\n".join(lines)

    # ── Observational learning (agent loop feeds outcomes) ──

    def observe(self, fact: str) -> str | None:
        """Sara observes what happened and learns.

        Called by the agent loop after actions execute — not by the LLM directly.
        Returns the path label if learned, None if unparseable.
        """
        result = self.brain.teach(fact)
        if result is not None:
            self.brain.conn.commit()
            return result.path_label
        return None

    def observe_many(self, facts: list[str]) -> int:
        """Observe multiple facts. Returns count of successfully learned."""
        count = 0
        for fact in facts:
            result = self.brain.teach(fact)
            if result is not None:
                count += 1
        if count > 0:
            self.brain.conn.commit()
        return count

    def teach(self, statement: str) -> str:
        """Teach Sara a fact. Returns a string suitable for the LLM to read back.

        Unlike `observe()` which returns None on parse failure, this returns
        an explanatory string so the LLM knows what happened.
        """
        result = self.brain.teach(statement)
        if result is None:
            return (
                f"Could not parse: '{statement}'. "
                f"Try 'X is Y' or 'X are Y' or 'X contains/requires/includes Y'."
            )
        return f"Learned: {result.path_label} (path #{result.path_id})"

    def refute(self, statement: str) -> str:
        """Refute a fact. Sara marks it as known-to-be-false but never deletes it."""
        result = self.brain.refute(statement)
        if result is None:
            return (
                f"Could not parse: '{statement}'. "
                f"Try 'X is Y' format."
            )
        return (
            f"Refuted: {result.path_label} (path #{result.path_id}, "
            f"marked as known-to-be-false; the path is preserved as evidence "
            f"of what was once claimed)"
        )

    # ── Brain cleanup (LLM-callable) ──

    def scan_pollution(self) -> str:
        """Scan the brain for likely pollution. READ-ONLY.

        Returns a summary of article-typo neurons, pronoun-subject neurons,
        and suspected content-word typos. Does NOT modify anything.
        """
        from ..cortex.cleanup import (
            find_article_typo_neurons,
            find_pronoun_neurons,
            find_suspected_typo_neurons,
        )
        articles = find_article_typo_neurons(self.brain)
        pronouns = find_pronoun_neurons(self.brain)
        typos = find_suspected_typo_neurons(self.brain)

        lines = ["Brain pollution scan:"]
        lines.append(f"  Article-typo neurons: {len(articles)} (always safe to clean)")
        for c in articles[:10]:
            lines.append(f"    [{c.path_count} paths] {c.label!r}")
        if len(articles) > 10:
            lines.append(f"    ... and {len(articles) - 10} more")

        lines.append(f"  Pronoun-subject neurons: {len(pronouns)} (always safe to clean)")
        for c in pronouns[:10]:
            lines.append(f"    [{c.path_count} paths] {c.label!r}")
        if len(pronouns) > 10:
            lines.append(f"    ... and {len(pronouns) - 10} more")

        lines.append(f"  Suspected content-word typos: {len(typos)} (REQUIRES USER REVIEW)")
        for c in typos[:10]:
            lines.append(
                f"    {c.label!r} (~{c.canonical!r}, edit dist {c.edit_distance})"
            )
        if len(typos) > 10:
            lines.append(f"    ... and {len(typos) - 10} more")

        if not articles and not pronouns and not typos:
            lines.append("  No pollution found. Brain is clean.")

        return "\n".join(lines)

    def list_article_candidates(self) -> str:
        """List paths attached to neurons whose label looks like an article typo.

        READ-ONLY. Sara cannot decide to clean these. The LLM presents
        them to the user, who approves or rejects each one explicitly
        via brain_refute. A user typing "tteh" in their dialect may
        have meant exactly that — Sara has no authority to silently
        erase their language.
        """
        from ..cortex.cleanup import find_article_typo_neurons
        candidates = find_article_typo_neurons(self.brain)
        if not candidates:
            return "No article-typo candidates found."
        lines = [
            f"Found {len(candidates)} article-typo CANDIDATES (USER REVIEW REQUIRED):",
            "Sara WILL NOT auto-clean these. Each one needs your approval.",
            "What looks like 'teh' in English may be a real word in another dialect.",
            "",
        ]
        for c in candidates[:50]:
            lines.append(
                f"  {c.label!r} ({c.path_count} paths attached)"
            )
        if len(candidates) > 50:
            lines.append(f"  ... and {len(candidates) - 50} more")
        lines.append("")
        lines.append(
            "To refute a specific one, the user must explicitly say "
            "'refute X is Y' for each path they want marked false."
        )
        return "\n".join(lines)

    def list_pronoun_candidates(self) -> str:
        """List paths attached to pronoun-subject neurons.

        READ-ONLY. Same principle as list_article_candidates — Sara
        presents candidates, the user decides per instance. Even pronouns
        that Sara assumes are bugs may be intentional in some contexts.
        """
        from ..cortex.cleanup import find_pronoun_neurons
        candidates = find_pronoun_neurons(self.brain)
        if not candidates:
            return "No pronoun-subject candidates found."
        lines = [
            f"Found {len(candidates)} pronoun-subject CANDIDATES (USER REVIEW REQUIRED):",
            "Sara WILL NOT auto-clean these. Each one needs your approval.",
            "",
        ]
        for c in candidates[:50]:
            lines.append(f"  {c.label!r} ({c.path_count} paths attached)")
        if len(candidates) > 50:
            lines.append(f"  ... and {len(candidates) - 50} more")
        return "\n".join(lines)

    def list_suspected_typos(self) -> str:
        """List suspected content-word typos for USER REVIEW.

        Sara cannot auto-clean these. The LLM can only list them. Cleaning
        requires explicit user decision (medication safety: metformin vs
        metoprolol are not the same drug even though their names are close).
        """
        from ..cortex.cleanup import find_suspected_typo_neurons
        candidates = find_suspected_typo_neurons(self.brain)
        if not candidates:
            return "No suspected content-word typos found."
        lines = [
            f"Suspected typos requiring USER REVIEW ({len(candidates)} candidates):",
            "Sara WILL NOT auto-clean these. The user must review and decide.",
            "Drug names that look similar are different drugs — never auto-merge.",
            "",
        ]
        for c in candidates[:30]:
            lines.append(
                f"  {c.label!r} ({c.path_count} paths) — possibly typo of "
                f"{c.canonical!r} ({c.canonical_path_count} paths, "
                f"edit distance {c.edit_distance})"
            )
        if len(candidates) > 30:
            lines.append(f"  ... and {len(candidates) - 30} more")
        return "\n".join(lines)

    # ── Disambiguation ──

    def did_you_mean(self, term: str) -> str:
        """Check for close matches to a term. Returns candidates for disambiguation."""
        candidates = self.brain.did_you_mean(term)
        if not candidates:
            # No fuzzy matches needed — either exact match exists or nothing close
            n = self.brain.neuron_repo.resolve(term.strip().lower())
            if n:
                return f"'{term}' resolved to '{n.label}' (exact match)."
            return f"No matches found for '{term}'."

        lines = [f"Did you mean one of these? (searching for '{term}')"]
        for c in candidates:
            desc = f" — {c['description']}" if c['description'] else ""
            lines.append(f"  - {c['label']} ({c['type']}){desc}")
        return "\n".join(lines)

    # ── Document Ingestion ──

    def ingest(self, source: str, on_chunk=None) -> str:
        """Ingest a document from a file path or URL into Sara Brain.

        Sara reads the document via the LLM cortex, extracts facts,
        learns them as paths, and reports what she understood.
        """
        source = source.strip()

        # Determine if URL or file path
        if source.startswith("http://") or source.startswith("https://"):
            text = self._fetch_url(source)
            if text.startswith("Error"):
                return text
            label = source
        else:
            p = Path(source)
            if not p.is_file():
                return f"File not found: {source}"
            try:
                text = p.read_text(encoding="utf-8")
            except Exception as e:
                return f"Error reading {source}: {e}"
            label = p.name

        if not text.strip():
            return f"Empty document: {source}"

        try:
            result = self.brain.ingest(text, source=label, on_chunk=on_chunk)
        except ValueError as e:
            return f"Ingest error (is LLM configured?): {e}"

        lines = [f"Ingested: {label}"]
        lines.append(f"  Facts learned: {result.total_taught}")
        if result.unknown_concepts:
            lines.append(f"  Unknown concepts explored: {', '.join(result.unknown_concepts)}")
        if result.summary:
            lines.append(f"  Summary: {result.summary}")

        stats = self.brain.stats()
        lines.append(f"  Brain now: {stats['neurons']} neurons, {stats['segments']} segments, {stats['paths']} paths")
        return "\n".join(lines)

    @staticmethod
    def _fetch_url(url: str) -> str:
        """Fetch text content from a URL. Strips HTML tags for basic extraction."""
        import re
        import urllib.request

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SaraBrain/0.1 (document ingest)"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error fetching {url}: {e}"

        # Strip HTML tags for basic text extraction
        text = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        # Convert block-level tags to paragraph breaks before stripping
        text = re.sub(r"</?(p|div|br|h[1-6]|li|tr|td|th|blockquote|section|article)[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse runs of whitespace within lines, preserve paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]*\n[\n\s]*", "\n\n", text)
        text = text.strip()

        if len(text) > 50000:
            text = text[:50000]

        return text

    # ── Import/Export ──

    def import_brain(self, file_path: str) -> str:
        """Import a JSON brain export into Sara's database."""
        p = Path(file_path)
        if not p.is_file():
            return f"File not found: {file_path}"

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return f"Error reading {file_path}: {e}"

        conn = self.brain.conn
        conn.execute("PRAGMA foreign_keys=OFF")

        counts = {"neurons": 0, "segments": 0, "paths": 0}
        for n in data.get("neurons", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO neurons (id,label,neuron_type,created_at,metadata) VALUES (?,?,?,?,?)",
                    (n["id"], n["label"], n["neuron_type"], n.get("created_at", 0), None),
                )
                counts["neurons"] += 1
            except Exception:
                pass
        for s in data.get("segments", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO segments (id,source_id,target_id,relation,strength,traversals,created_at,last_used) VALUES (?,?,?,?,?,?,?,?)",
                    (s["id"], s["source_id"], s["target_id"], s["relation"],
                     s["strength"], s["traversals"], s.get("created_at", 0), s.get("last_used", 0)),
                )
                counts["segments"] += 1
            except Exception:
                pass
        for p_rec in data.get("paths", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO paths (id,origin_id,terminus_id,source_text,created_at) VALUES (?,?,?,?,?)",
                    (p_rec["id"], p_rec["origin_id"], p_rec["terminus_id"],
                     p_rec.get("source_text"), p_rec.get("created_at", 0)),
                )
                counts["paths"] += 1
            except Exception:
                pass
        for ps in data.get("path_steps", []):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO path_steps (id,path_id,step_order,segment_id) VALUES (?,?,?,?)",
                    (ps["id"], ps["path_id"], ps["step_order"], ps["segment_id"]),
                )
            except Exception:
                pass

        conn.execute("PRAGMA foreign_keys=ON")
        conn.commit()

        stats = self.brain.stats()
        return (
            f"Imported from {p.name}: "
            f"{counts['neurons']} neurons, {counts['segments']} segments, {counts['paths']} paths. "
            f"Brain now has: {stats['neurons']} neurons, {stats['segments']} segments, {stats['paths']} paths."
        )
