"""Interactive HamRobyLLM chat — REPL against a Sara brain.

  $ sara-cortex-chat --brain path/to/your/sara.db

Loads the grammar LM + router head once, then takes questions one at a
time. No LLM in the loop. Slash commands:

  /help        list commands
  /trace       toggle: show routing decision + classifier confidence
  /verbose     toggle: show raw substrate output instead of synthesized prose
  /brain PATH  switch brains without restarting
  /model       show which checkpoints are loaded
  /quit, /exit, Ctrl-D   leave
"""
from __future__ import annotations

import argparse
import atexit
import os
import sys
from pathlib import Path

# readline gives input() arrow-key history + line editing on Linux/macOS.
try:
    import readline
except ImportError:
    readline = None

from sara_brain.core.brain import Brain
from sara_reader.tools import execute_tool

from .clarify import (
    Clarification, ConceptFix, apply_wh_fix, detect_wh_typo,
    find_concept_candidates, is_cancel, parse_choice,
)
from .dig import find_siblings, is_comprehensive_intent
from .router import MODEL_FULL, CortexRouter
from .synthesizer import parse_edges_from_gathered, synthesize


# ANSI colors — disabled if stdout isn't a TTY
def _color(code: str) -> str:
    return code if sys.stdout.isatty() else ""


CYAN = _color("\033[36m")
GREEN = _color("\033[32m")
YELLOW = _color("\033[33m")
DIM = _color("\033[2m")
BOLD = _color("\033[1m")
RESET = _color("\033[0m")


def banner(model: str, brain_path: Path) -> str:
    return (
        f"{BOLD}{CYAN}╭─ {model} ──────────────────────────────╮{RESET}\n"
        f"{BOLD}{CYAN}│{RESET} brain: {brain_path}\n"
        f"{BOLD}{CYAN}│{RESET} type a question, /help for commands, /quit to exit\n"
        f"{BOLD}{CYAN}╰────────────────────────────────────────────────────╯{RESET}"
    )


HELP = """commands:
  /help              show this help
  /teach STATEMENT   teach Sara a new fact (e.g. /teach ssng1 is a goal)
  /refute STATEMENT  refute / negate an existing fact
  /teach-vocab REL PHRASE
                     teach the vocab brain a new relation -> English form.
                     v043: ADDS as alternate form (rotated at inference for
                     stylistic variety). Requires --use-hamrobysum.
  /refute-vocab REL [PHRASE]
                     remove vocab mapping(s). Without PHRASE: removes ALL
                     forms for the relation. With PHRASE: removes that one form.
  /list-vocab [REL]  show vocab mappings (all, or just for one relation).
  /multihop          toggle multi-hop reasoning on/off (v045). When on,
                     questions like "why X" or "how does Y" trigger
                     bounded BFS over substrate edges.
  /dig               expand the last query — pull sibling substrate concepts
                     and synthesize their neighborhoods with the original
  /dig CONCEPT       drill into a specific named concept directly
  /depth N           re-run the last brain_explore query at hop distance N
  /trace             toggle: show routing decision + confidence
  /verbose           toggle: print raw substrate result (no synthesis)
  /brain PATH        switch to a different brain.db
  /model             show loaded checkpoints
  /quit /exit Ctrl-D leave

In natural language, phrases like "tell me everything about X" or
"give me the complete picture of X" auto-trigger the dig expansion."""


class ChatSession:
    def __init__(
        self,
        brain_path: Path,
        grammar_ckpt: Path,
        head_ckpt: Path,
        device: str,
        hamrobysum_ckpt: Path | None = None,
        vocab_brain: Path | None = None,
        multihop: bool = False,
    ):
        self.grammar_ckpt = grammar_ckpt
        self.head_ckpt = head_ckpt
        self.device = device
        self.show_trace = False
        self.show_raw = False
        self.router = CortexRouter(
            grammar_ckpt=grammar_ckpt,
            head_ckpt=head_ckpt,
            substrate_db=brain_path,
            device=device,
        )
        self._load_brain(brain_path)
        self.pending: Clarification | None = None
        # v039 slice 3 + v040: optional HamRoby-Sum neural synthesizer
        # with vocab brain for predicate-slot expansion.
        self.hamrobysum_ckpt = hamrobysum_ckpt
        self.vocab_brain_path = vocab_brain
        self._hamrobysum_model = None
        self._vocab_lookup: dict[str, list[str]] = {}
        # v045: multi-hop reasoning toggle. When on, questions whose
        # shape suggests chaining (per multihop.should_multihop) get
        # routed through plan_chain instead of single-hop fetch.
        self.multihop_enabled = multihop
        if hamrobysum_ckpt is not None:
            from .inference_synth import load_synth_checkpoint, load_vocab_brain
            import torch
            print(f"{DIM}[hamrobysum] loading {hamrobysum_ckpt}{RESET}")
            self._hamrobysum_model = load_synth_checkpoint(
                hamrobysum_ckpt, torch.device(device),
            )
            if vocab_brain is not None and vocab_brain.exists():
                self._vocab_lookup = load_vocab_brain(vocab_brain)
                print(f"{DIM}[hamrobysum] loaded {len(self._vocab_lookup)} "
                      f"relation -> english_form mappings from {vocab_brain}{RESET}")
            else:
                print(f"{DIM}[hamrobysum] no vocab brain at {vocab_brain}; "
                      f"predicates fall back to relation_name underscores->spaces{RESET}")
        # Track the most recent successful query so /dig and /depth can
        # extend it without the user retyping the topic.
        self.last_question: str | None = None
        self.last_decision: dict | None = None  # {"tool": ..., **args}
        self.last_topic: str | None = None      # the substrate label or concept

    def _load_brain(self, brain_path: Path) -> None:
        self.brain_path = brain_path
        self.brain = Brain(str(brain_path))
        # Re-bind the router's substrate index too.
        from .router_args import SubstrateIndex
        self.router.substrate = SubstrateIndex(brain_path)

    def _synthesize_one_gathered(
        self, question: str | None, gathered: list[dict],
    ) -> str:
        """Render a single gathered list (one or more entries from the
        SAME hop) into prose. Caller decides how to glue multiple
        hops together (see `_synthesize`)."""
        if self._hamrobysum_model is None:
            return synthesize(question or "", gathered)

        from .inference_synth import synthesize_cluster
        from .synth_data import cluster_by_subject
        import torch

        edges = parse_edges_from_gathered(gathered)
        if not edges:
            return synthesize(question or "", gathered)
        clusters = cluster_by_subject(edges)
        device = torch.device(self.device)
        out_parts: list[str] = []
        for subject, cluster in clusters.items():
            prose = synthesize_cluster(
                self._hamrobysum_model, cluster, device,
                max_new_tokens=80, temperature=0.0,
                repetition_penalty=1.1, no_repeat_ngram_size=4,
                vocab_lookup=self._vocab_lookup,
            )
            stripped = prose.strip(" .,;:!?")
            if not stripped:
                # Degenerate cluster — fall back to template for THIS cluster.
                fallback_gathered = [{"call": {"tool": "brain_explore",
                                               "args": {"label": subject}},
                                       "result": "\n".join(
                                           f"'{e.src}' --[{e.rel}]--> '{e.tgt}'"
                                           + ('_attribute' if e.target_was_attribute else '')
                                           + (' [REFUTED]' if e.refuted else '')
                                           for e in cluster
                                       )}]
                prose = synthesize(question or subject, fallback_gathered)
            out_parts.append(prose)
        return " ".join(p for p in out_parts if p)

    def _synthesize(self, question: str | None, gathered: list[dict]) -> str:
        """Render gathered substrate facts as prose. Routes through
        HamRoby-Sum if loaded, falling back to the v032 template
        renderer per-cluster on degenerate output (v039 slice 3).

        v045: when `gathered` has multiple entries (multi-hop), render
        each entry separately and join with " Additionally, " — the
        connector is structural, not invented reasoning. Single-entry
        gathered (the typical case) renders unchanged."""
        if not gathered:
            return ""
        if len(gathered) == 1:
            return self._synthesize_one_gathered(question, gathered)
        # Multi-hop: render each gathered entry separately, connect.
        parts: list[str] = []
        for g in gathered:
            prose = self._synthesize_one_gathered(question, [g])
            stripped = prose.strip()
            if stripped:
                parts.append(stripped)
        if not parts:
            return ""
        return parts[0] + "".join(
            (" Additionally, " + p[0].lower() + p[1:]) if p else ""
            for p in parts[1:]
        )

    def ask(self, question: str) -> None:
        # Stage 1: detect leading-token typos (waht -> what, etc.)
        wh_fix = detect_wh_typo(question)
        if wh_fix:
            self.pending = Clarification(original_question=question, wh_fix=wh_fix)
            print(f"{YELLOW}{self.pending.render_prompt()}{RESET}")
            return
        # Comprehensive intent in the question itself triggers dig automatically.
        comprehensive = is_comprehensive_intent(question)
        self._route_and_run(question, expand=comprehensive)

    def _route_and_run(self, question: str, expand: bool = False) -> None:
        decision = self.router.route(question)
        if self.show_trace:
            print(f"{DIM}[{decision.model}] tool={decision.tool}  "
                  f"cls_conf={decision.classifier_confidence:.2f}  "
                  f"args={decision.args}  why={decision.rationale}{RESET}")
        result = execute_tool(self.brain, decision.tool, decision.args)

        # v045 follow-up (b): brain_value sometimes refuses with a
        # "No definitional edges found" guard for concepts the
        # substrate knows about but has no is_a/defined_as edge for.
        # Honest behaviour but unhelpful — the user wanted SOMETHING
        # about the concept. Auto-fallback to brain_explore so the
        # user sees the available edges instead of a bare refusal.
        if (decision.tool == "brain_value"
                and isinstance(result, str)
                and result.startswith("No definitional edges found")):
            anchor = (decision.args.get("concept")
                      or decision.args.get("label")
                      or decision.args.get("term"))
            if anchor:
                if self.show_trace:
                    print(f"{DIM}[fallback] brain_value -> brain_explore for {anchor!r}{RESET}")
                import dataclasses
                fallback_args = {"label": anchor, "depth": 1}
                result = execute_tool(self.brain, "brain_explore", fallback_args)
                decision = dataclasses.replace(
                    decision,
                    tool="brain_explore",
                    args=fallback_args,
                    rationale=f"{decision.rationale} (fell back from brain_value)",
                )

        # Stage 2: substrate "no neuron matching" -> ask did_you_mean.
        miss_field, miss_value = self._extract_miss(decision.tool, decision.args, result)
        if miss_field is not None:
            cands = find_concept_candidates(self.brain, miss_value)
            if cands:
                pending_decision = {"tool": decision.tool, **decision.args}
                self.pending = Clarification(
                    original_question=question,
                    concept_fix=ConceptFix(
                        original=miss_value, candidates=cands, field=miss_field,
                    ),
                    pending_router_decision=pending_decision,
                )
                print(f"{YELLOW}{self.pending.render_prompt()}{RESET}")
                return
            # No candidates — fall through to the honest "no neuron" message.

        # Remember the last query so /dig and /depth can extend it.
        self.last_question = question
        self.last_decision = {"tool": decision.tool, **decision.args}
        self.last_topic = self._topic_from_decision(self.last_decision)

        # v045: multi-hop reasoning. If --multihop is enabled and the
        # question shape suggests chaining, run the bounded BFS
        # orchestrator instead of the single-hop fetch.
        if self.multihop_enabled:
            from .multihop import should_multihop, plan_chain
            if should_multihop(question):
                if self.show_trace:
                    print(f"{DIM}[multihop] question shape triggers chain orchestration{RESET}")
                gathered = plan_chain(self.brain, self.last_decision)
                if self.show_trace:
                    print(f"{DIM}[multihop] gathered {len(gathered)} hops{RESET}")
            else:
                gathered = [{"call": {"tool": decision.tool, "args": decision.args},
                             "result": result}]
        else:
            gathered = [{"call": {"tool": decision.tool, "args": decision.args},
                         "result": result}]

        if expand and self.last_topic:
            self._gather_siblings_into(self.last_topic, gathered)

        if self.show_raw:
            for fact in gathered:
                print(fact["result"])
            return
        prose = self._synthesize(question, gathered)
        print(prose)

    @staticmethod
    def _topic_from_decision(decision: dict) -> str | None:
        for f in ("concept", "label", "term"):
            if f in decision:
                return decision[f]
        return None

    def _gather_siblings_into(
        self, topic: str, gathered: list[dict], max_siblings: int = 8,
    ) -> None:
        """Find substrate concepts whose words overlap with `topic` and
        append their brain_explore output to `gathered`. Skips concepts
        already present so we don't duplicate work."""
        already = {self._topic_from_decision(g["call"]["args"]) or "" for g in gathered}
        siblings = find_siblings(self.brain_path, topic, exclude=already,
                                 max_results=max_siblings)
        if not siblings:
            return
        print(f"{DIM}also exploring: {', '.join(siblings)}{RESET}")
        for sib in siblings:
            args = {"label": sib, "depth": 1}
            try:
                res = execute_tool(self.brain, "brain_explore", args)
            except Exception as e:
                res = f"<<error exploring {sib!r}: {e}>>"
            gathered.append({"call": {"tool": "brain_explore", "args": args},
                             "result": res})

    def do_dig(self, arg: str) -> None:
        """If `arg` is empty, expand the last query. Otherwise drill into
        the named concept (treats `arg` as a substrate label)."""
        if arg.strip():
            label = arg.strip()
            args = {"label": label, "depth": 1}
            try:
                res = execute_tool(self.brain, "brain_explore", args)
            except Exception as e:
                print(f"{YELLOW}error: {e}{RESET}")
                return
            self.last_question = f"tell me everything about {label}"
            self.last_decision = {"tool": "brain_explore", **args}
            self.last_topic = label
            gathered = [{"call": {"tool": "brain_explore", "args": args}, "result": res}]
            self._gather_siblings_into(label, gathered)
            print(self._synthesize(self.last_question, gathered))
            return

        if self.last_topic is None:
            print(f"{YELLOW}nothing to dig — ask a question first{RESET}")
            return
        # Re-run the last query and extend it with siblings.
        original = execute_tool(self.brain, self.last_decision["tool"],
                                {k: v for k, v in self.last_decision.items() if k != "tool"})
        gathered = [{"call": {"tool": self.last_decision["tool"],
                              "args": {k: v for k, v in self.last_decision.items() if k != "tool"}},
                     "result": original}]
        self._gather_siblings_into(self.last_topic, gathered)
        print(self._synthesize(self.last_question or self.last_topic, gathered))

    def do_depth(self, arg: str) -> None:
        try:
            depth = int(arg.strip())
        except ValueError:
            print(f"{YELLOW}usage: /depth N (1..4){RESET}")
            return
        if not 1 <= depth <= 4:
            print(f"{YELLOW}depth must be 1..4 (got {depth}){RESET}")
            return
        if self.last_topic is None:
            print(f"{YELLOW}nothing to widen — ask a question first{RESET}")
            return
        args = {"label": self.last_topic, "depth": depth}
        try:
            res = execute_tool(self.brain, "brain_explore", args)
        except Exception as e:
            print(f"{YELLOW}error: {e}{RESET}")
            return
        self.last_decision = {"tool": "brain_explore", **args}
        gathered = [{"call": {"tool": "brain_explore", "args": args}, "result": res}]
        print(self._synthesize(self.last_question or self.last_topic, gathered))

    @staticmethod
    def _extract_miss(tool: str, args: dict, result: str) -> tuple[str | None, str | None]:
        """If the tool returned a 'no neuron matching' response, return
        (which arg field needs replacing, what the missing value was)."""
        if "no neuron matching" not in result.lower():
            return None, None
        for field in ("concept", "label", "term"):
            if field in args:
                return field, args[field]
        return None, None

    def handle_pending_response(self, line: str) -> None:
        """User typed something while a clarification was pending. Either
        a numeric choice, 'no' to cancel, or a brand new question."""
        if not self.pending:
            return
        if is_cancel(line):
            print(f"{DIM}cancelled{RESET}")
            self.pending = None
            return

        n_opts = (len(self.pending.wh_fix.candidates)
                  if self.pending.wh_fix else
                  len(self.pending.concept_fix.candidates))
        choice = parse_choice(line, n_opts)
        if choice is None:
            # Treat as a fresh question.
            self.pending = None
            self.ask(line)
            return

        if self.pending.wh_fix:
            fixed = apply_wh_fix(self.pending.original_question,
                                 self.pending.wh_fix,
                                 self.pending.wh_fix.candidates[choice])
            print(f"{DIM}-> {fixed}{RESET}")
            self.pending = None
            self.ask(fixed)
            return

        # Concept fix: re-run the same tool with the chosen substrate label.
        cf = self.pending.concept_fix
        chosen = cf.candidates[choice]["label"]
        decision = dict(self.pending.pending_router_decision)
        decision[cf.field] = chosen
        original_q = self.pending.original_question
        self.pending = None
        print(f"{DIM}-> {cf.field}={chosen!r}{RESET}")
        tool = decision.pop("tool")
        result = execute_tool(self.brain, tool, decision)
        if self.show_raw:
            print(result)
            return
        gathered = [{"call": {"tool": tool, "args": decision}, "result": result}]
        print(self._synthesize(original_q, gathered))

    def handle_command(self, line: str) -> bool:
        """Run a slash command. Returns False if the session should end."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/trace":
            self.show_trace = not self.show_trace
            print(f"trace = {self.show_trace}")
        elif cmd == "/verbose":
            self.show_raw = not self.show_raw
            print(f"verbose (raw substrate output) = {self.show_raw}")
        elif cmd == "/brain":
            if not arg:
                print(f"current brain: {self.brain_path}")
            else:
                p = Path(arg)
                if not p.exists():
                    print(f"{YELLOW}brain not found: {p}{RESET}")
                else:
                    self._load_brain(p)
                    print(f"switched to: {p}")
        elif cmd == "/model":
            print(f"  grammar:    {self.grammar_ckpt}")
            print(f"  router:     {self.head_ckpt}")
            print(f"  hamrobysum: {self.hamrobysum_ckpt or '(off — using v032 templates)'}")
            print(f"  multihop:   {'on' if self.multihop_enabled else 'off'}")
        elif cmd == "/teach":
            if not arg:
                print(f"{YELLOW}usage: /teach <statement>{RESET}")
            else:
                self._do_teach(arg)
        elif cmd == "/refute":
            if not arg:
                print(f"{YELLOW}usage: /refute <statement>{RESET}")
            else:
                self._do_refute(arg)
        elif cmd == "/teach-vocab":
            if not arg or " " not in arg.strip():
                print(f"{YELLOW}usage: /teach-vocab RELATION PHRASE...{RESET}")
            else:
                self._do_teach_vocab(arg)
        elif cmd == "/refute-vocab":
            if not arg.strip():
                print(f"{YELLOW}usage: /refute-vocab RELATION [PHRASE]{RESET}")
            else:
                self._do_refute_vocab(arg)
        elif cmd == "/list-vocab":
            self._do_list_vocab(arg)
        elif cmd == "/multihop":
            self.multihop_enabled = not self.multihop_enabled
            state = "on" if self.multihop_enabled else "off"
            print(f"{GREEN}multihop: {state}{RESET}")
        elif cmd == "/dig":
            self.do_dig(arg)
        elif cmd == "/depth":
            self.do_depth(arg)
        else:
            print(f"{YELLOW}unknown command: {cmd}  (try /help){RESET}")
        return True

    def _do_teach(self, statement: str) -> None:
        try:
            result = self.brain.teach(statement)
        except Exception as e:
            print(f"{YELLOW}teach error: {e}{RESET}")
            return
        if result is None:
            print(f"{YELLOW}couldn't parse: {statement!r}{RESET}")
            return
        path_id = getattr(result, "path_id", "?")
        print(f"{GREEN}learned (path #{path_id}): {statement}{RESET}")

    def _vocab_check(self) -> bool:
        """Common precondition for vocab management commands. Returns
        True if usable; prints + returns False otherwise."""
        if self._hamrobysum_model is None or self.vocab_brain_path is None:
            print(f"{YELLOW}vocab brain not loaded — restart with --use-hamrobysum{RESET}")
            return False
        if not self.vocab_brain_path.exists():
            print(f"{YELLOW}vocab brain not found at {self.vocab_brain_path}{RESET}")
            return False
        return True

    def _do_teach_vocab(self, arg: str) -> None:
        """Teach the vocab brain a new relation -> English mapping.
        v043: ADDS as an alternate form rather than replacing.
        To replace, /refute-vocab RELATION first then /teach-vocab.
        Duplicate (same relation + same phrase) is a no-op."""
        if not self._vocab_check():
            return

        parts = arg.strip().split(maxsplit=1)
        if len(parts) < 2:
            print(f"{YELLOW}usage: /teach-vocab RELATION PHRASE...{RESET}")
            return
        relation, phrase = parts[0].strip(), parts[1].strip()
        if not relation or not phrase:
            print(f"{YELLOW}usage: /teach-vocab RELATION PHRASE...{RESET}")
            return

        import sqlite3
        import time
        conn = sqlite3.connect(str(self.vocab_brain_path))
        try:
            # Find or create the relation neuron.
            row = conn.execute(
                "SELECT id FROM neurons WHERE label = ?", (relation,)
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    "INSERT INTO neurons (label, neuron_type, created_at) "
                    "VALUES (?, ?, ?)",
                    (relation, "vocab", time.time()),
                )
                rel_id = cur.lastrowid
            else:
                rel_id = row[0]
            # Find or create the phrase neuron.
            row = conn.execute(
                "SELECT id FROM neurons WHERE label = ?", (phrase,)
            ).fetchone()
            if row is None:
                cur = conn.execute(
                    "INSERT INTO neurons (label, neuron_type, created_at) "
                    "VALUES (?, ?, ?)",
                    (phrase, "vocab", time.time()),
                )
                phrase_id = cur.lastrowid
            else:
                phrase_id = row[0]
            # ADD (not replace): silently no-op if the same form already
            # exists for this relation (UNIQUE(source, target, relation)
            # constraint is the safety net).
            cur = conn.execute(
                "INSERT OR IGNORE INTO segments "
                "(source_id, target_id, relation, created_at) "
                "VALUES (?, ?, ?, ?)",
                (rel_id, phrase_id, "english_form", time.time()),
            )
            added = cur.rowcount > 0
            conn.commit()
        finally:
            conn.close()

        # Update in-memory lookup (append-not-replace).
        forms = self._vocab_lookup.setdefault(relation, [])
        if phrase not in forms:
            forms.append(phrase)
        if added:
            n_forms = len(forms)
            note = f" ({n_forms} forms total)" if n_forms > 1 else ""
            print(f"{GREEN}vocab added: {relation} -> {phrase!r}{note}{RESET}")
        else:
            print(f"{DIM}vocab unchanged: {relation} -> {phrase!r} already exists{RESET}")

    def _do_refute_vocab(self, arg: str) -> None:
        """Remove vocab mapping(s) for a relation. v043.

        /refute-vocab RELATION         — remove ALL forms for the relation
        /refute-vocab RELATION PHRASE  — remove only the specific form
        """
        if not self._vocab_check():
            return

        parts = arg.strip().split(maxsplit=1)
        if not parts or not parts[0].strip():
            print(f"{YELLOW}usage: /refute-vocab RELATION [PHRASE]{RESET}")
            return
        relation = parts[0].strip()
        phrase = parts[1].strip() if len(parts) > 1 else None

        import sqlite3
        conn = sqlite3.connect(str(self.vocab_brain_path))
        try:
            row = conn.execute(
                "SELECT id FROM neurons WHERE label = ?", (relation,)
            ).fetchone()
            if row is None:
                print(f"{YELLOW}no vocab entry for relation {relation!r}{RESET}")
                return
            rel_id = row[0]
            if phrase is None:
                # Remove all forms.
                cur = conn.execute(
                    "DELETE FROM segments WHERE source_id = ? AND relation = ?",
                    (rel_id, "english_form"),
                )
                n = cur.rowcount
                conn.commit()
                if n > 0:
                    self._vocab_lookup.pop(relation, None)
                    print(f"{GREEN}vocab refuted: removed all {n} form(s) for {relation!r}{RESET}")
                else:
                    print(f"{YELLOW}no forms to remove for {relation!r}{RESET}")
            else:
                # Remove only the specific phrase.
                row = conn.execute(
                    "SELECT id FROM neurons WHERE label = ?", (phrase,)
                ).fetchone()
                if row is None:
                    print(f"{YELLOW}no neuron for phrase {phrase!r}{RESET}")
                    return
                phrase_id = row[0]
                cur = conn.execute(
                    "DELETE FROM segments WHERE source_id = ? AND target_id = ? "
                    "AND relation = ?",
                    (rel_id, phrase_id, "english_form"),
                )
                n = cur.rowcount
                conn.commit()
                if n > 0:
                    forms = self._vocab_lookup.get(relation, [])
                    if phrase in forms:
                        forms.remove(phrase)
                    if not forms:
                        self._vocab_lookup.pop(relation, None)
                    remaining = len(self._vocab_lookup.get(relation, []))
                    note = f"; {remaining} form(s) remain" if remaining else ""
                    print(f"{GREEN}vocab refuted: {relation} -> {phrase!r}{note}{RESET}")
                else:
                    print(f"{YELLOW}{relation} -> {phrase!r} not found in vocab brain{RESET}")
        finally:
            conn.close()

    def _do_list_vocab(self, arg: str) -> None:
        """Show vocab mappings. v043.

        /list-vocab            — print all relation -> form(s)
        /list-vocab RELATION   — print all forms for one relation
        """
        if not self._vocab_check():
            return
        target = arg.strip() or None
        if not self._vocab_lookup:
            print(f"{YELLOW}vocab brain has no mappings{RESET}")
            return
        if target:
            forms = self._vocab_lookup.get(target)
            if not forms:
                print(f"{YELLOW}no vocab entry for {target!r}{RESET}")
                return
            print(f"{CYAN}{target}{RESET}")
            for f in forms:
                print(f"  -> {f!r}")
            return
        # No arg — print everything alphabetized.
        n_relations = len(self._vocab_lookup)
        n_forms = sum(len(v) for v in self._vocab_lookup.values())
        print(f"{DIM}{n_relations} relation(s), {n_forms} form(s) total{RESET}")
        for relation in sorted(self._vocab_lookup):
            forms = self._vocab_lookup[relation]
            if len(forms) == 1:
                print(f"  {relation:25s} -> {forms[0]!r}")
            else:
                print(f"  {CYAN}{relation}{RESET}")
                for f in forms:
                    print(f"    -> {f!r}")

    def _do_refute(self, statement: str) -> None:
        try:
            result = self.brain.refute(statement)
        except Exception as e:
            print(f"{YELLOW}refute error: {e}{RESET}")
            return
        if result is None:
            print(f"{YELLOW}nothing matching to refute: {statement!r}{RESET}")
            return
        path_id = getattr(result, "path_id", "?")
        print(f"{GREEN}refuted (path #{path_id}): {statement}{RESET}")


def _save_history(path: Path) -> None:
    if readline is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(path))
    except OSError:
        pass


def main() -> int:
    p = argparse.ArgumentParser(description="Interactive HamRobyLLM chat")
    p.add_argument("--brain", type=Path, required=True,
                   help="Path to a Sara brain.db (must end in .db)")
    p.add_argument("--grammar-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/grammar_base_015000.pt"))
    p.add_argument("--head-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/router_head.pt"))
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda; cpu is plenty for serving")
    p.add_argument("--use-hamrobysum", action="store_true",
                   help="Route prose synthesis through HamRoby-Sum (per "
                        "v039) with v032 template fallback for degenerate "
                        "clusters. Default: pure v032 templates.")
    p.add_argument("--hamrobysum-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/hamroby_sum_en_002500.pt"),
                   help="HamRoby-Sum checkpoint (only used with "
                        "--use-hamrobysum). Default: hamroby_sum_en_002500.pt.")
    p.add_argument("--vocab-brain", type=Path,
                   default=Path("src/sara_brain/cortex/vocab/vocab_en.db"),
                   help="Vocab brain (per v040): a Sara brain.db that maps "
                        "relation names to English phrases. Used by "
                        "--use-hamrobysum to expand <Pn> predicate slots. "
                        "Default: vocab_en.db.")
    p.add_argument("--multihop", action="store_true",
                   help="Enable multi-hop reasoning over substrate (per "
                        "v045). Questions whose shape suggests chaining "
                        "(why / how does / because / caused by / ...) get "
                        "routed through bounded BFS over substrate edges. "
                        "Single-hop questions ('what is X') stay single-hop. "
                        "Default: off.")
    args = p.parse_args()

    if not args.brain.exists():
        print(f"brain not found: {args.brain}", file=sys.stderr)
        return 1
    if not args.grammar_ckpt.exists():
        print(f"grammar checkpoint not found: {args.grammar_ckpt}", file=sys.stderr)
        return 1
    if not args.head_ckpt.exists():
        print(f"router head checkpoint not found: {args.head_ckpt}", file=sys.stderr)
        return 1
    if args.use_hamrobysum and not args.hamrobysum_ckpt.exists():
        print(f"hamrobysum checkpoint not found: {args.hamrobysum_ckpt}", file=sys.stderr)
        return 1

    if readline is not None:
        hist = Path(os.path.expanduser("~/.hamroby_history"))
        try:
            readline.read_history_file(str(hist))
        except (OSError, FileNotFoundError):
            pass
        readline.set_history_length(2000)
        atexit.register(_save_history, hist)

    print(f"{DIM}loading {MODEL_FULL}...{RESET}", flush=True)
    session = ChatSession(
        args.brain, args.grammar_ckpt, args.head_ckpt, args.device,
        hamrobysum_ckpt=args.hamrobysum_ckpt if args.use_hamrobysum else None,
        vocab_brain=args.vocab_brain if args.use_hamrobysum else None,
        multihop=args.multihop,
    )
    print(banner(MODEL_FULL, args.brain))

    while True:
        try:
            line = input(f"{BOLD}{GREEN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            cont = session.handle_command(line)
            if not cont:
                return 0
            continue
        try:
            if session.pending is not None:
                session.handle_pending_response(line)
            else:
                session.ask(line)
        except Exception as e:
            print(f"{YELLOW}error: {e}{RESET}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
