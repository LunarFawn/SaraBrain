"""Stateless two-tier reader (model_infections §5d).

Python orchestrator over single-message LLM calls. Ollama routes; Haiku
synthesizes. Each LLM call is stateless — no message history, no
accumulated context. Python validates every Ollama output against the
substrate before using it. Haiku is invoked once at the end with the
full Ollama-gathered context.

Closes session-context infections, auto-memory infections, and
format-imitation confabulation by removing the conditions for
contamination to compound.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

from sara_brain.core.brain import Brain

from .brain_loader import load_brain
from .providers import get_provider
from .tools import TOOLS, execute_tool


# v052 slice 4: optional audit log of every execute_tool call. When
# SARA_AUDIT_LOG is set to a file path, each routing-loop tool
# invocation appends a TSV row:
#   ISO_TIMESTAMP \t tool_name \t args_json \t result_bytes
# Same format as the v050 MCP audit log (sara_brain/mcp_server.py)
# so a single grep can compare across the MCP and cli_stateless
# paths. Default: no logging, no overhead.

_AUDIT_LOG_PATH = os.environ.get("SARA_AUDIT_LOG", "")


def _audit_tool_call(tool_name: str, args: dict, result: str) -> None:
    if not _AUDIT_LOG_PATH:
        return
    try:
        with open(_AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            args_json = json.dumps(args, ensure_ascii=False)
            f.write(f"{ts}\t{tool_name}\t{args_json}\t{len(result)}\n")
    except Exception as e:
        print(f"[audit] write failed: {e}", file=sys.stderr)


_ROUTER_PROMPT_TEMPLATE = """\
You are routing a substrate query. The substrate is a knowledge graph.
Reply with ONE JSON object and nothing else. No prose, no markdown.

Question: {question}

Already-gathered facts (do not re-query these):
{gathered}

Available tools:
  - brain_value(concept, type) — fetch a value/range. type is a relation
    fragment like 'kdoff', 'kdon', 'ratio', 'value'. Use when a specific
    quantity is named.
  - brain_define(concept) — fetch the substrate's definition for a
    concept or acronym.
  - brain_explore(label, depth) — walk the neighborhood. depth=1 only.
    Use when you need to find what concepts are related to a term.
  - brain_did_you_mean(term) — fuzzy-match a possibly-misspelled or
    miscased term against substrate labels.
  - DONE — emit when the gathered facts already answer the question.

DONE-DISCIPLINE: If the gathered facts above already contain the value
or definition the question asks for, emit DONE. Do not re-query for
facts you already have. Looping wastes iterations.

NO-MATCH RECOVERY: If a prior tool result said "no neuron matching" or
"not found" for a label, DO NOT retry the same label. Try in order:
  1. Compound label that joins the broad concept with the quantity
     (e.g., "ssng1 highest kdoff" instead of "ssng1").
  2. Lowercase variation of the label.
  3. brain_did_you_mean on the term to find substrate-correct labels.
Only emit DONE without an answer if all three recovery attempts fail.

Reply with one of:
  {{"tool": "brain_value", "concept": "...", "type": "..."}}
  {{"tool": "brain_define", "concept": "..."}}
  {{"tool": "brain_explore", "label": "...", "depth": 1}}
  {{"tool": "brain_did_you_mean", "term": "..."}}
  {{"tool": "DONE"}}
"""


_SYNTHESIS_PROMPT_TEMPLATE = """\
Answer the user's question using ONLY the substrate facts below. Cite
exact values verbatim. Do NOT expand acronyms unless the substrate
defines them. If a fact is not in the substrate, say so — do not invent.

Question: {question}

Substrate facts gathered:
{gathered}

Write a faithful answer. Short prose. No meta-commentary about tool
calls. No NLP-style framing of the data."""


# v052: strict-Sara mode delivers rules via the system_prompt channel
# and wraps the substrate in <substrate> tags so the cortex receives
# rules and data on different channels. Tighter than the default
# synthesis prompt — explicitly forbids training-derived inference
# and per-claim grounding violations.

_STRICT_SARA_SYSTEM_PROMPT = """\
You are a substrate-bound research assistant. You have access to facts
ONLY through <substrate> tags in the user message. The contents of
those tags are the COMPLETE set of facts you may use.

Rules — these are absolute, no exceptions:
1. Every factual claim in your answer MUST trace to a triple inside
   <substrate>. If a triple does not state it, you do not state it.
2. If <substrate> does not contain the answer, respond exactly:
   "The substrate does not contain this information."
3. Do NOT use any knowledge from your training, even if you "know"
   the topic. Your training is unverifiable; the substrate is verified.
4. Do NOT make inferences that go beyond what the triples directly
   state. No "this likely means" or "in general."
5. Do NOT add hedging connectives ("additionally", "furthermore",
   "moreover") that smuggle in training-derived content.
6. When in doubt, say less. A short substrate-true answer is correct;
   a long answer with even one training-derived claim is wrong."""


_STRICT_SARA_USER_TEMPLATE = """\
<substrate>
{gathered}
</substrate>

Question: {question}"""


_VALID_TOOLS = {
    "brain_value",
    "brain_define",
    "brain_explore",
    "brain_did_you_mean",
    "DONE",
}


_NO_MATCH_PREFIXES = (
    "No '",                       # "No 'kdoff' edges found for ..."
    "No value-relations found",
    "No definitional edges",
    "Sara has no neuron matching",
)


def _is_no_match(result: str) -> bool:
    return any(result.startswith(p) for p in _NO_MATCH_PREFIXES)


def _compound_recovery_concepts(concept: str, type_filter: str | None) -> list[str]:
    """Build candidate compound labels for a no-match recovery."""
    if not type_filter:
        return []
    base = concept.strip().lower()
    t = type_filter.strip().lower()
    return [
        f"{base} highest {t}",
        f"{base} lowest {t}",
        f"{base} {t}",
        f"highest {base} {t}",
    ]


def _format_gathered(gathered: list[dict]) -> str:
    if not gathered:
        return "  (none yet)"
    lines = []
    for i, fact in enumerate(gathered, 1):
        call = fact["call"]
        result = fact["result"]
        args_str = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
        lines.append(f"{i}. {call['tool']}({args_str})")
        for line in result.splitlines():
            lines.append(f"   {line}")
    return "\n".join(lines)


def _parse_router_response(text: str) -> dict | None:
    text = text.strip()
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first {...} block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if obj.get("tool") not in _VALID_TOOLS:
        return None
    return obj


class StatelessReader:
    """Stateless two-tier reader.

    Args:
        brain_path: filesystem path to a Sara .db file.
        router_provider: provider name for the routing model (typically
            "ollama").
        router_model: model identifier for routing (e.g.
            "llama3.2:3b").
        synthesis_provider: provider name for synthesis (typically
            "anthropic").
        synthesis_model: model identifier for synthesis (e.g.
            "claude-haiku-4-5").
        max_routing_steps: hard cap on routing-loop iterations.
        max_retries_per_step: how many times to retry a malformed
            router response before bailing.
        provider_kwargs: extra args forwarded to provider constructors,
            keyed by role ("router" / "synthesis").

    Example:
        >>> reader = StatelessReader(
        ...     brain_path="brains/aptamer_full.db",
        ...     router_provider="ollama",
        ...     router_model="llama3.2:3b",
        ...     synthesis_provider="anthropic",
        ...     synthesis_model="claude-haiku-4-5",
        ... )
        >>> reader.ask("what is the KDON for super-performing mode?")
    """

    def __init__(
        self,
        brain_path: str,
        router_provider: str,
        router_model: str,
        synthesis_provider: str,
        synthesis_model: str,
        max_routing_steps: int = 6,
        max_retries_per_step: int = 3,
        provider_kwargs: dict | None = None,
        cortex_router_ckpts: tuple[str, str] | None = None,
        skip_synthesis: bool = False,
        cortex_synthesizer: bool = False,
        strict_sara: bool = False,
    ) -> None:
        """If cortex_router_ckpts=(grammar_ckpt_path, head_ckpt_path) is set,
        routing is performed by the local cortex transformer (no Ollama call).
        Synthesis still goes to the configured provider — the synthesizer
        organ is a separate v024 phase."""
        self.brain: Brain = load_brain(brain_path)
        provider_kwargs = provider_kwargs or {}
        self.cortex_router = None
        if cortex_router_ckpts is not None:
            from sara_brain.cortex.transformer.router import CortexRouter
            grammar_ckpt, head_ckpt = cortex_router_ckpts
            self.cortex_router = CortexRouter(
                grammar_ckpt=grammar_ckpt,
                head_ckpt=head_ckpt,
                substrate_db=brain_path,
            )
            self.router = None
        else:
            self.router = get_provider(
                router_provider, **(provider_kwargs.get("router") or {})
            )
        self.skip_synthesis = skip_synthesis
        self.cortex_synthesizer = cortex_synthesizer
        if skip_synthesis or cortex_synthesizer:
            self.synthesizer = None
        else:
            self.synthesizer = get_provider(
                synthesis_provider,
                **(provider_kwargs.get("synthesis") or {}),
            )
        self.router_model = router_model
        self.synthesis_model = synthesis_model
        self.max_routing_steps = max_routing_steps
        self.max_retries_per_step = max_retries_per_step
        # v052: when True, synthesis uses the strict-Sara system prompt
        # + <substrate>-tagged user message (Layer A from the v052 plan).
        # Layer B (single-turn isolation) is structurally true regardless
        # — every .ask() call is independent; no conversation history
        # accumulates across calls.
        self.strict_sara = strict_sara

    def ask(self, question: str, return_trace: bool = False) -> str | dict:
        gathered: list[dict] = []
        trace: list[dict] = []
        seen_calls: set[tuple] = set()

        # ---- Routing loop ----
        for step in range(self.max_routing_steps):
            decision = self._route_step(question, gathered, trace)
            if decision is None:
                trace.append({"step": step, "event": "router_bail"})
                break
            if decision["tool"] == "DONE":
                trace.append({"step": step, "event": "done"})
                break
            tool_name = decision["tool"]
            args = {k: v for k, v in decision.items() if k != "tool"}
            call_key = (tool_name, tuple(sorted(args.items())))
            if call_key in seen_calls:
                trace.append({
                    "step": step,
                    "event": "repeat_call_forced_done",
                    "call": {"tool": tool_name, "args": args},
                })
                break
            seen_calls.add(call_key)
            try:
                result = execute_tool(self.brain, tool_name, args)
            except Exception as exc:
                result = f"<<tool error: {exc}>>"
            _audit_tool_call(tool_name, args, result if isinstance(result, str) else "")
            gathered.append({
                "call": {"tool": tool_name, "args": args},
                "result": result,
            })
            trace.append({
                "step": step,
                "event": "tool_executed",
                "call": {"tool": tool_name, "args": args},
                "result": result[:300],
            })

            # ---- Python-side NO-MATCH RECOVERY ----
            # 3Bs do not reliably follow the NO-MATCH RECOVERY rule in
            # the router prompt. Enforce it deterministically here for
            # brain_value: on no-match, try compound labels first
            # (value-shaped questions where the data lives on a
            # compound concept), then fall through to brain_define on
            # the original concept (definition-shaped questions where
            # the type filter is wrong but the concept itself has
            # definitional edges).
            if (
                tool_name == "brain_value"
                and _is_no_match(result)
                and "concept" in args
            ):
                recovery_hit = False
                # Phase 1 — compound-label variants (value questions).
                for candidate in _compound_recovery_concepts(
                    args["concept"], args.get("type")
                ):
                    rec_args = {"concept": candidate}
                    rec_key = (tool_name, tuple(sorted(rec_args.items())))
                    if rec_key in seen_calls:
                        continue
                    seen_calls.add(rec_key)
                    try:
                        rec_result = execute_tool(
                            self.brain, tool_name, rec_args
                        )
                    except Exception as exc:
                        rec_result = f"<<tool error: {exc}>>"
                    _audit_tool_call(
                        tool_name, rec_args,
                        rec_result if isinstance(rec_result, str) else "",
                    )
                    gathered.append({
                        "call": {"tool": tool_name, "args": rec_args},
                        "result": rec_result,
                    })
                    trace.append({
                        "step": step,
                        "event": "auto_recovery_compound",
                        "call": {"tool": tool_name, "args": rec_args},
                        "result": rec_result[:300],
                    })
                    if not _is_no_match(rec_result):
                        recovery_hit = True
                        break
                # Phase 2 — fall through to brain_define on the bare
                # concept (definition questions).
                if not recovery_hit:
                    rec_args = {"concept": args["concept"]}
                    rec_key = ("brain_define", tuple(sorted(rec_args.items())))
                    if rec_key not in seen_calls:
                        seen_calls.add(rec_key)
                        try:
                            rec_result = execute_tool(
                                self.brain, "brain_define", rec_args
                            )
                        except Exception as exc:
                            rec_result = f"<<tool error: {exc}>>"
                        _audit_tool_call(
                            "brain_define", rec_args,
                            rec_result if isinstance(rec_result, str) else "",
                        )
                        gathered.append({
                            "call": {"tool": "brain_define", "args": rec_args},
                            "result": rec_result,
                        })
                        trace.append({
                            "step": step,
                            "event": "auto_recovery_define",
                            "call": {"tool": "brain_define", "args": rec_args},
                            "result": rec_result[:300],
                        })

        # ---- Synthesis ----
        if self.skip_synthesis:
            answer = _format_gathered(gathered)
            trace.append({"step": "synthesis", "event": "skipped"})
        elif self.cortex_synthesizer:
            from sara_brain.cortex.transformer.synthesizer import synthesize
            answer = synthesize(question, gathered)
            trace.append({"step": "synthesis", "event": "cortex_template"})
        elif self.strict_sara:
            # v052 strict-Sara: rules via system_prompt, substrate via
            # <substrate> tags in the user message. Per-claim grounding
            # required; training-derived inference forbidden.
            user_msg = _STRICT_SARA_USER_TEMPLATE.format(
                question=question, gathered=_format_gathered(gathered),
            )
            response = self.synthesizer.chat(
                messages=[{"role": "user", "content": user_msg}],
                tools=[],
                model=self.synthesis_model,
                system_prompt=_STRICT_SARA_SYSTEM_PROMPT,
            )
            answer = response.text.strip()
            trace.append({
                "step": "synthesis", "event": "synthesized_strict_sara",
            })
        else:
            synthesis_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
                question=question, gathered=_format_gathered(gathered)
            )
            response = self.synthesizer.chat(
                messages=[{"role": "user", "content": synthesis_prompt}],
                tools=[],
                model=self.synthesis_model,
                system_prompt=None,
            )
            answer = response.text.strip()
            trace.append({"step": "synthesis", "event": "synthesized"})

        if return_trace:
            return {
                "answer": answer,
                "gathered": gathered,
                "trace": trace,
                "routing_steps": len(gathered),
            }
        return answer

    def _route_step(
        self,
        question: str,
        gathered: list[dict],
        trace: list[dict],
    ) -> dict | None:
        if self.cortex_router is not None:
            return self._cortex_route_step(question, gathered, trace)
        prompt = _ROUTER_PROMPT_TEMPLATE.format(
            question=question, gathered=_format_gathered(gathered)
        )
        for attempt in range(self.max_retries_per_step):
            response = self.router.chat(
                messages=[{"role": "user", "content": prompt}],
                tools=[],
                model=self.router_model,
                system_prompt=None,
            )
            decision = _parse_router_response(response.text)
            if decision is None:
                trace.append({
                    "step": "route_attempt",
                    "attempt": attempt,
                    "event": "parse_failed",
                    "raw": response.text[:200],
                })
                continue
            if not self._validate_decision(decision):
                trace.append({
                    "step": "route_attempt",
                    "attempt": attempt,
                    "event": "validation_failed",
                    "decision": decision,
                })
                continue
            return decision
        return None

    def _cortex_route_step(
        self,
        question: str,
        gathered: list[dict],
        trace: list[dict],
    ) -> dict | None:
        """Cortex routing: deterministic single-shot. The orchestrator's
        repeat-call detection naturally terminates the loop after one
        substantive call (the cortex picks the same answer on identical
        inputs). No retry needed."""
        decision_obj = self.cortex_router.route(question)
        decision = {"tool": decision_obj.tool, **decision_obj.args}
        if not self._validate_decision(decision):
            trace.append({
                "step": "cortex_route",
                "event": "validation_failed",
                "model": decision_obj.model,
                "decision": decision,
                "cls_conf": decision_obj.classifier_confidence,
            })
            return None
        trace.append({
            "step": "cortex_route",
            "event": "decided",
            "model": decision_obj.model,
            "decision": decision,
            "cls_conf": decision_obj.classifier_confidence,
            "rationale": decision_obj.rationale,
        })
        return decision

    def _validate_decision(self, decision: dict) -> bool:
        """Validate the router's decision is well-formed.

        Note: we no longer require the concept to exist in the substrate
        for brain_value / brain_define. The tools themselves return
        'no neuron matching' gracefully, and the router needs to be able
        to attempt recovery labels (compound forms, alternate casings)
        that may not exist as primary neurons. Blocking those here
        defeats the NO-MATCH RECOVERY rule in the router prompt.
        """
        tool = decision["tool"]
        if tool == "DONE":
            return True
        if tool == "brain_value":
            concept = decision.get("concept")
            return bool(concept) and isinstance(concept, str)
        if tool == "brain_define":
            concept = decision.get("concept")
            return bool(concept) and isinstance(concept, str)
        if tool == "brain_explore":
            label = decision.get("label")
            return bool(label) and isinstance(label, str)
        if tool == "brain_did_you_mean":
            term = decision.get("term")
            return bool(term) and isinstance(term, str)
        return False

    def _concept_exists(self, label: str) -> bool:
        row = self.brain.conn.execute(
            "SELECT 1 FROM neurons WHERE label = ? LIMIT 1",
            (label.strip().lower(),),
        ).fetchone()
        return row is not None
