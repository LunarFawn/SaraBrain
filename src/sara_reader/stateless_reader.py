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


# v052 follow-up: heuristic topic extraction for --explore-first.
# Strips wh-prefixes, articles, and trailing punctuation. The result
# becomes the brain_explore label; brain_explore tolerates rough
# input via its own fuzzy matching upstream.
_WH_PREFIX_RE = re.compile(
    r"^(what is|what are|what does|what do|how does|how do|how can|"
    r"why does|why do|why is|tell me about|describe|explain|define|"
    r"who is|who are|where is|where are)\s+",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _extract_topic(question: str) -> str:
    q = (question or "").strip().lower()
    q = _WH_PREFIX_RE.sub("", q, count=1)
    q = _LEADING_ARTICLE_RE.sub("", q, count=1)
    q = q.rstrip("?.!,;: ")
    return q


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

NO-DEFINITION RECOVERY: If brain_define returned a message containing
"no definitional edges" or "has no definition/identity relation", the
concept EXISTS in the substrate but is described through verbs and
associations rather than 'X is Y' definitional triples. Your next call
MUST be brain_explore(label=<same concept>, depth=1) to retrieve the
verb-form and part_of edges that describe the concept. Do NOT emit
DONE on a no-definition result alone — the brain has content; you
just need a wider tool to surface it. Books, narratives, and prose-
ingested substrates almost never produce definitional triples; their
content lives in brain_explore output.

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


# ---- v053 wavefront helpers (the brain's native query mechanism) ----

# Stopwords for question seed extraction. Conservative: keep anything
# that might be a content concept in the brain; only strip closed-class
# function words and benchmark-style framing words.
_WAVEFRONT_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with",
    "as", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "this", "that", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "whose", "where", "when", "why",
    "how", "tell", "about", "describe", "explain", "define",
    "following", "best", "most", "many", "some", "any", "each", "every",
    "both", "and", "or", "but", "not", "no", "yes", "if", "then", "than",
    "from", "into", "through", "over", "under", "between", "across",
    "after", "before", "during", "while", "since", "until", "because",
    "though", "although", "however", "therefore", "moreover", "thus",
})


def _extract_seed_concepts(question: str) -> list[str]:
    """Pull content-word seeds from the question for wavefront propagation.

    Lowercased; non-alphabetic tokens dropped (numbers and acronyms in
    weird shapes need a different path). Keeps multi-word potential
    seeds by including bigrams of adjacent content words — gives the
    wavefront a chance to seed compound labels like 'directional
    selection' when both 'directional' and 'selection' appear.
    """
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", question)
    lowered = [w.lower() for w in words]
    content = [w for w in lowered if w not in _WAVEFRONT_STOPWORDS and len(w) >= 3]
    # Bigrams from adjacent content positions (after stopword stripping):
    # walk the original token sequence so the bigram preserves order.
    bigrams: list[str] = []
    prev_was_content = False
    prev_word = ""
    for w in lowered:
        if w in _WAVEFRONT_STOPWORDS or len(w) < 3:
            prev_was_content = False
            continue
        if prev_was_content and prev_word:
            bigrams.append(f"{prev_word} {w}")
        prev_word = w
        prev_was_content = True
    # Combine; dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for s in bigrams + content:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _format_wavefront_substrate(brain, seeds: list[str],
                                convergence_map: dict,
                                intersections) -> str:
    """Render the wavefront's output as a substrate fact string.

    Lines:
      Wavefront(seeds=[...]) → {N_intersect} intersections, {N_conv} reached.
      Intersections (multi-wavefront convergence — the recognition result):
        - <label> (strength=...)
      Reached (all neurons touched):
        - <label> (strength=...)
    """
    lines: list[str] = []
    lines.append(
        f"Wavefront from {len(seeds)} seed(s) {seeds!r}: "
        f"{len(intersections)} intersection(s), "
        f"{len(convergence_map)} neuron(s) reached.",
    )

    # Resolve intersection neurons to labels.
    inter_items = (
        intersections.items() if isinstance(intersections, dict)
        else [(t[0], t[1]) for t in intersections]
    )
    inter_resolved: list[tuple[str, float]] = []
    for nid, weight in inter_items:
        n = brain.neuron_repo.get_by_id(nid)
        if n is None:
            continue
        inter_resolved.append((n.label, float(weight)))
    inter_resolved.sort(key=lambda x: -x[1])

    if inter_resolved:
        lines.append("")
        lines.append(
            "Intersections (multi-wavefront convergence — "
            "the recognition result):",
        )
        for label, w in inter_resolved[:25]:
            lines.append(f"  - {label!r} (strength={w:.2f})")

    # Then the broader convergence map for context.
    if convergence_map:
        conv_resolved: list[tuple[str, float]] = []
        for nid, weight in convergence_map.items():
            n = brain.neuron_repo.get_by_id(nid)
            if n is None:
                continue
            conv_resolved.append((n.label, float(weight)))
        conv_resolved.sort(key=lambda x: -x[1])
        lines.append("")
        lines.append(
            f"Reached (full convergence map, top 30 of {len(conv_resolved)}):",
        )
        for label, w in conv_resolved[:30]:
            lines.append(f"  - {label!r} (strength={w:.2f})")

    return "\n".join(lines)


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
        explore_first: bool = False,
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
        # v052 follow-up: when True, every .ask() prepends a
        # brain_explore depth=3 call with a heuristic-extracted
        # topic from the question. Catches the case where the
        # router picks brain_define / brain_value (narrow tools
        # that miss most of the substrate) and the answer comes
        # back thin. Per Pearl 2026a §2.4 the associative
        # neighborhood IS the signal — explore-first guarantees
        # that signal is in the gathered list regardless of what
        # else the router picks.
        self.explore_first = explore_first

    def ask(self, question: str, return_trace: bool = False) -> str | dict:
        gathered: list[dict] = []
        trace: list[dict] = []
        seen_calls: set[tuple] = set()

        # ---- Wavefront-first (v053 — restored as the brain's primary
        # query mechanism per Pearl 2026a / rev8) ----
        # Parallel wavefront propagation IS the brain's defining
        # function. Every question runs wavefront FIRST, before any
        # LLM-driven tool selection. Question content words become
        # propagation seeds; intersections become recognition results.
        # The synthesizer receives the convergence output as the
        # primary substrate. Tools like brain_value / brain_define
        # remain for supplementary drill-downs but never as
        # alternatives to wavefront.
        #
        # Per memory rule `feedback_wavefront_is_the_brain.md`: this is
        # not optional and not behind a flag. Demoting wavefront to one
        # tool option in a flat menu was the v050/v052 architectural
        # error this restoration corrects.
        wavefront_result = self._run_wavefront(question)
        if wavefront_result is not None:
            gathered.append({
                "call": {
                    "tool": "brain_wavefront",
                    "args": {"seeds": wavefront_result["seeds"]},
                },
                "result": wavefront_result["substrate"],
            })
            trace.append({
                "step": "wavefront_first",
                "event": "tool_executed",
                "call": {
                    "tool": "brain_wavefront",
                    "args": {"seeds": wavefront_result["seeds"]},
                },
                "result": wavefront_result["substrate"][:300],
            })

        # ---- Explore-first (v052 follow-up) ----
        # Always start with brain_explore depth=3 on a heuristic-
        # extracted topic. Captures the associative neighborhood per
        # Pearl 2026a §2.4 so the synthesizer has rich substrate to
        # answer from, even when downstream routing picks narrower
        # tools (brain_define / brain_value).
        if self.explore_first:
            topic = _extract_topic(question)
            if topic:
                ef_args = {"label": topic, "depth": 3}
                ef_key = ("brain_explore", tuple(sorted(ef_args.items())))
                seen_calls.add(ef_key)
                try:
                    ef_result = execute_tool(self.brain, "brain_explore", ef_args)
                except Exception as exc:
                    ef_result = f"<<tool error: {exc}>>"
                _audit_tool_call(
                    "brain_explore", ef_args,
                    ef_result if isinstance(ef_result, str) else "",
                )
                gathered.append({
                    "call": {"tool": "brain_explore", "args": ef_args},
                    "result": ef_result,
                })
                trace.append({
                    "step": "explore_first",
                    "event": "tool_executed",
                    "call": {"tool": "brain_explore", "args": ef_args},
                    "result": ef_result[:300],
                })

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

            # ---- Python-side NO-DEFINITION RECOVERY ----
            # Both routers (cortex + small Ollama) call brain_define for
            # "what is X?" questions, but prose-ingested brains (books,
            # narratives, conversational substrates) rarely produce
            # 'X is Y' definitional triples — content lives in verb-form
            # and part_of edges. When brain_define returns "no
            # definitional edges" but the concept exists, automatically
            # call brain_explore(label=concept, depth=1) so the
            # synthesis layer has substantive substrate to ground in.
            if (
                tool_name == "brain_define"
                and isinstance(result, str)
                and result.startswith("No definitional edges")
                and "concept" in args
            ):
                rec_args = {"label": args["concept"], "depth": 1}
                rec_key = ("brain_explore", tuple(sorted(rec_args.items())))
                if rec_key not in seen_calls:
                    seen_calls.add(rec_key)
                    try:
                        rec_result = execute_tool(
                            self.brain, "brain_explore", rec_args
                        )
                    except Exception as exc:
                        rec_result = f"<<tool error: {exc}>>"
                    _audit_tool_call(
                        "brain_explore", rec_args,
                        rec_result if isinstance(rec_result, str) else "",
                    )
                    gathered.append({
                        "call": {"tool": "brain_explore", "args": rec_args},
                        "result": rec_result,
                    })
                    trace.append({
                        "step": step,
                        "event": "auto_recovery_no_definition",
                        "call": {"tool": "brain_explore", "args": rec_args},
                        "result": rec_result[:300] if isinstance(rec_result, str) else "",
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

    def _run_wavefront(self, question: str) -> dict | None:
        """Run the brain's native wavefront query for the question.

        Extracts content-word seeds from the question, propagates each
        in parallel through the substrate (property → relation →
        concept paths), and returns the convergence map as a substrate
        fact the synthesizer can ground in. Read-only — never mutates
        the graph.

        Returns None when no seeds could be extracted. Otherwise:
            {
              "seeds":     ["word1", "word2", ...],
              "substrate": "<rendered convergence map + intersections>",
            }
        """
        seeds = _extract_seed_concepts(question)
        if not seeds:
            return None
        # Wavefront depth: large brains (dictionary + content >100k
        # neurons + ~1M edges) produce too much noise at the
        # Recognizer's default max_depth=3 because the dictionary's
        # synonym fan-out reaches almost everything. Cap at 2 for the
        # chat query path — keeps direct + 1-hop bridges intact while
        # avoiding the second-order synonym flood. Override with
        # SARA_WAVEFRONT_DEPTH env var if needed.
        target_depth = int(os.environ.get("SARA_WAVEFRONT_DEPTH", "2"))
        original_depth = self.brain.recognizer.max_depth
        try:
            self.brain.recognizer.max_depth = target_depth
            with self.brain.short_term(event_type="ask_wavefront") as st:
                self.brain.propagate_into(
                    seeds, st, exact_only=True,
                )
                convergence_map = dict(st.convergence_map)
                intersections = st.intersections(min_sources=2)
        except Exception as exc:
            return {
                "seeds": seeds,
                "substrate": f"<<wavefront error: {exc}>>",
            }
        finally:
            self.brain.recognizer.max_depth = original_depth
        substrate = _format_wavefront_substrate(
            self.brain, seeds, convergence_map, intersections,
        )
        return {"seeds": seeds, "substrate": substrate}

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
