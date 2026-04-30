"""SaraReader — public API for the sara_reader SDK.

The LLM is the orchestrator. SaraReader sets up retrieval tools, sends
the user's question to the model, executes any tool calls the model
makes against the loaded brain, and loops until the model produces a
final answer.
"""
from __future__ import annotations

import re
from typing import Any

from sara_brain.core.brain import Brain

from .brain_loader import load_brain
from .providers import get_provider, ToolCall
from .tools import execute_tool, get_tool_schemas


_ACRONYM_EXPANSION_PATTERN = re.compile(r"\b([A-Z]{2,})\b\s*\(([^)]+)\)")


_DEFAULT_SYSTEM_PROMPT = (
    "You are answering using Sara Brain, a path-of-thought knowledge graph "
    "the user has taught with specific facts and framings. Ground your "
    "answer in what brain_explore returns. If Sara has no relevant "
    "content, say so explicitly rather than inventing.\n\n"
    "RETRIEVAL PROTOCOL — stepped depth, never start wide:\n"
    "  1. ALWAYS start with brain_explore(label, depth=1) on the most "
    "specific term in the question. depth=1 returns the immediate "
    "neighborhood — usually enough for value/range/property questions.\n"
    "  2. If depth=1 answers the question, ANSWER. Do not escalate.\n"
    "  3. If depth=1 returned nothing useful but suggested neighboring "
    "terms, retry depth=1 on a more specific or alternate label before "
    "escalating depth.\n"
    "  4. Only escalate to depth=2 if depth=1 was insufficient and you "
    "have a clear reason the next hop will help.\n"
    "  5. Only escalate to depth=3 for genuinely conceptual questions "
    "('how does X relate to Y?', 'tell me about X').\n"
    "  6. depth=4 is for broad orientation only. If the result is "
    "TRUNCATED or has hundreds of edges, you are over-querying — back "
    "off, pick a more specific label, and try again at lower depth.\n\n"
    "STOP CONDITIONS:\n"
    "  - You have the answer → answer and stop.\n"
    "  - The label does not exist in Sara → try one alternate seed; if "
    "still nothing, say Sara has no content on this and stop.\n"
    "  - Output is too large to make sense of → say so explicitly and "
    "stop. Do not pivot to generic NLP-style commentary.\n"
    "  - The question imports an assumption you cannot verify in the "
    "graph (e.g. 'which X needs Y to do well' when the graph stores "
    "'X has Y' but not 'X needs Y') → DO NOT invent the bridge. "
    "Surface what you found and what you could not, and tell the user "
    "the question may be framed in a way Sara cannot cleanly answer.\n\n"
    "HONEST-PUSHBACK TEMPLATE — use this whenever the substrate is "
    "thinner than the question demands, the data is broad, or the "
    "question's framing doesn't map onto the taught relations:\n"
    "  'This question is broader than what Sara has cleanly taught. "
    "Here is the closest match I found: [cite the actual edges]. "
    "I cannot confirm [the part of the question that imports an "
    "untaught bridge]. You may want to rephrase as [a more specific "
    "question that maps onto what's there], or ask whether [the "
    "missing concept] is in the substrate.'\n"
    "Do this instead of guessing, instead of NLP-meta commentary, and "
    "instead of confidently picking the closest-looking node.\n\n"
    "DEFINITION-CHECK PROTOCOL (mandatory before composing the "
    "final answer):\n"
    "  Before mentioning any acronym, domain term, or named concept "
    "in your final answer, call brain_define(concept) on it. If the "
    "tool returns 'no neuron matching' or 'no definitional edges', "
    "DO NOT invent an expansion or definition from training — either "
    "omit the term or quote only the value/range you retrieved. "
    "This prevents confabulation like 'KDON (kill-dead-on-demand)' "
    "when the substrate's actual definition is 'aptamer affinity to "
    "on state measures kdon'.\n"
    "  This applies even to terms that sound familiar from training. "
    "If you write the term in your answer, the substrate must have "
    "defined it — or you must not define it.\n\n"
    "Tool selection:\n"
    "  - brain_define(concept) — MANDATORY before mentioning any "
    "term. Returns the substrate's definition only.\n"
    "  - brain_value(concept, type) — preferred for any specific "
    "number, range, ratio, or quantitative property of a known "
    "concept. Returns ONLY the value edges. Cite verbatim.\n"
    "    `concept` = the thing being asked about (e.g. "
    "'super-performing mode', 'ssng1 highest kdoff').\n"
    "    `type` = the kind of value being requested (e.g. 'kdoff', "
    "'kdon', 'ratio', 'value'). PASS `type` whenever the question "
    "names a specific quantity — it forces one quantity per call "
    "and prevents merging two relations into one threshold.\n"
    "    COMPOUND CONCEPT RULE: when the question names a broad "
    "concept AND a specific quantity, prefer the compound concept "
    "label.\n"
    "    Examples:\n"
    "      Q: 'highest KDOFF for SSNG1?' → "
    "brain_value(concept='ssng1 highest kdoff', type='value')\n"
    "      Q: 'KDOFF range for super-performing mode?' → "
    "brain_value(concept='super-performing mode', type='kdoff')\n"
    "      Q: 'KDON for super-performing mode?' → "
    "brain_value(concept='super-performing mode', type='kdon')\n"
    "      Q: 'all properties of super-performing mode?' → "
    "brain_value(concept='super-performing mode')   (no type)\n"
    "    If the compound concept returns 'no neuron matching', fall "
    "back to brain_value on the broad concept (still with type), "
    "before declaring no answer.\n"
    "  - brain_explore(label, depth) — for conceptual / 'what is X?' "
    "retrieval and for finding what concepts neighbor a term\n"
    "  - brain_why / brain_trace — only when direction matters\n"
    "  - brain_recognize — when identifying a concept from properties\n"
    "  - brain_did_you_mean — fuzzy match if the term seems off\n"
)


class SaraReader:
    """Provider-agnostic Sara consumer.

    Example:
        >>> reader = SaraReader(
        ...     brain_path="/path/to/aptamer_full.db",
        ...     provider="anthropic",
        ...     model="claude-haiku-4-5",
        ... )
        >>> reader.ask("what is the molecular snare?")
    """

    def __init__(
        self,
        brain_path: str,
        provider: str,
        model: str,
        system_prompt: str | None = None,
        max_rounds: int = 8,
        provider_kwargs: dict | None = None,
    ) -> None:
        """Construct a reader.

        Args:
            brain_path: filesystem path to a Sara .db file.
            provider: "anthropic" or "ollama". "openai" is NOT supported
                — see package README for the policy.
            model: model identifier specific to the provider, e.g.
                "claude-haiku-4-5" or "llama3.2:3b".
            system_prompt: override the default retrieval-encouraging
                system prompt.
            max_rounds: maximum tool-call rounds before giving up.
            provider_kwargs: extra args passed to the provider
                constructor (e.g., api_key for anthropic).
        """
        self.brain: Brain = load_brain(brain_path)
        self.provider = get_provider(provider, **(provider_kwargs or {}))
        self.model = model
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self.max_rounds = max_rounds
        self._tool_schemas = get_tool_schemas()

    def ask(self, question: str, return_trace: bool = False) -> str | dict:
        """Ask Sara a question. The LLM decides how to retrieve.

        Args:
            question: the user's question.
            return_trace: if True, return a dict with the answer and
                the full conversation/tool-call trace for inspection.

        Returns:
            The model's final text answer (str), or a dict with answer
            and trace if return_trace=True.
        """
        messages: list[dict] = [{"role": "user", "content": question}]
        trace: list[dict] = []

        for round_idx in range(self.max_rounds):
            response = self.provider.chat(
                messages=messages,
                tools=self._tool_schemas,
                model=self.model,
                system_prompt=self.system_prompt,
            )
            # Record the assistant's turn in messages
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.text,
            }
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)
            trace.append({"round": round_idx, "assistant": assistant_msg})

            if not response.tool_calls:
                # Final answer — fact-check before returning
                checked, challenges = self._fact_check_answer(response.text)
                if return_trace:
                    return {
                        "answer": checked,
                        "raw_answer": response.text,
                        "challenges": challenges,
                        "trace": trace,
                        "rounds": round_idx + 1,
                    }
                return checked

            # Execute each tool call against the brain
            for tc in response.tool_calls:
                result = execute_tool(self.brain, tc.name, tc.arguments)
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
                messages.append(tool_msg)
                trace.append({
                    "round": round_idx,
                    "tool_call": {"name": tc.name, "arguments": tc.arguments},
                    "tool_result": result[:500] + ("..." if len(result) > 500 else ""),
                })

        # Hit max_rounds without a final answer
        final = "<<MAX_ROUNDS_EXCEEDED — model kept requesting tools>>"
        if return_trace:
            return {"answer": final, "trace": trace, "rounds": self.max_rounds}
        return final

    def _fact_check_answer(self, answer: str) -> tuple[str, list[dict]]:
        """Substrate-validate the model's final answer.

        Checks acronym-expansion patterns: ``ACRONYM (expansion)``. For
        each match, runs brain_define on the acronym. If the
        expansion's tokens don't appear in the substrate's definition,
        the parenthetical is replaced with the substrate definition
        (when available) or stripped (when the concept isn't in the
        substrate at all). A substrate-check block is appended listing
        what was changed and why.

        Returns ``(formatted_answer, challenges)``.
        """
        challenges: list[dict] = []

        def replace(match: re.Match) -> str:
            acronym = match.group(1)
            expansion = match.group(2).strip()
            verdict = self._check_acronym_expansion(acronym, expansion)
            if verdict is None:
                return match.group(0)
            challenges.append({
                "acronym": acronym,
                "claimed_expansion": expansion,
                "verdict": verdict["kind"],
                "substrate_definition": verdict["substrate_definition"],
            })
            if verdict["substrate_definition"]:
                return f"{acronym} ({verdict['substrate_definition']})"
            return acronym  # nothing in substrate to substitute

        cleaned = _ACRONYM_EXPANSION_PATTERN.sub(replace, answer)
        cleaned = re.sub(r" +", " ", cleaned).strip()

        if not challenges:
            return cleaned, challenges

        block_lines = ["", "─── Substrate check ───"]
        for c in challenges:
            if c["substrate_definition"]:
                block_lines.append(
                    f"⚠ {c['acronym']}: model originally wrote "
                    f"\"{c['claimed_expansion']}\"; substrate defines "
                    f"{c['acronym']} as \"{c['substrate_definition']}\". "
                    f"Replaced with substrate definition."
                )
            else:
                block_lines.append(
                    f"⚠ {c['acronym']}: model originally wrote "
                    f"\"{c['claimed_expansion']}\"; no substrate "
                    f"definition exists for {c['acronym']}. Expansion "
                    f"was training-driven and removed."
                )

        return cleaned + "\n" + "\n".join(block_lines), challenges

    def _check_acronym_expansion(
        self, acronym: str, expansion: str
    ) -> dict | None:
        """Look up acronym in brain_define and judge the expansion.

        Returns None if the expansion is acceptable. Returns a verdict
        dict if the expansion contradicts the substrate.
        """
        from .tools import TOOLS

        result = TOOLS["brain_define"]["executor"](
            self.brain, {"concept": acronym}
        )

        if result.startswith("Sara has no neuron"):
            return {
                "kind": "acronym not in substrate",
                "substrate_definition": None,
            }
        if result.startswith("No definitional edges"):
            return None  # concept exists but has no definition; can't judge

        substrate_text = result.lower()
        expansion_tokens = [
            t for t in re.split(r"\W+", expansion.lower()) if len(t) >= 3
        ]
        if expansion_tokens and any(
            t in substrate_text for t in expansion_tokens
        ):
            return None  # at least one token matches; accept

        substrate_def = self._extract_definitional_phrase(result, acronym)
        return {
            "kind": "expansion contradicts substrate",
            "substrate_definition": substrate_def,
        }

    @staticmethod
    def _extract_definitional_phrase(
        brain_define_output: str, acronym: str
    ) -> str | None:
        """Pull the human-readable definition phrase from a brain_define
        result.

        Each non-header line looks like:
          'aptamer affinity to on state' --[measures]--> 'kdon'
        We pick the side that is NOT the acronym itself.
        """
        edge_pattern = re.compile(
            r"'([^']+)'\s*--\[([^\]]+)\]-->\s*'([^']+)'"
        )
        target = acronym.lower()
        for line in brain_define_output.splitlines():
            m = edge_pattern.search(line)
            if not m:
                continue
            src, _rel, tgt = m.group(1), m.group(2), m.group(3)
            if src.lower() == target:
                return tgt
            if tgt.lower() == target:
                return src
        return None
