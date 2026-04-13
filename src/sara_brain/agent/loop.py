"""Agent loop — the core ReAct loop with Sara Brain validation.

Flow per turn:
1. User speaks (through LLM)
2. LLM proposes action (tool calls or text)
3. Sara validates proposals against known paths
4. If conflict → Sara corrects → LLM adjusts → repeat
5. If approved → execute → Sara observes outcomes
6. LLM responds to user
"""

from __future__ import annotations

import json
import sys

from ..core.brain import Brain
from . import ollama
from .bridge import AgentBridge
from .sandbox import Sandbox
from .session import SessionStore
from .system import build_system_prompt
from .tools import dispatch, get_tool_definitions
from .validator import Validator


class AgentLoop:
    """Llama (cortex) + Sara Brain (cerebellum) agent."""

    def __init__(
        self,
        brain: Brain,
        model: str,
        base_url: str = "http://localhost:11434",
        max_tool_rounds: int = 10,
        max_validations: int = 3,
        sandbox_timeout: int = 30,
        cwd: str | None = None,
    ) -> None:
        self.brain = brain
        self.bridge = AgentBridge(brain)
        self.validator = Validator(self.bridge)
        self.sandbox = Sandbox(timeout=sandbox_timeout, cwd=cwd)
        self.session_store = SessionStore()
        self.model = model
        self.base_url = base_url
        self.max_tool_rounds = max_tool_rounds
        self.max_validations = max_validations
        self.cwd = cwd or str(__import__("pathlib").Path.cwd())
        self.messages: list[dict] = []
        self.tools = get_tool_definitions()
        # Recently-refuted claims that the LLM must DISREGARD on the next
        # turn. Prevents context poisoning where the model anchors on its
        # own previous wrong answers from chat history.
        self._recent_refutations: list[str] = []
        self.session_id: str | None = None

    def turn(self, user_input: str) -> str:
        """Process one user turn through the cerebellum loop.

        Returns the LLM's final text response.

        Sara is consulted FIRST, before the LLM. The cortex (LLM) is reduced
        to language I/O — perception in, words out. All knowledge operations
        (recall, teaching, refutation) flow through Sara automatically. The
        model never decides whether to consult the brain; the brain is
        always in the loop.
        """
        # ── SLASH COMMANDS: direct user control of Sara, no LLM involved ──
        # Lets the user teach, refute, query, and inspect the brain without
        # any model in the loop. The user has total authority over knowledge.
        if user_input.strip().startswith("/"):
            slash_response = self._handle_slash(user_input.strip())
            if slash_response is not None:
                self.messages.append({"role": "user", "content": user_input})
                self.messages.append(
                    {"role": "assistant", "content": slash_response}
                )
                self._save_session()
                return slash_response

        self.messages.append({"role": "user", "content": user_input})

        # ── SHORT-CIRCUIT: if Sara has no relevant knowledge AND the user
        # is asking about a specific concept, the brain answers directly
        # without invoking the LLM. This forecloses hallucination at the
        # structural level — there is no model call to fabricate a response.
        short_circuit = self._sara_short_circuit(user_input)
        if short_circuit is not None:
            self.messages.append({"role": "assistant", "content": short_circuit})
            self._save_session()
            return short_circuit

        # ── SARA TURN: brain runs first, before the LLM gets the input ──
        sara_notes = self._sara_turn(user_input)

        system_prompt = build_system_prompt(self.bridge, self.cwd, user_input)
        if sara_notes:
            system_prompt += f"\n\n## Sara's Pre-Turn Operations\n{sara_notes}"

        # ── ANTI-POISONING: inject refuted claims as DISREGARD instructions
        # so the model can't anchor on its own previous wrong answers from
        # chat history. Once the model has been told something is refuted,
        # we keep telling it for the next 3 turns to make sure it sticks.
        if self._recent_refutations:
            disregard = "\n".join(f"  - {r}" for r in self._recent_refutations)
            system_prompt += (
                f"\n\n## CRITICAL: DISREGARD THESE REFUTED CLAIMS\n"
                f"The following claims have been REFUTED by the user. They are "
                f"KNOWN-TO-BE-FALSE and stored in Sara's brain with negative "
                f"strength. You MUST NOT repeat them in your response. If they "
                f"appear in the chat history above, treat them as poisoned "
                f"context and ignore. Use ONLY Sara's currently-grounded paths.\n"
                f"{disregard}"
            )

        system_msg = {"role": "system", "content": system_prompt}

        for _round in range(self.max_tool_rounds):
            # Call Ollama with full history + tools
            response = ollama.chat(
                base_url=self.base_url,
                model=self.model,
                messages=[system_msg] + self.messages,
                tools=self.tools,
            )

            result = ollama.extract_response(response)
            content = result["content"]
            tool_calls = result["tool_calls"]

            # If the LLM responds with text and no tool calls → done
            if content and not tool_calls:
                self.messages.append(
                    {"role": "assistant", "content": content}
                )
                # Post-turn: Sara learns from the LLM's response
                self._post_turn_observe(content)
                self._save_session()
                # Append a compact provenance summary so the user can see
                # at a glance how grounded the response is.
                summary = self._provenance_summary(content)
                if summary:
                    return f"{content}\n\n{summary}"
                return content

            # If tool calls were found (structured or parsed from text)
            if tool_calls:
                # Check if these were parsed from text (small model fallback)
                is_text_parsed = any(
                    tc.get("id", "").startswith("text_parsed_")
                    for tc in tool_calls
                )

                if is_text_parsed:
                    # For text-parsed tool calls, execute and feed results
                    # back as a user message (model doesn't expect tool role)
                    if content:
                        self.messages.append(
                            {"role": "assistant", "content": content}
                        )
                    results = []
                    for tc in tool_calls:
                        name = tc.get("function", {}).get("name", "")
                        tool_result = self._validate_and_execute(tc)
                        results.append(f"[Tool: {name}]\n{tool_result}")
                    combined = "\n\n".join(results)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"Tool results:\n\n{combined}\n\nNow respond to the user based on these results. Be concise.",
                        }
                    )
                else:
                    # Structured tool calls — standard flow
                    assistant_msg: dict = {"role": "assistant"}
                    if content:
                        assistant_msg["content"] = content
                    assistant_msg["tool_calls"] = tool_calls
                    self.messages.append(assistant_msg)

                    for tc in tool_calls:
                        tool_result = self._validate_and_execute(tc)
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "content": tool_result,
                            }
                        )
                continue  # LLM sees tool results on next iteration

            # Edge case: no content and no tool calls
            fallback = "I'm not sure how to proceed. Could you rephrase?"
            self.messages.append(
                {"role": "assistant", "content": fallback}
            )
            self._save_session()
            return fallback

        # Max rounds reached — ask LLM to wrap up
        self.messages.append(
            {
                "role": "user",
                "content": "(System: max tool rounds reached. Please provide your final response.)",
            }
        )
        response = ollama.chat(
            base_url=self.base_url,
            model=self.model,
            messages=[system_msg] + self.messages,
        )
        result = ollama.extract_response(response)
        final = result["content"] or "Reached maximum rounds."
        self.messages.append({"role": "assistant", "content": final})
        self._post_turn_observe(final)
        self._save_session()
        return final

    def _handle_slash(self, command: str) -> str | None:
        """Handle slash commands. Returns the response string or None if not a recognized command.

        Available commands:
            /teach <statement>     — Commit a fact to Sara
            /refute <statement>    — Mark a fact as known-to-be-false
            /forget <statement>    — Alias for /refute
            /know <topic>          — Show what Sara knows about a topic (raw paths)
            /why <topic>           — Show paths leading to a concept
            /trace <topic>         — Show paths going out from a concept
            /last                  — Show Sara's grounding analysis of the LLM's last response
            /stats                 — Brain statistics
            /help                  — List commands
        """
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/help", "/?"):
            return (
                "Slash commands (bypass the LLM, talk to Sara directly):\n"
                "  /teach <fact>     — commit a fact to Sara's brain\n"
                "  /refute <fact>    — mark a fact as known-to-be-false\n"
                "  /forget <fact>    — alias for /refute\n"
                "  /know <topic>     — show paths leading to and from a topic\n"
                "  /why <topic>      — show paths leading TO a topic\n"
                "  /trace <topic>    — show paths going FROM a topic\n"
                "  /last             — analyze Sara's grounding of the LLM's last response\n"
                "  /stats            — brain statistics\n"
                "  /help             — this message"
            )

        if cmd == "/teach":
            if not arg:
                return "Usage: /teach <fact>  (e.g., /teach the edubba was a sumerian school)"
            return self.bridge.teach(arg)

        if cmd in ("/refute", "/forget"):
            if not arg:
                return f"Usage: {cmd} <fact>  (e.g., {cmd} the earth is flat)"
            return self.bridge.refute(arg)

        if cmd == "/know":
            if not arg:
                return "Usage: /know <topic>"
            return self.bridge.query(arg)

        if cmd == "/why":
            if not arg:
                return "Usage: /why <topic>"
            traces = self.bridge.brain.why(arg)
            if not traces:
                return f"No paths lead to '{arg}'."
            lines = [f"Paths to '{arg}':"]
            for t in traces:
                marker = " [REFUTED]" if t.is_refuted else ""
                src = f' (from: "{t.source_text}")' if t.source_text else ""
                lines.append(f"  {t} weight={t.weight:+.2f}{marker}{src}")
            return "\n".join(lines)

        if cmd == "/trace":
            if not arg:
                return "Usage: /trace <topic>"
            traces = self.bridge.brain.trace(arg)
            if not traces:
                return f"No outgoing paths from '{arg}'."
            lines = [f"Paths from '{arg}':"]
            for t in traces[:30]:
                marker = " [REFUTED]" if t.is_refuted else ""
                lines.append(f"  {t} weight={t.weight:+.2f}{marker}")
            return "\n".join(lines)

        if cmd == "/stats":
            return self.bridge.stats()

        if cmd == "/last":
            return self._analyze_last_response()

        # Not a recognized slash command — let it fall through to normal processing
        return None

    def _provenance_summary(self, response: str) -> str:
        """Compact one-line provenance summary appended to LLM responses.

        Examples:
            [Sara: 4/4 sentences grounded — fully sourced]
            [Sara: 2/4 sentences grounded — use /last for details, /refute to fix]
            [Sara: 0/4 sentences grounded — model is hallucinating]
        """
        sentences = self._split_sentences(response)
        if not sentences:
            return ""
        grounded = 0
        for sent in sentences:
            tag = self._sentence_grounding_tag(sent)
            if tag.startswith("[grounded"):
                grounded += 1
        total = len(sentences)
        if grounded == total:
            label = "fully sourced"
        elif grounded == 0:
            label = "model is generating without Sara — possible hallucination"
        elif grounded >= total / 2:
            label = "mostly grounded — use /last for details"
        else:
            label = "mostly ungrounded — use /last to inspect, /refute to fix"
        return f"[Sara: {grounded}/{total} sentences grounded — {label}]"

    def _analyze_last_response(self) -> str:
        """Show Sara's grounding analysis of the most recent assistant message.

        Walks the recent message history backward to find the last assistant
        response, then breaks it into sentences and reports which are
        grounded in Sara's brain vs which appear to be model invention.
        """
        last_assistant = None
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = msg["content"]
                break
        if not last_assistant:
            return "No previous assistant response to analyze."

        sentences = self._split_sentences(last_assistant)
        if not sentences:
            return "No declarative sentences found in the last response."

        lines = ["Grounding analysis of the LLM's last response:"]
        lines.append("")
        grounded = 0
        ungrounded = 0
        for sent in sentences:
            tag = self._sentence_grounding_tag(sent)
            lines.append(f"  {tag} {sent}")
            if tag.startswith("[grounded"):
                grounded += 1
            else:
                ungrounded += 1
        lines.append("")
        lines.append(
            f"Summary: {grounded} grounded, {ungrounded} ungrounded out of "
            f"{grounded + ungrounded} sentences."
        )
        if ungrounded > 0:
            lines.append(
                "Use /refute <fact> to mark any wrong claims as known-to-be-false."
            )
        return "\n".join(lines)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into rough sentences for grounding analysis."""
        import re
        text = re.sub(r"\*\*|`|#+\s*", "", text)
        sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sents if len(s.strip()) > 10]

    def _sentence_grounding_tag(self, sentence: str) -> str:
        """Return a [grounded] or [ungrounded] tag for a sentence.

        BOTH checks must pass for a sentence to be marked grounded:

        1. WORD CHECK — content words appear in Sara's neuron labels
        2. PATH CHECK — the specific subject→object claim extracted from
           the sentence exists as a path in Sara's graph

        If either fails, the sentence is ungrounded. No partial credit.
        A sentence that uses Sara's vocabulary to dress up an invented
        claim will fail the path check and be correctly marked ungrounded.
        """
        import re

        # ── Word check (existing) ──
        words = re.findall(r"[a-z][a-z']+", sentence.lower())
        skip = {
            "the", "and", "are", "was", "were", "for", "with", "from", "this",
            "that", "have", "had", "has", "been", "being", "their", "them",
            "they", "these", "those", "what", "which", "who", "whose", "would",
            "could", "should", "will", "your", "you", "his", "her", "its",
            "into", "about", "also", "such", "one", "two", "many", "some",
        }
        candidates = [w for w in words if len(w) > 3 and w not in skip]
        if not candidates:
            return "[ungrounded]"

        word_hits = 0
        for word in candidates:
            try:
                if self.bridge.brain.neuron_repo.resolve(word) is not None:
                    word_hits += 1
            except Exception:
                pass

        word_ratio = word_hits / max(1, len(candidates))
        word_pass = word_ratio >= 0.5

        if not word_pass:
            return "[ungrounded]    "

        # ── Path check (new) ──
        # Parse the sentence to extract subject and object, then check
        # if a connecting path exists in Sara's graph.
        path_pass = False
        try:
            from ..cortex.parser import EnhancedParser
            cortex_parser = getattr(self, "_cortex_parser", None)
            if cortex_parser is None:
                cortex_parser = EnhancedParser(self.bridge.brain.taxonomy)
                self._cortex_parser = cortex_parser

            parsed = cortex_parser.parse(sentence)
            if parsed and parsed.facts:
                for fact in parsed.facts:
                    # Resolve subject and object in Sara's brain
                    subj = self.bridge.brain.neuron_repo.resolve(
                        fact.subject.strip().lower()
                    )
                    obj = self.bridge.brain.neuron_repo.resolve(
                        fact.obj.strip().lower()
                    )
                    if subj is None or obj is None:
                        continue

                    # Check if a path connects them (either direction)
                    paths_to_subj = self.bridge.brain.path_repo.get_paths_to(subj.id)
                    for p in paths_to_subj:
                        if p.origin_id == obj.id:
                            path_pass = True
                            break

                    if not path_pass:
                        paths_to_obj = self.bridge.brain.path_repo.get_paths_to(obj.id)
                        for p in paths_to_obj:
                            if p.origin_id == subj.id:
                                path_pass = True
                                break

                    if path_pass:
                        break
        except Exception:
            # If parsing fails, fall through — path check doesn't pass
            pass

        # Both must pass
        if word_pass and path_pass:
            return f"[grounded {word_hits}/{len(candidates)}]"
        elif word_pass and not path_pass:
            return "[ungrounded]    "
        else:
            return "[ungrounded]    "

    def _sara_short_circuit(self, user_input: str) -> str | None:
        """Decide if Sara can answer directly without involving the LLM.

        Returns a string response if the question is about a specific
        concept Sara has zero knowledge of (and no close fuzzy matches).
        Returns None if the LLM should handle this turn normally.

        This is the structural anti-hallucination mechanism: when the
        brain has nothing relevant, the brain says so. The cortex never
        gets a chance to fabricate.
        """
        text = user_input.strip()
        if not text:
            return None

        text_lower = text.lower()

        # Only short-circuit on questions. Declarative statements should
        # flow through normally so the auto-teach can fire on them.
        is_question = (
            text_lower.endswith("?")
            or any(text_lower.startswith(w) for w in (
                "what ", "what's ", "whats ", "who ", "whos ", "who's ",
                "where ", "when ", "why ", "how ", "tell me ", "show me ",
                "define ", "describe ", "explain ",
            ))
        )
        if not is_question:
            return None

        # Extract candidate topic words from the question.
        # Skip question words, articles, and common verbs.
        skip = {
            "what", "whats", "what's", "who", "whos", "who's", "where",
            "when", "why", "how", "is", "are", "was", "were", "the",
            "a", "an", "of", "in", "on", "at", "to", "for", "with",
            "and", "or", "but", "do", "does", "did", "you", "your",
            "tell", "me", "show", "define", "describe", "explain",
            "about", "this", "that", "these", "those", "can", "could",
            "would", "should", "have", "has", "had", "be", "been", "being",
        }
        words = [
            w.strip(".,;:!?\"'()-")
            for w in text_lower.split()
        ]
        # Topic candidates: words longer than 3 chars that aren't stopwords
        candidates = [w for w in words if len(w) > 3 and w not in skip]
        if not candidates:
            return None

        # Check each candidate for direct resolve, fuzzy match, OR
        # presence as a substring in any neuron label
        unresolved = []
        for word in candidates:
            try:
                if self.bridge.brain.neuron_repo.resolve(word) is not None:
                    # Sara knows it directly or via inflection/fuzzy.
                    return None  # let the LLM answer
                # Check did_you_mean for nearby matches
                cands = self.bridge.brain.did_you_mean(word)
                if cands:
                    return None  # Sara has something close — let LLM use it
                unresolved.append(word)
            except Exception:
                return None  # any error → fail open, let LLM try

        # If at least one candidate looks like a "specific topic" (not
        # in our generic skip list and longer than 4 chars), and ALL
        # candidates are unresolved, Sara has nothing to offer.
        specific = [w for w in unresolved if len(w) > 4]
        if not specific:
            return None

        # Short-circuit: tell the user honestly that Sara doesn't know.
        topic_list = ", ".join(f"'{w}'" for w in specific[:3])
        return (
            f"Sara has no knowledge of {topic_list}. "
            f"The language model alone is not trustworthy on topics outside "
            f"the brain — it will hallucinate. "
            f"You can teach Sara directly with: /teach <fact>  "
            f"(for example: '/teach the edubba was a sumerian school'). "
            f"Or paste a document/URL with /ingest <source>."
        )

    # Phrases that indicate the user is correcting a previous claim.
    # Sorted by length descending so longer prefixes match first.
    # The natural English ways someone says "you got that wrong."
    _CORRECTION_PREFIXES = tuple(sorted([
        # Direct refusals
        "no, ", "no ", "nope, ", "nope ",
        # That/this is wrong
        "that's wrong", "thats wrong", "this is wrong", "that is wrong",
        "this was wrong", "that was wrong",
        "that's incorrect", "thats incorrect", "this is incorrect",
        "that is incorrect", "this was incorrect", "that was incorrect",
        "that's not right", "thats not right", "this is not right",
        "that is not right", "that's not true", "thats not true",
        "this is not true", "that is not true",
        # You-statements
        "you're wrong", "youre wrong", "you are wrong",
        "you're incorrect", "youre incorrect", "you are incorrect",
        "you said", "you keep saying", "stop saying",
        "why do you keep", "why are you saying", "why do you say",
        "you got that wrong", "you got it wrong", "you have it wrong",
        # Actually patterns
        "actually,", "actually ", "in fact,", "in fact ",
        # Let me / to be patterns
        "let me correct", "let me clarify", "let me be clear",
        "to correct,", "to clarify,", "to be clear,", "to be precise,",
        "correction:", "correction,", "clarification:",
        # Imperative fixes
        "this needs to be fixed", "that needs to be fixed",
        "you need to fix", "fix this", "fix that",
        # Disagreement markers
        "not true", "not correct", "not accurate",
        "thats false", "that's false", "that is false",
        "false,", "wrong,", "incorrect,",
    ], key=len, reverse=True))

    def _detect_correction(self, user_input: str) -> str | None:
        """If the user is correcting a previous claim, extract the corrected
        statement so it can be auto-taught.

        Strips nested correction prefixes. Looks past commas if the
        leading fragment doesn't parse cleanly. Returns "" (empty string)
        for refute-only corrections like "this is wrong" where the user
        wants to refute the previous answer without providing a new one.
        Returns None if not a correction at all.
        """
        cleaned = user_input.strip()
        found_one = False

        # Strip up to 3 nested correction prefixes
        for _ in range(3):
            text_lower = cleaned.lower()
            stripped_this_round = False
            for prefix in self._CORRECTION_PREFIXES:
                if text_lower.startswith(prefix):
                    cleaned = cleaned[len(prefix):]
                    cleaned = cleaned.lstrip(",.:; ").strip()
                    found_one = True
                    stripped_this_round = True
                    break
            if not stripped_this_round:
                break

        if not found_one:
            return None

        # Refute-only correction (no new statement provided)
        if not cleaned:
            return ""

        # If what's left has a comma, the user might have written something
        # like "you keep saying X, the truth is Y" — try to find the actual
        # new statement by looking past commas. Prefer a comma-tail that
        # parses cleanly over the full text, because the parser is too
        # lenient and will accept "that, the edubba was for X" as a
        # statement with subject "that, edubba".
        if "," in cleaned:
            parser = getattr(getattr(self, "bridge", None), "brain", None)
            parser = getattr(parser, "parser", None) if parser else None
            if parser is not None:
                try:
                    parts = cleaned.split(",")
                    # Try each comma-separated tail, longest first
                    for i in range(1, len(parts)):
                        tail = ",".join(parts[i:]).strip()
                        if not tail:
                            continue
                        parsed = parser.parse(tail)
                        # Accept the tail if it parses AND the subject
                        # doesn't contain a comma (that means it's clean)
                        if parsed is not None and "," not in parsed.subject:
                            return tail
                except Exception:
                    pass

        return cleaned

    def _sara_turn(self, user_input: str) -> str:
        """Run Sara's brain BEFORE the LLM sees the input.

        The cortex (LLM) does not get to decide whether to consult the brain.
        The brain is consulted unconditionally, every turn. This is the
        architectural commitment that the LLM is the senses, not the source
        of cognition.

        Three operations run automatically:

        1. **Auto-recognize** — split the input into property words and run
           parallel wavefront recognition. Surface what Sara identifies.

        2. **Auto-teach** — if the input is a declarative statement (not a
           question, contains a copula or relational verb), call brain.teach()
           directly. Knowledge persists immediately, no LLM discretion.

        3. **Auto-disambiguate** — for any non-trivial words that don't
           resolve, run brain.did_you_mean() to surface fuzzy candidates.

        Returns a string suitable for injection into the system prompt
        describing what Sara did. Empty string if no operations fired.
        """
        notes: list[str] = []
        text = user_input.strip()
        if not text:
            return ""

        text_lower = text.lower()
        is_question = (
            text_lower.endswith("?")
            or any(text_lower.startswith(w) for w in (
                "what ", "who ", "where ", "when ", "why ", "how ",
                "is ", "are ", "do ", "does ", "did ", "can ",
                "could ", "would ", "should ", "tell me ", "show me ",
            ))
        )

        # ── 0. Correction detection (runs before normal teach) ──
        # If the user is explicitly correcting a previous claim,
        # extract the corrected statement and teach it. The refutation
        # of the previous claim is handled separately by walking the
        # last assistant message and refuting any sentence whose key
        # nouns appear in the corrected statement.
        # Empty string means "refute-only" — user wants to refute the
        # previous response without providing a new statement.
        correction = self._detect_correction(text)
        if correction is not None:
            # Find the last assistant message and refute claims that
            # contradict the user's correction.
            refuted_count = 0
            last_assistant = None
            for msg in reversed(self.messages[:-1]):  # skip current user msg
                if msg.get("role") == "assistant" and msg.get("content"):
                    last_assistant = msg["content"]
                    break
            if last_assistant:
                # Best-effort: try to refute each sentence of the assistant's
                # response. Sentences that contradict the user's correction
                # get marked known-to-be-false. We also record them in
                # _recent_refutations so the next turn's system prompt can
                # tell the model to disregard them — preventing context
                # poisoning where the model anchors on its own wrong answers.
                for sent in self._split_sentences(last_assistant):
                    try:
                        r = self.bridge.brain.refute(sent)
                        if r is not None:
                            refuted_count += 1
                            # Truncate the sentence for the disregard list
                            self._recent_refutations.append(sent[:140])
                    except Exception:
                        pass
                if refuted_count > 0:
                    self.bridge.brain.conn.commit()
                    notes.append(
                        f"AUTO-REFUTED: {refuted_count} claim(s) from the previous response"
                    )
                    # Cap the refutations list so it doesn't grow forever
                    self._recent_refutations = self._recent_refutations[-10:]

            # Teach the corrected statement (if one was provided)
            if correction:  # non-empty means user gave a replacement
                try:
                    parsed = self.bridge.brain.parser.parse(correction)
                    if parsed is not None and parsed.negated:
                        # Even the correction itself can be negated
                        positive = f"{parsed.subject} is {parsed.obj}"
                        r = self.bridge.brain.refute(positive)
                        if r is not None:
                            self.bridge.brain.conn.commit()
                            notes.append(
                                f"AUTO-REFUTED (correction with negation): "
                                f"{r.path_label} (path #{r.path_id})"
                            )
                    else:
                        result = self.bridge.brain.teach(correction)
                        if result is not None:
                            self.bridge.brain.conn.commit()
                            notes.append(
                                f"AUTO-TAUGHT (correction): {result.path_label} (path #{result.path_id})"
                            )
                except Exception:
                    pass

        # ── 1. Auto-teach OR auto-refute declarative statements ──
        # Uses the cortex EnhancedParser, which handles:
        #   - Compound statements ("X is Y. A is B" → two facts)
        #   - Source extraction ("according to wikipedia, X is Y")
        #   - Negation in multiple forms ("X is not Y", "X did not Y")
        #   - Pronoun rejection ("it was X" → no useless paths)
        #   - Quantifier weighting
        elif not is_question:
            try:
                from ..cortex.parser import EnhancedParser
                cortex_parser = getattr(self, "_cortex_parser", None)
                if cortex_parser is None:
                    cortex_parser = EnhancedParser(self.bridge.brain.taxonomy)
                    self._cortex_parser = cortex_parser

                parsed_turn = cortex_parser.parse(text)
                taught = 0
                refuted = 0
                for fact in parsed_turn.facts:
                    stmt = fact.original_text or text
                    if fact.negated:
                        # Build a positive form to refute. The cortex uses
                        # the same relation taxonomy so this round-trips.
                        if fact.relation == "is_a":
                            positive = f"{fact.subject} is {fact.obj}"
                        elif fact.relation.startswith("has_"):
                            positive = f"{fact.subject} is {fact.obj}"
                        elif fact.relation == "has":
                            positive = f"{fact.subject} has {fact.obj}"
                        else:
                            positive = f"{fact.subject} {fact.relation} {fact.obj}"
                        result = self.bridge.brain.refute(positive)
                        if result is not None:
                            self.bridge.brain.conn.commit()
                            self._recent_refutations.append(positive[:140])
                            self._recent_refutations = self._recent_refutations[-10:]
                            refuted += 1
                            notes.append(
                                f"AUTO-REFUTED: {positive} "
                                f"(path #{result.path_id})"
                            )
                    else:
                        result = self.bridge.brain.teach(stmt)
                        if result is not None:
                            self.bridge.brain.conn.commit()
                            taught += 1
                            notes.append(
                                f"AUTO-TAUGHT: {result.path_label} "
                                f"(path #{result.path_id})"
                            )
                if taught + refuted > 1:
                    notes.append(
                        f"COMPOUND PARSED: {taught} teach + {refuted} refute "
                        f"from one input"
                    )
            except Exception:
                # Parsing failures are expected and silent
                pass

        # ── 2. Auto-context (already done by build_system_prompt) ──
        # The system prompt builder calls bridge.context(user_input) and
        # injects relevant facts. We don't duplicate that here, but we do
        # add a note so the model knows Sara was consulted.
        notes.append("AUTO-QUERIED: Sara's relevant knowledge has been pre-loaded into this prompt above.")

        # ── 3. Auto-disambiguate unknown words ──
        # For words that don't resolve, find fuzzy candidates so the model
        # can recognize Sara already knows the term under a slightly
        # different spelling.
        words = [
            w.strip(".,;:!?\"'()-")
            for w in text_lower.split()
            if len(w.strip(".,;:!?\"'()-")) > 3
        ]
        skip = {
            "the", "and", "what", "this", "that", "with", "from", "have",
            "your", "yours", "mine", "their", "where", "when", "they",
            "them", "these", "those", "into", "about", "tell", "show",
            "know", "think", "would", "could", "should", "will", "does",
            "doing", "didnt", "havent", "isnt", "arent", "wasnt", "werent",
        }
        unresolved: list[str] = []
        for word in words[:8]:  # cap at 8 words to keep prompt small
            if word in skip:
                continue
            try:
                if self.bridge.brain.neuron_repo.resolve(word) is None:
                    candidates = self.bridge.brain.did_you_mean(word)
                    if candidates:
                        cand_strs = [c["label"] for c in candidates[:3]]
                        unresolved.append(
                            f"  '{word}' not in brain. Closest: {', '.join(cand_strs)}"
                        )
                    else:
                        unresolved.append(f"  '{word}' not in brain. No close matches.")
            except Exception:
                pass
        if unresolved:
            notes.append("AUTO-DISAMBIGUATED:\n" + "\n".join(unresolved))

        return "\n\n".join(notes)

    def _validate_and_execute(self, tool_call: dict) -> str:
        """Validate a tool call with Sara, execute if approved, observe outcome."""
        func = tool_call.get("function", {})
        name = func.get("name", "")
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            return f"Error: could not parse arguments for {name}"

        # Brain read tools don't need validation — they're just queries
        if name.startswith("brain_"):
            return dispatch(name, arguments, self.bridge, self.sandbox, self.cwd)

        # Action tools: validate with Sara first
        proposal = f"Tool: {name}, Arguments: {json.dumps(arguments)}"
        validation = self.validator.check_proposal(proposal)

        if not validation.approved and validation.correction:
            # Sara found a conflict — return the correction as the tool result
            # The LLM will see this and adjust its approach
            return (
                f"SARA CORRECTION: {validation.correction}\n\n"
                f"Please adjust your approach and try again."
            )

        # Approved — execute the tool
        result = dispatch(name, arguments, self.bridge, self.sandbox, self.cwd)

        # Observational learning: Sara records what happened
        observations = self._extract_observations(name, arguments, result)
        for obs in observations:
            self.bridge.observe(obs)

        return result

    def _extract_observations(
        self, tool_name: str, arguments: dict, result: str
    ) -> list[str]:
        """Extract teachable observations from a tool execution.

        Sara learns from what the LLM does — like the Perceiver learns from vision.
        """
        observations = []

        if tool_name == "read_file":
            path = arguments.get("path", "")
            if "Error" not in result and "not found" not in result:
                # Extract filename for cleaner labels
                from pathlib import Path as P
                fname = P(path).stem
                observations.append(f"{fname} is a document")

        elif tool_name == "write_file":
            path = arguments.get("path", "")
            if "Written:" in result:
                from pathlib import Path as P
                fname = P(path).stem
                observations.append(f"{fname} was created")

        elif tool_name == "execute_python":
            if "return code: 0" in result:
                observations.append("python code is successful")
            elif "return code:" in result:
                observations.append("python code has errors")

        elif tool_name == "shell_command":
            cmd = arguments.get("command", "").split()
            if cmd and "return code: 0" in result:
                observations.append(f"{cmd[0]} is successful")

        return observations

    def _post_turn_observe(self, llm_response: str) -> None:
        """After the LLM responds, extract key facts and teach Sara.

        The LLM's summary of what it read/did contains the distilled knowledge.
        Sara learns from the LLM's observations — like learning from a cortex
        that just processed sensory input.
        """
        # Look through recent messages for tool results from this turn
        # If the LLM just read a file and summarized it, the summary
        # contains the key facts Sara should learn
        recent_tools: list[tuple[str, str]] = []
        for msg in reversed(self.messages):
            if msg.get("role") == "user" and msg["content"].startswith("Tool results:"):
                # Text-parsed tool results
                recent_tools.append(("tool_results", msg["content"]))
                break
            if msg.get("role") == "tool":
                recent_tools.append(("tool", msg.get("content", "")))
            if msg.get("role") == "user" and not msg["content"].startswith("Tool"):
                break  # Reached the user's original message

        if not recent_tools:
            return

        # The LLM's response IS the processed observation.
        # Extract sentences that look like facts and teach Sara.
        facts = self._extract_facts_from_summary(llm_response)
        for fact in facts:
            self.bridge.observe(fact)

    def _extract_facts_from_summary(self, text: str) -> list[str]:
        """Extract teachable facts from the LLM's summary response.

        Looks for sentences containing relation verbs the parser understands
        (is, are, contains, includes, requires). Also extracts key noun phrases
        as simpler "X is Y" statements when possible.
        """
        import re

        facts = []
        seen: set[str] = set()

        def _clean(s: str) -> str:
            """Strip markdown formatting but preserve hyphens in words."""
            s = re.sub(r"\*\*|`|#+\s*", "", s)  # bold, code, headings
            s = re.sub(r"^\s*[-*]\s+", "", s)    # bullet points
            s = re.sub(r"^\s*\d+[.)]\s+", "", s) # numbered lists
            return s.strip()

        def _add(fact: str) -> None:
            clean = _clean(fact)
            if 10 < len(clean) < 120 and clean not in seen:
                seen.add(clean)
                facts.append(clean)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Normalize hedging language before extraction so
        # "appears to be", "seems to be" etc. become "is"
        hedge_map = [
            (r"\bappears to be\b", "is"),
            (r"\bseems to be\b", "is"),
            (r"\bcan be considered\b", "is"),
            (r"\bis considered\b", "is"),
            (r"\bis primarily\b", "is"),
            (r"\bare primarily\b", "are"),
            (r"\bis essentially\b", "is"),
            (r"\bare essentially\b", "are"),
            (r"\bfocuses on\b", "includes"),
            (r"\bhighlights\b", "includes"),
            (r"\bstores\b", "contains"),
        ]

        # Relation verbs the Sara parser understands
        verbs = r"(?:is|are|contains|includes|requires|follows|excludes)"

        for sentence in sentences:
            sentence = sentence.strip().rstrip(".")
            if not sentence:
                continue

            # Normalize hedging language
            normalized = sentence
            for pattern, replacement in hedge_map:
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

            # Find "subject VERB object" — take the shortest subject
            # before the verb (last 1-4 words)
            match = re.search(
                r"([\w][\w\s-]{1,40}?)\s+"       # subject (lazy)
                rf"\b({verbs})\b\s+"               # verb
                r"([\w][\w\s_-]+)",                # object (includes underscores)
                normalized,
                re.IGNORECASE,
            )
            if not match:
                continue

            subj = match.group(1).strip()
            verb = match.group(2).strip().lower()
            obj = match.group(3).strip()

            # Trim subject to last 1-4 meaningful words
            subj_words = subj.split()
            if len(subj_words) > 4:
                subj = " ".join(subj_words[-4:])

            # Trim object at first comma/conjunction/relative clause
            obj = re.split(r",|\band\b|\bbut\b|\bnot\b|\bwhich\b|\bthat\b|\bwhen\b", obj)[0].strip()

            # Skip noise: pronouns, prepositions, filler words
            skip_subjects = {"here", "there", "it", "this", "that", "these", "those", "i", "you", "we"}
            skip_starts = {"with", "from", "by", "for", "in", "on", "at", "to", "as", "if", "suggesting", "while"}
            first_word = subj.split()[0].lower() if subj else ""
            if subj.lower() in skip_subjects or first_word in skip_starts:
                continue

            if obj and len(obj) > 2:
                _add(f"{subj} {verb} {obj}")

        return facts[:10]  # Cap at 10 per turn

    def resume_session(self, session_id: str) -> bool:
        """Resume a previous session. Returns True if found."""
        messages = self.session_store.load(session_id)
        if messages is None:
            return False
        self.messages = messages
        self.session_id = session_id
        return True

    def _save_session(self) -> None:
        """Persist current conversation to disk."""
        if self.session_id is None:
            self.session_id = self.session_store.new_session_id()
        self.session_store.save(self.session_id, self.messages)

    def run_interactive(self) -> None:
        """Run the interactive REPL loop."""
        while True:
            try:
                user_input = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nShutting down — sleep, not death.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/quit"):
                print("Shutting down — sleep, not death.")
                break

            try:
                response = self.turn(user_input)
                print(f"\nsara> {response}")
            except Exception as e:
                print(f"\nError: {e}", file=sys.stderr)
