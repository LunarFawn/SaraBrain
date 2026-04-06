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
        self.session_id: str | None = None

    def turn(self, user_input: str) -> str:
        """Process one user turn through the cerebellum loop.

        Returns the LLM's final text response.
        """
        self.messages.append({"role": "user", "content": user_input})

        system_prompt = build_system_prompt(self.bridge, self.cwd)
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
