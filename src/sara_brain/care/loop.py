"""Care loop — wraps AgentLoop with role-aware behavior and interaction logging."""

from __future__ import annotations

import time

from ..agent.bridge import AgentBridge
from ..agent.loop import AgentLoop
from ..core.brain import Brain
from ..core.temporal import TemporalLinker
from ..core.trust import TrustManager
from ..storage.account_repo import Account
from ..storage.interaction_repo import Interaction
from . import system_prompts


class CareLoop:
    """Wraps the agent loop with care-specific behavior.

    - Injects role-specific system prompts
    - Logs every interaction
    - Links learned facts to temporal neurons
    - Manages trust status based on who is speaking
    """

    def __init__(self, brain: Brain, account: Account,
                 model: str, base_url: str = "http://localhost:11434",
                 sandbox_timeout: int = 30) -> None:
        self.brain = brain
        self.account = account
        self.bridge = AgentBridge(brain)
        self.temporal = TemporalLinker(
            brain.neuron_repo, brain.segment_repo, brain.path_repo
        )
        self.trust = TrustManager(
            brain.neuron_repo, brain.segment_repo,
            brain.path_repo, brain.account_repo
        )

        # Build the underlying agent loop
        self.agent = AgentLoop(
            brain=brain,
            model=model,
            base_url=base_url,
            sandbox_timeout=sandbox_timeout,
        )

        # Override the system prompt builder
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build role-specific system prompt."""
        patient = self.brain.account_repo.get_reader()
        patient_name = patient.name if patient else "the patient"

        if self.account.role == "reader":
            # Gather known facts about the patient for context
            known_facts = self._gather_patient_facts()
            return system_prompts.reader_prompt(self.account.name, known_facts)
        elif self.account.role == "teacher":
            return system_prompts.teacher_prompt(self.account.name, patient_name)
        elif self.account.role == "doctor":
            return system_prompts.doctor_prompt(self.account.name, patient_name)
        else:
            return system_prompts.reader_prompt(self.account.name)

    def _gather_patient_facts(self) -> str:
        """Gather what Sara knows about the patient for the system prompt."""
        if not self.account.neuron_id:
            return ""
        traces = self.brain.why(self.account.name.lower())
        if not traces:
            return ""
        lines = []
        for t in traces:
            if t.source_text and not t.source_text.startswith("["):
                lines.append(f"- {t.source_text}")
        return "\n".join(lines[:20])  # Cap at 20 facts for prompt size

    def turn(self, user_input: str) -> str:
        """Process one turn with interaction logging and trust management."""
        # Determine interaction type
        interaction_type = self._classify_input(user_input)

        # Process through the agent loop
        response = self.agent.turn(user_input)

        # Log the interaction
        interaction = Interaction(
            id=None,
            account_id=self.account.id,
            interaction_type=interaction_type,
            content=user_input,
            response=response,
        )
        self.brain.interaction_repo.record(interaction)
        self.brain.conn.commit()

        return response

    def _classify_input(self, text: str) -> str:
        """Classify user input by interaction type."""
        lower = text.lower().strip()

        # Questions
        question_words = ("who ", "what ", "when ", "where ", "why ", "how ",
                          "is ", "are ", "do ", "does ", "did ", "can ",
                          "?")
        if any(lower.startswith(w) for w in question_words) or lower.endswith("?"):
            return "ask"

        # Review requests (teacher/doctor)
        review_phrases = ("how did", "how was", "how were", "what happened",
                          "summary", "report", "review", "show me",
                          "contested", "observations", "trend")
        if any(phrase in lower for phrase in review_phrases):
            return "review"

        # Teaching (teacher/doctor explicit teaching)
        if self.account.role in ("teacher", "doctor"):
            teach_signals = ("teach ", "remember that", "know that",
                             " is ", " are ", " has ", " was ")
            if any(signal in lower for signal in teach_signals):
                return "teach"

        # Patient telling Sara something
        if self.account.role == "reader":
            return "tell"

        return "tell"

    def run_interactive(self) -> None:
        """Run the interactive care loop."""
        # Warm greeting based on role
        if self.account.role == "reader":
            print(f"\n  Hello, {self.account.name}. It's good to talk to you.")
            print("  Just tell me what's on your mind, or ask me anything.\n")
        elif self.account.role == "teacher":
            patient = self.brain.account_repo.get_reader()
            patient_name = patient.name if patient else "the patient"
            print(f"\n  Hi {self.account.name}. Sara Care is ready.")
            print(f"  You can teach me about {patient_name}, or ask how their day went.\n")
        elif self.account.role == "doctor":
            patient = self.brain.account_repo.get_reader()
            patient_name = patient.name if patient else "the patient"
            print(f"\n  {self.account.name}, Sara Care clinical interface ready.")
            print(f"  Patient: {patient_name}")
            print("  You can review interactions, annotate observations, or check trends.\n")

        prompt_name = self.account.name.lower().split()[0]

        while True:
            try:
                user_input = input(f"{prompt_name}> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye. Sara will remember everything.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/quit", "bye", "goodbye"):
                print("  Goodbye. Sara will remember everything.")
                break

            try:
                response = self.turn(user_input)
                print(f"\nsara> {response}\n")
            except Exception as e:
                print(f"\n  Something went wrong: {e}\n")
