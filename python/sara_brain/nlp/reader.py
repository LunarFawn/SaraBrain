"""Document reading via LLM — Sara's reading comprehension cortex.

The LLM reads a document and extracts facts/rules as teachable statements.
Same role as VisionObserver but for text instead of images.
Stdlib only.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from .provider import LLMProvider, AnthropicProvider
from .translator import is_blocked_domain
from ..innate.primitives import get_sensory, get_structural, get_relational


class DocumentReader:
    """Reads documents through the LLM cortex, extracts teachable facts."""

    def __init__(self, api_url: str, api_key: str, model: str,
                 provider: LLMProvider | None = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider or AnthropicProvider()

    def _call_api(self, system: str | None, user_text: str,
                  max_tokens: int = 2000) -> str | None:
        if self.provider.needs_api_key() and is_blocked_domain(self.api_url):
            raise ValueError(
                f"Blocked API domain: {self.api_url}. "
                "Only Anthropic (Claude) endpoints are allowed."
            )

        messages = [{"role": "user", "content": user_text}]
        payload = self.provider.build_chat_payload(
            model=self.model,
            system=system,
            messages=messages,
            temperature=0,
            max_tokens=max_tokens,
        )

        url = self.provider.build_endpoint_url(self.api_url)
        headers = self.provider.build_headers(self.api_key)
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                text = self.provider.parse_text_response(body)
                return text.strip() if text else None
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
            return None

    def read(self, document_text: str) -> list[str]:
        """Extract facts/rules from a document as teachable statements.

        Returns list of simple statements like:
            "python functions use snake_case"
            "imports go at top of file"
        """
        primitives = ", ".join(sorted(get_structural()))

        system = (
            "You are a reading cortex. You read documents and extract every "
            "fact, rule, convention, and pattern as simple statements.\n"
            "Each statement should be one fact on its own line.\n"
            "Use the format: <subject> is/has/follows/requires <property>\n"
            f"Structural concepts you can use: {primitives}\n"
            "Be thorough. Extract everything. One fact per line.\n"
            "Simple words only. No explanations, no commentary."
        )

        raw = self._call_api(system, document_text)
        if raw is None:
            return []
        return self._parse_statements(raw)

    def inquire(self, document_text: str,
                associations: dict[str, list[str]]) -> list[str]:
        """Ask directed questions about the document based on Sara's associations.

        Like vision's directed inquiry — Sara knows about certain categories
        and asks the cortex to look for those specifically.
        """
        if not associations:
            return []

        lines = []
        for assoc, props in associations.items():
            lines.append(f"- {assoc}: known values are {', '.join(props)}")

        system = (
            "You are a reading cortex. The brain already knows about these "
            "associations and their known values:\n"
            + "\n".join(lines) + "\n\n"
            "Read the document and find any facts related to these associations "
            "that were not already extracted.\n"
            "One fact per line. Format: <subject> is/has/follows/requires <property>\n"
            "Simple words only. If nothing new, say NONE."
        )

        raw = self._call_api(system, document_text)
        if raw is None or raw.strip().upper() == "NONE":
            return []
        return self._parse_statements(raw)

    def summarize(self, learned_facts: list[str]) -> str:
        """Articulate what Sara learned — the LLM voices Sara's understanding."""
        facts_text = "\n".join(f"- {f}" for f in learned_facts)

        system = (
            "You are voicing what a brain has learned. Given these facts it "
            "absorbed, explain in plain language what it now understands.\n"
            "Speak as Sara: 'I learned that...'\n"
            "Be concise. Group related facts together."
        )

        raw = self._call_api(system, facts_text, max_tokens=500)
        return raw or "I'm not sure how to describe what I learned."

    def explain(self, concept: str) -> list[str]:
        """Break down an unknown concept into teachable facts using innate primitives.

        This is option C — cortex helps, parent confirms.
        """
        sensory = ", ".join(sorted(get_sensory()))
        structural = ", ".join(sorted(get_structural()))
        relational = ", ".join(sorted(get_relational()))

        system = (
            "You are a reading cortex helping a brain understand a new concept.\n"
            "Break it down into simple facts using only these primitives:\n"
            f"  Sensory: {sensory}\n"
            f"  Structural: {structural}\n"
            f"  Relational: {relational}\n\n"
            "One fact per line. Format: <concept> is/has/contains/requires <property>\n"
            "Simple words only. No explanations."
        )

        raw = self._call_api(system, f"What is '{concept}'?", max_tokens=300)
        if raw is None:
            return []
        return self._parse_statements(raw)

    @staticmethod
    def _parse_statements(raw_text: str) -> list[str]:
        """Parse LLM output into clean statement lines."""
        statements: list[str] = []
        for line in raw_text.splitlines():
            cleaned = line.strip().lstrip("-*•·0123456789.)")
            cleaned = cleaned.strip()
            if not cleaned:
                continue
            if cleaned.upper() == "NONE":
                continue
            if len(cleaned) > 200:
                continue
            if any(kw in cleaned.lower() for kw in ("http", "www", "import ", "def ", "class ")):
                continue
            statements.append(cleaned)
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for s in statements:
            lower = s.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(s)
        return result
