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

    @staticmethod
    def _chunk_by_paragraphs(text: str, max_chars: int = 1500) -> list[str]:
        """Split text into chunks at paragraph boundaries.

        Default chunker for plain text and HTML-derived text.
        """
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            # If a single paragraph exceeds the limit, split on newlines
            if len(para) > max_chars:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                lines = para.split("\n")
                line_buf: list[str] = []
                line_len = 0
                for line in lines:
                    if line_len + len(line) > max_chars and line_buf:
                        chunks.append("\n".join(line_buf))
                        line_buf = []
                        line_len = 0
                    line_buf.append(line)
                    line_len += len(line) + 1
                if line_buf:
                    chunks.append("\n".join(line_buf))
                continue

            if current_len + len(para) > max_chars and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para) + 2

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    @staticmethod
    def _chunk_markdown(text: str, max_chars: int = 1500) -> list[str]:
        """Split markdown at heading boundaries.

        Keeps each section together so the LLM gets coherent context.
        Falls back to paragraph splitting within oversized sections.
        """
        import re
        # Split on headings (##, ###, etc.) keeping the heading with its body
        sections = re.split(r"(?=^#{1,6}\s)", text, flags=re.MULTILINE)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) > max_chars:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                # Section too big — fall back to paragraph splitting
                chunks.extend(
                    DocumentReader._chunk_by_paragraphs(section, max_chars)
                )
                continue

            if current_len + len(section) > max_chars and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(section)
            current_len += len(section) + 2

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    @staticmethod
    def _chunk_code(text: str, max_chars: int = 2000) -> list[str]:
        """Split source code at function/class boundaries.

        Larger chunk size since code is denser and splitting mid-function
        loses context.
        """
        import re
        # Split on top-level def/class lines
        parts = re.split(r"(?=^(?:def |class |async def ))", text, flags=re.MULTILINE)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part = part.rstrip()
            if not part:
                continue
            if len(part) > max_chars:
                if current:
                    chunks.append("\n".join(current))
                    current = []
                    current_len = 0
                # Single function too big — split on blank lines
                chunks.extend(
                    DocumentReader._chunk_by_paragraphs(part, max_chars)
                )
                continue

            if current_len + len(part) > max_chars and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(part)
            current_len += len(part) + 1

        if current:
            chunks.append("\n".join(current))
        return chunks

    @staticmethod
    def _detect_format(text: str, source: str = "") -> str:
        """Detect document format from source label or content."""
        source_lower = source.lower()
        if source_lower.endswith((".md", ".markdown")):
            return "markdown"
        if source_lower.endswith((".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h")):
            return "code"
        # Heuristic: markdown headings in content
        if text.lstrip().startswith("#") and "\n## " in text:
            return "markdown"
        # Heuristic: code signatures in content
        if "\ndef " in text or "\nclass " in text:
            return "code"
        return "text"

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 1500, source: str = "") -> list[str]:
        """Split text into chunks using format-appropriate strategy.

        Detects whether the content is markdown, source code, or plain
        text and applies the right chunking method.
        """
        fmt = DocumentReader._detect_format(text, source)
        if fmt == "markdown":
            return DocumentReader._chunk_markdown(text, max_chars)
        if fmt == "code":
            return DocumentReader._chunk_code(text, max_chars)
        return DocumentReader._chunk_by_paragraphs(text, max_chars)

    def read(self, document_text: str, source: str = "",
             on_chunk=None) -> list[str]:
        """Extract facts/rules from a document as teachable statements.

        Chunks the document using format-appropriate splitting so small
        models can focus on each section without dropping facts.

        Args:
            on_chunk: Optional callback(chunk_num, total_chunks, facts_so_far)
                      called after each chunk is processed.

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
            "Do not skip dates, numbers, or names.\n"
            "Simple words only. No explanations, no commentary."
        )

        chunks = self._chunk_text(document_text, source=source)
        all_statements: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            raw = self._call_api(system, chunk)
            if raw is not None:
                all_statements.extend(self._parse_statements(raw))
            if on_chunk is not None:
                on_chunk(i, len(chunks), len(all_statements))

        # Deduplicate across chunks
        seen: set[str] = set()
        result: list[str] = []
        for s in all_statements:
            lower = s.lower()
            if lower not in seen:
                seen.add(lower)
                result.append(s)
        return result

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
