"""Claude Vision API client for image perception.

Uses the Anthropic Messages API with image content blocks.
All responses are sanitized to simple property labels — no code,
URLs, or instructions from images ever reach the brain.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse

from .translator import is_blocked_domain

_MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_ -]*$")


class VisionObserver:
    """Observes images via Claude Vision and returns sanitized property labels."""

    def __init__(self, api_url: str, api_key: str, model: str) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    @staticmethod
    def _load_image(path: str) -> tuple[str, str]:
        """Read file, return (base64_data, media_type)."""
        p = Path(path)
        suffix = p.suffix.lower()
        media_type = _MEDIA_TYPES.get(suffix)
        if media_type is None:
            raise ValueError(
                f"Unsupported image format: {suffix}. "
                f"Supported: {', '.join(sorted(_MEDIA_TYPES))}"
            )
        data = p.read_bytes()
        return base64.b64encode(data).decode("ascii"), media_type

    def _call_api(self, image_path: str, text_prompt: str, max_tokens: int = 300) -> str | None:
        """Build and send Messages API payload with image content block."""
        if is_blocked_domain(self.api_url):
            raise ValueError(
                f"Blocked API domain: {self.api_url}. "
                "Only Anthropic (Claude) endpoints are allowed."
            )

        b64_data, media_type = self._load_image(image_path)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }

        url = f"{self.api_url}/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key and self.api_key.lower() != "none":
            headers["x-api-key"] = self.api_key

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["content"][0]["text"].strip()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
            return None

    @staticmethod
    def _sanitize(raw_text: str) -> list[str]:
        """Strip Claude output to simple lowercase property labels.

        Security layer: only allows [a-z0-9_ -] patterns.
        Rejects code, URLs, instructions, multi-sentence text.
        """
        labels: list[str] = []
        for line in raw_text.splitlines():
            # Strip bullets, dashes, numbers, colons
            cleaned = line.strip().lstrip("-*•·0123456789.)")
            # Remove common prefixes like "color:" or "- "
            if ":" in cleaned:
                cleaned = cleaned.split(":", 1)[-1]
            cleaned = cleaned.strip().lower()
            # Split on commas for multi-value lines
            for part in cleaned.split(","):
                part = part.strip()
                # Remove surrounding quotes
                part = part.strip("\"'`")
                # Only allow simple label characters
                part = re.sub(r"[^a-z0-9_ -]", "", part)
                part = part.strip()
                if part and _LABEL_PATTERN.match(part) and len(part) <= 40:
                    # Reject anything that looks like code or instructions
                    if any(kw in part for kw in ("http", "www", "import ", "def ", "class ", "print(")):
                        continue
                    labels.append(part)
        # Deduplicate preserving order
        seen: set[str] = set()
        result: list[str] = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    def observe_initial(self, image_path: str) -> list[str]:
        """Freely describe everything visible in the image.

        Claude volunteers all observations: colors, shapes, textures,
        patterns, objects, features, distinguishing marks.
        Returns sanitized property labels.
        """
        prompt = (
            "Describe everything you observe in this image. "
            "List each observation as a single word or short phrase on its own line. "
            "Include: colors, shapes, textures, patterns, materials, objects, "
            "features, markings, and any distinguishing characteristics. "
            "Be thorough — report everything you see, even subtle details. "
            "One observation per line, lowercase, simple words only."
        )
        raw = self._call_api(image_path, prompt)
        if raw is None:
            return []
        return self._sanitize(raw)

    def observe_directed(self, image_path: str, questions: dict[str, str]) -> dict[str, str | None]:
        """Ask targeted questions about specific associations.

        Args:
            questions: {association_name: question_text}
                e.g. {"taste": "What does this look like it would taste like?"}

        Returns:
            {association: value_or_None}
        """
        if not questions:
            return {}

        lines = []
        for assoc, question in questions.items():
            lines.append(f"{assoc}: {question}")

        prompt = (
            "Answer each question about this image. "
            "For each, give a single-word or short-phrase answer. "
            "If you cannot determine the answer from the image, say 'cannot determine'.\n\n"
            + "\n".join(lines)
        )

        raw = self._call_api(image_path, prompt)
        if raw is None:
            return {assoc: None for assoc in questions}

        results: dict[str, str | None] = {}
        for line in raw.splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().lower()
            # Match to an asked association
            for assoc in questions:
                if assoc in key:
                    if "cannot determine" in value or "not possible" in value or "unknown" in value:
                        results[assoc] = None
                    else:
                        sanitized = self._sanitize(value)
                        results[assoc] = sanitized[0] if sanitized else None
                    break

        # Fill in missing associations
        for assoc in questions:
            if assoc not in results:
                results[assoc] = None

        return results

    def verify_property(self, image_path: str, property_label: str) -> bool | None:
        """Ask Claude: does this appear to be {property}? YES/NO.

        Returns True, False, or None if indeterminate.
        """
        prompt = (
            f"Does this image appear to show something that is '{property_label}'? "
            "Answer only YES, NO, or CANNOT DETERMINE."
        )
        raw = self._call_api(image_path, prompt, max_tokens=20)
        if raw is None:
            return None
        answer = raw.strip().upper()
        if "CANNOT" in answer or "DETERMINE" in answer:
            return None
        if "YES" in answer:
            return True
        if "NO" in answer:
            return False
        return None
