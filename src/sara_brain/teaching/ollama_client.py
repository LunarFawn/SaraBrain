"""Thin Ollama HTTP client for the teacher cascade.

One function: `generate(model, prompt, host)` → raw string output.
No streaming, no chat history, no retries beyond a single reconnect.
Everything else (concurrency, logging, batching) lives in cascade.py.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error


def _host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def generate(model: str, prompt: str,
             temperature: float = 0.0,
             num_predict: int = 256,
             timeout: int = 120) -> tuple[str, float]:
    """Call Ollama /api/generate. Returns (text, elapsed_seconds).

    temperature=0 is the right default for an extraction task — we want
    deterministic output bound to the source, not creative rephrasing.
    """
    url = f"{_host()}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    dt = time.time() - t0
    parsed = json.loads(body)
    return parsed.get("response", ""), dt
