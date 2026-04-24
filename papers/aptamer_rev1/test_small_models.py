"""Test llama3.2:1b and llama3.2:3b reading Sara's aptamer substrate.

Protocol:
  - Load all 169 triples from aptamer_exec.db
  - For each test question, ask each model twice:
      (a) WITH Sara context (triples injected as reference)
      (b) WITHOUT Sara context (bare question, measures training-alone)
  - Log every response with token counts
  - Do NOT grade here — the session running this script is infected
    (it's the teaching session). Grading must happen in a fresh session
    or by the author directly.

Usage:
    .venv/bin/python papers/aptamer_rev1/test_small_models.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


OLLAMA_URL = "http://localhost:11434/api/generate"
DB_PATH = "aptamer_exec.db"
MODELS = ["llama3.2:1b", "llama3.2:3b"]

QUESTIONS = [
    "What is the molecular snare?",
    "What is marker theory?",
    "What does the serena rna analysis tool do and what metrics does it provide?",
    "What is the knob?",
    "What happens during state transitions in rna aptamers?",
    "What are SSNG1, SSNG2, and SSNG3?",
    "What is the 5'3' static stem?",
    "What is the fold signal region?",
]


def load_triples(db_path: str) -> list[tuple[str, str, str]]:
    """Reconstruct every (subject, relation, object) triple by walking paths."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    sql = """
    SELECT p.id, p.source_text
    FROM paths p
    ORDER BY p.id
    """
    triples: list[tuple[str, str, str]] = []
    for pid, source_text in cur.execute(sql).fetchall():
        # path_steps → segments → neurons
        step_sql = """
        SELECT n1.label, s.relation, n2.label
        FROM path_steps ps
        JOIN segments s ON s.id = ps.segment_id
        JOIN neurons n1 ON n1.id = s.source_id
        JOIN neurons n2 ON n2.id = s.target_id
        WHERE ps.path_id = ?
        ORDER BY ps.step_order
        """
        steps = cur.execute(step_sql, (pid,)).fetchall()
        # Each path is property → relation → concept; we want the logical
        # (subject, relation, obj) which matches source_text better.
        # The source_text we stored is "subject relation obj" (or similar).
        # Recover it by splitting the first attribute step.
        if len(steps) == 2:
            # step 0: prop → relation_attr (relation = actual verb)
            # step 1: relation_attr → concept (relation = "describes")
            _, verb, _ = steps[0]
            prop = steps[0][0]
            concept = steps[1][2]
            # Source-text form: concept verb prop
            triples.append((concept, verb, prop))
    conn.close()
    return triples


def format_triples_as_context(triples: list[tuple[str, str, str]]) -> str:
    lines = []
    for s, r, o in triples:
        lines.append(f"  {s} --[{r}]--> {o}")
    return "\n".join(lines)


def ask_ollama(model: str, prompt: str, timeout: int = 120) -> tuple[str, int]:
    """POST to Ollama /api/generate. Returns (response_text, eval_count)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 800,
        },
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        return body.get("response", ""), body.get("eval_count", 0)
    except urllib.error.URLError as e:
        return f"<<OLLAMA_ERROR: {e}>>", 0
    except Exception as e:
        return f"<<ERROR: {e}>>", 0


def build_sara_prompt(context: str, question: str) -> str:
    return f"""You are reading a knowledge graph called Sara Brain. Below are all the facts Sara holds about an RNA aptamer research paper.

<sara_triples>
{context}
</sara_triples>

Based ONLY on the triples above, answer the following question. If the answer isn't in the triples, say "Sara doesn't have this information." Do not add outside knowledge.

Question: {question}

Answer:"""


def build_bare_prompt(question: str) -> str:
    return f"""Answer the following question based on your knowledge.

Question: {question}

Answer:"""


def main() -> None:
    triples = load_triples(DB_PATH)
    print(f"Loaded {len(triples)} triples from {DB_PATH}", file=sys.stderr)
    context = format_triples_as_context(triples)
    print(f"Context length: {len(context)} chars", file=sys.stderr)

    results = []
    for q_idx, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[{q_idx}/{len(QUESTIONS)}] Q: {question}", file=sys.stderr)
        print('='*70, file=sys.stderr)

        for model in MODELS:
            for condition, prompt in [
                ("with_sara", build_sara_prompt(context, question)),
                ("bare", build_bare_prompt(question)),
            ]:
                t0 = time.time()
                answer, eval_count = ask_ollama(model, prompt)
                elapsed = time.time() - t0
                entry = {
                    "question": question,
                    "model": model,
                    "condition": condition,
                    "answer": answer.strip(),
                    "output_tokens": eval_count,
                    "elapsed_s": round(elapsed, 1),
                }
                results.append(entry)
                label = f"{model:15s} {condition:10s}"
                print(f"\n--- {label} ({eval_count} tok, {elapsed:.1f}s) ---",
                      file=sys.stderr)
                print(answer.strip(), file=sys.stderr)

    # Write results to JSON for later grading
    output_path = Path("papers/aptamer_rev1/small_model_results.json")
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nWrote {len(results)} results to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
