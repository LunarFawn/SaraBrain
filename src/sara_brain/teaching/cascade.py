"""Teacher cascade orchestrator.

Three phases, one-model-at-a-time by design (fits 8 GB M2 mini):
  Phase 1a  — 5 claim 1B sensors across all sentences → claim_out.jsonl
  Phase 1b  — 3 prose 1B sensors across all sentences → prose_out.jsonl
              (then unload 1B)
  Phase 2   — 3B claim-integrator + prose-integrator + synthesizer
              → teach_out.jsonl
              (then unload 3B)

Concurrency within a phase: a thread pool fires multiple sensor
prompts in parallel against the loaded Ollama model. Ollama's
`OLLAMA_NUM_PARALLEL` on the server side decides how many truly
run in parallel.

Everything is logged — raw prompt output, timings, witness votes —
so a bad claim can be traced back to the exact sensor and input.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import prompts
from .ollama_client import generate


@dataclass
class SensorOut:
    sentence: str
    sensor: str
    raw: str
    elapsed_s: float


def _call_sensor(model: str, sensor: str, prompt_tpl: str,
                 sentence: str) -> SensorOut:
    prompt = prompt_tpl.format(sentence=sentence)
    raw, dt = generate(model, prompt)
    return SensorOut(sentence=sentence, sensor=sensor,
                     raw=raw.strip(), elapsed_s=dt)


def _split_lines(value):
    """Recursively convert multi-line strings into lists of lines.

    JSON escapes literal newlines inside strings as \\n, which makes
    multi-line LLM output unreadable in a pretty-printed file. Splitting
    on newlines before serialization renders each line on its own row.
    """
    if isinstance(value, str):
        if "\n" in value:
            return value.splitlines()
        return value
    if isinstance(value, dict):
        return {k: _split_lines(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_split_lines(x) for x in value]
    return value


def _write_pretty_array(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as a single pretty-printed JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pretty = _split_lines(records)
    with path.open("w") as f:
        json.dump(pretty, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_claim_phase(sentences: list[str], model: str,
                    out_path: Path, max_workers: int = 5) -> None:
    """Phase 1a. All 5 claim sensors across every sentence."""
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, sentence in enumerate(sentences, 1):
            futures = [
                pool.submit(
                    _call_sensor, model, name, prompts.CLAIM_PROMPTS[name],
                    sentence,
                )
                for name in prompts.CLAIM_PROMPTS
            ]
            by_sensor: dict[str, dict] = {}
            for fut in futures:
                r = fut.result()
                by_sensor[r.sensor] = {
                    "raw": r.raw,
                    "elapsed_s": round(r.elapsed_s, 3),
                }
            records.append({
                "sentence": sentence,
                "phase": "claim",
                "model": model,
                "sensors": by_sensor,
                "timestamp": time.time(),
            })
            _write_pretty_array(out_path, records)
            print(f"  claim [{i}/{len(sentences)}] {sentence[:60]}…")


def run_prose_phase(sentences: list[str], model: str,
                    out_path: Path, max_workers: int = 3) -> None:
    """Phase 1b. All 3 prose sensors across every sentence."""
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, sentence in enumerate(sentences, 1):
            futures = [
                pool.submit(
                    _call_sensor, model, name, prompts.PROSE_PROMPTS[name],
                    sentence,
                )
                for name in prompts.PROSE_PROMPTS
            ]
            by_sensor: dict[str, dict] = {}
            for fut in futures:
                r = fut.result()
                by_sensor[r.sensor] = {
                    "raw": r.raw,
                    "elapsed_s": round(r.elapsed_s, 3),
                }
            records.append({
                "sentence": sentence,
                "phase": "prose",
                "model": model,
                "sensors": by_sensor,
                "timestamp": time.time(),
            })
            _write_pretty_array(out_path, records)
            print(f"  prose [{i}/{len(sentences)}] {sentence[:60]}…")


def run_integration_phase(claim_path: Path, prose_path: Path,
                          model: str, out_path: Path) -> None:
    """Phase 2. For each sentence, run claim-integrator, prose-integrator,
    then synthesizer. Produces the final teach list."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    claim_recs = {r["sentence"]: r for r in _read_array(claim_path)}
    prose_recs = {r["sentence"]: r for r in _read_array(prose_path)}
    sentences = list(claim_recs.keys())

    records: list[dict] = []
    for i, sentence in enumerate(sentences, 1):
        c = claim_recs[sentence]["sensors"]
        p = prose_recs.get(sentence, {"sensors": {}})["sensors"]

        # Claim integrator
        claim_prompt = prompts.CLAIM_INTEGRATOR.format(
            sentence=sentence,
            definition=c.get("definition", {}).get("raw", "NONE"),
            process=c.get("process", {}).get("raw", "NONE"),
            causation=c.get("causation", {}).get("raw", "NONE"),
            temporal=c.get("temporal", {}).get("raw", "NONE"),
            datetime=c.get("datetime", {}).get("raw", "NONE"),
        )
        claim_int_out, claim_int_dt = generate(model, claim_prompt)

        # Prose integrator
        prose_prompt = prompts.PROSE_INTEGRATOR.format(
            sentence=sentence,
            topic=p.get("topic", {}).get("raw", "NONE"),
            relation=p.get("relation", {}).get("raw", "NONE"),
            context=p.get("context", {}).get("raw", "NONE"),
        )
        prose_int_out, prose_int_dt = generate(model, prose_prompt)

        topic, relation, context = _parse_prose_integrator(prose_int_out)

        # Synthesizer
        synth_prompt = prompts.SYNTHESIZER.format(
            sentence=sentence,
            claims=claim_int_out.strip(),
            topic=topic, relation=relation, context=context,
        )
        synth_out, synth_dt = generate(model, synth_prompt)

        final_claims = _clean_claims(synth_out)

        records.append({
            "sentence": sentence,
            "phase": "integration",
            "model": model,
            "claim_integrator": {
                "raw": claim_int_out.strip(),
                "elapsed_s": round(claim_int_dt, 3),
            },
            "prose_integrator": {
                "raw": prose_int_out.strip(),
                "elapsed_s": round(prose_int_dt, 3),
                "topic": topic, "relation": relation, "context": context,
            },
            "synthesizer": {
                "raw": synth_out.strip(),
                "elapsed_s": round(synth_dt, 3),
            },
            "final_claims": final_claims,
            "timestamp": time.time(),
        })
        _write_pretty_array(out_path, records)
        print(f"  integrate [{i}/{len(sentences)}] "
              f"→ {len(final_claims)} claims")


def _rejoin_lines(value, known_string_keys=("raw",)):
    """Reverse of `_split_lines` for runtime use. Rejoins list-of-lines
    values under known string keys back into newline-joined strings.
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in known_string_keys and isinstance(v, list):
                out[k] = "\n".join(v)
            else:
                out[k] = _rejoin_lines(v, known_string_keys)
        return out
    if isinstance(value, list):
        return [_rejoin_lines(x, known_string_keys) for x in value]
    return value


def _read_array(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    return _rejoin_lines(data)


def _parse_prose_integrator(text: str) -> tuple[str, str, str]:
    topic = relation = context = "NONE"
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("topic:"):
            topic = line.split(":", 1)[1].strip() or "NONE"
        elif line.lower().startswith("relation:"):
            relation = line.split(":", 1)[1].strip() or "NONE"
        elif line.lower().startswith("context:"):
            context = line.split(":", 1)[1].strip() or "NONE"
    return topic, relation, context


def _clean_claims(text: str) -> list[str]:
    claims: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        # Strip common list-bullet prefixes
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            claims.append(line)
    return claims
