"""sara-demo — Live demonstration CLI for Sara Brain.

Teach facts, ask questions, compare against a 1B model, show path traces.
Designed for live conference demos on minimal hardware (Pi 4, Uno Q).

Usage:
    sara-demo teach --brain demo.db
    sara-demo ask "What is X?" --brain demo.db
    sara-demo ask "What is X?" --brain demo.db --compare llama3.2:1b
    sara-demo show --brain demo.db --concept "molecular snare"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sara_brain.core.brain import Brain
from sara_brain.core.wavefront_renderer import render_wavefront_facts
from sara_reader.stateless_reader import _extract_seed_concepts, _filter_seeds_by_substrate


# ANSI colors for terminal
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def cmd_teach(args):
    """Interactive teaching mode."""
    brain = Brain(args.brain)
    print(f"{BOLD}Sara Brain — Teaching Mode{RESET}")
    print(f"Brain: {args.brain} ({brain.stats()['neurons']} neurons)")
    print(f"Type facts as: subject | relation | object")
    print(f"Or plain sentences (auto-parsed as subject verb object)")
    print(f"Ctrl-D to finish.\n")

    taught = 0
    try:
        while True:
            line = input(f"{CYAN}teach>{RESET} ").strip()
            if not line:
                continue
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) == 3:
                    s, r, o = parts
                else:
                    print("  Format: subject | relation | object")
                    continue
            else:
                # Auto-parse: first word(s) before a verb = subject, verb = relation, rest = object
                words = line.split()
                if len(words) < 3:
                    print("  Need at least 3 words (subject verb object)")
                    continue
                # Simple heuristic: second word is the relation
                s = words[0]
                r = words[1]
                o = " ".join(words[2:])

            t0 = time.time()
            brain.teach_triple(s, r, o, source_text=line)
            dt = (time.time() - t0) * 1000
            taught += 1
            print(f"  {GREEN}✓ taught{RESET} ({dt:.0f}ms) — {s} → {r} → {o}")
    except (EOFError, KeyboardInterrupt):
        pass

    print(f"\n{taught} facts taught. Brain: {brain.stats()['neurons']} neurons.")
    brain.close()


def cmd_ask(args):
    """Ask Sara a question, optionally compare with a 1B model."""
    brain = Brain(args.brain)

    # Get seeds from question
    candidates = _extract_seed_concepts(args.question)
    seeds = _filter_seeds_by_substrate(brain, candidates)
    if not seeds:
        seeds = candidates[:3]

    # Run wavefront + render
    t0 = time.time()
    facts = render_wavefront_facts(brain, seeds, depth=2, max_facts=15)
    sara_time = time.time() - t0

    # Display Sara's answer
    fact_lines = [l.strip("- ").strip() for l in facts.split("\n") if l.strip().startswith("- ")]

    print(f"\n{BOLD}Question:{RESET} {args.question}\n")
    print(f"{GREEN}{BOLD}Sara Brain says:{RESET}")
    if fact_lines:
        for f in fact_lines[:8]:
            print(f"  {GREEN}•{RESET} {f}")
        print(f"  {CYAN}({len(fact_lines)} facts, {sara_time*1000:.0f}ms, {brain.stats()['paths']} paths in brain){RESET}")
    else:
        print(f"  {CYAN}I don't have enough information about this topic.{RESET}")
        print(f"  {CYAN}Teach me with: sara-demo teach --brain {args.brain}{RESET}")

    # Compare with LLM if requested
    if args.compare:
        print(f"\n{RED}{BOLD}{args.compare} says:{RESET}")
        try:
            prompt = f"{args.question}\nAnswer concisely."
            body = json.dumps({"model": args.compare, "prompt": prompt, "stream": False}).encode()
            req = urllib.request.Request(
                f"http://{args.ollama_host}/api/generate",
                data=body, headers={"Content-Type": "application/json"})
            t0 = time.time()
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            llm_time = time.time() - t0
            print(f"  {RED}{resp['response'].strip()[:200]}{RESET}")
            print(f"  {CYAN}({llm_time*1000:.0f}ms, source: training weights, not inspectable){RESET}")
        except Exception as e:
            print(f"  {RED}(Ollama not available: {e}){RESET}")

    # Show path trace
    if fact_lines and args.trace:
        print(f"\n{BOLD}Path trace:{RESET}")
        for f in fact_lines[:5]:
            print(f"  {f}")

    brain.close()


def cmd_show(args):
    """Show what Sara knows about a concept."""
    brain = Brain(args.brain)
    facts = render_wavefront_facts(brain, [args.concept], depth=2, max_facts=20)
    print(f"\n{BOLD}Sara Brain — what I know about '{args.concept}':{RESET}\n")
    for line in facts.split("\n"):
        if line.strip().startswith("- "):
            print(f"  {GREEN}•{RESET} {line.strip('- ').strip()}")
        elif line.strip():
            print(f"  {CYAN}{line.strip()}{RESET}")
    print(f"\n  Brain stats: {brain.stats()}")
    brain.close()


def main():
    ap = argparse.ArgumentParser(prog="sara-demo", description="Sara Brain live demo")
    sub = ap.add_subparsers(dest="cmd")

    # teach
    p = sub.add_parser("teach", help="Interactive teaching mode")
    p.add_argument("--brain", default="demo.db", help="Brain database path")

    # ask
    p = sub.add_parser("ask", help="Ask Sara a question")
    p.add_argument("question", help="The question to ask")
    p.add_argument("--brain", default="demo.db", help="Brain database path")
    p.add_argument("--compare", default=None, help="Compare with Ollama model (e.g. llama3.2:1b)")
    p.add_argument("--ollama-host", default="localhost:11434", help="Ollama host")
    p.add_argument("--trace", action="store_true", help="Show path trace")

    # show
    p = sub.add_parser("show", help="Show what Sara knows about a concept")
    p.add_argument("concept", help="Concept to explore")
    p.add_argument("--brain", default="demo.db", help="Brain database path")

    args = ap.parse_args()
    if args.cmd == "teach":
        cmd_teach(args)
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "show":
        cmd_show(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
