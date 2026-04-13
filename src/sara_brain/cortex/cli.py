"""sara-cortex CLI — interactive shell that uses Sara Cortex first.

The cortex handles whatever it can. For things it can't handle (low
confidence, unparseable input), it falls through to the existing
sara-agent loop with Ollama as the fallback cortex.

This is the transition path: the cortex starts handling 80% of common
turns immediately, and the LLM only kicks in for the long tail. As the
cortex grammar grows, the LLM is consulted less and less. Eventually
the LLM disappears entirely.

Usage:
    sara-cortex                           # interactive
    sara-cortex --db /path/to/brain.db    # custom database
    sara-cortex --no-llm                  # cortex only, no fallback
    sara-cortex --model llama3.1          # llama fallback model
"""

from __future__ import annotations

import argparse
import sys

from ..config import default_db_path
from ..core.brain import Brain
from .router import Cortex


def _print_response(response, verbose: bool = False) -> None:
    print(f"\nsara> {response.text}\n")
    if verbose and response.operations:
        for op in response.operations:
            mark = "✓" if op.success else "✗"
            print(f"      [{mark} {op.op}] {op.target} {op.detail}")


def _handle_slash(brain, cortex, command: str) -> bool:
    """Handle a slash command. Returns True if the command was recognized
    and handled, False otherwise (so the input falls through to the cortex)."""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/help", "/?"):
        print(
            "\n  Slash commands:\n"
            "    /teach <fact>       — teach a fact directly\n"
            "    /refute <fact>      — refute a fact directly\n"
            "    /ingest <source>    — ingest a file or URL into Sara\n"
            "    /cluster <word>     — show cluster around a concept\n"
            "    /cleanup            — interactive brain cleanup (per-item review)\n"
            "    /scan               — read-only pollution scan\n"
            "    /stats              — brain statistics\n"
            "    /help               — this message\n"
        )
        return True

    if cmd == "/stats":
        s = brain.stats()
        print(
            f"\n  Neurons: {s['neurons']}\n"
            f"  Segments: {s['segments']}\n"
            f"  Paths: {s['paths']}\n"
        )
        return True

    if cmd == "/scan":
        from ..agent.bridge import AgentBridge
        print(f"\n{AgentBridge(brain).scan_pollution()}\n")
        return True

    if cmd == "/cluster":
        if not arg:
            print("\n  Usage: /cluster <word>\n")
            return True
        cluster = brain.cluster_around(arg)
        if not cluster:
            print(f"\n  Sara has no cluster for {arg!r}.\n")
            return True
        print(f"\n  Cluster for {arg!r}:")
        for c in cluster[:20]:
            print(f"    [{c['connections']:2d}] {c['label']!r} ({c['type']}, {c['hops']} hop)")
        print()
        return True

    if cmd == "/teach":
        if not arg:
            print("\n  Usage: /teach <fact>\n")
            return True
        result = brain.teach(arg)
        if result is None:
            print(f"\n  Could not parse: {arg!r}. Try 'X is Y' format.\n")
        else:
            brain.conn.commit()
            print(f"\n  Learned: {result.path_label}\n")
        return True

    if cmd == "/refute":
        if not arg:
            print("\n  Usage: /refute <fact>\n")
            return True
        result = brain.refute(arg)
        if result is None:
            print(f"\n  Could not parse: {arg!r}. Try 'X is Y' format.\n")
        else:
            brain.conn.commit()
            print(f"\n  Refuted: {result.path_label}\n")
        return True

    if cmd == "/ingest":
        if not arg:
            print("\n  Usage: /ingest <file_or_url>\n")
            print("  Examples:")
            print("    /ingest https://en.wikipedia.org/wiki/Sumer")
            print("    /ingest /path/to/document.txt\n")
            return True
        print(f"\n  Ingesting: {arg}")
        print("  (this may take a minute — the LLM extracts facts from the document)\n")
        try:
            from ..agent.bridge import AgentBridge
            bridge = AgentBridge(brain)
            result = bridge.ingest(arg)
            print(f"  {result}\n")
        except Exception as e:
            print(f"  Error: {e}\n")
        return True

    if cmd == "/cleanup":
        # Run the same interactive cleanup as the standalone CLI
        from .cleanup import (
            find_article_typo_neurons,
            find_pronoun_neurons,
            find_question_word_typos,
            find_stopword_subject_neurons,
            find_sentence_subject_neurons,
            find_punctuation_artifact_neurons,
            _review_category,
        )
        print()
        print("  Brain cleanup — per-item review (Sara never bulk-refutes)")
        print()

        article_typos = find_article_typo_neurons(brain)
        pronouns = find_pronoun_neurons(brain)
        question_typos = find_question_word_typos(brain)
        stopwords = find_stopword_subject_neurons(brain)
        sentences = find_sentence_subject_neurons(brain)
        punct = find_punctuation_artifact_neurons(brain)

        total = (
            len(article_typos) + len(pronouns) + len(question_typos)
            + len(stopwords) + len(sentences) + len(punct)
        )
        if total == 0:
            print("  No pollution candidates found. Brain is clean.\n")
            return True

        print(f"  Found {total} pollution candidates across categories:")
        print(f"    article-typo:        {len(article_typos)}")
        print(f"    pronoun:             {len(pronouns)}")
        print(f"    question-word typo:  {len(question_typos)}")
        print(f"    stopword subject:    {len(stopwords)}")
        print(f"    sentence subject:    {len(sentences)}")
        print(f"    punctuation artifact: {len(punct)}")
        print()
        confirm = input("  Start review? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("  Cleanup cancelled.\n")
            return True

        _review_category(
            brain, article_typos, "article-typo",
            "May be a typo OR a real word in your dialect.",
        )
        _review_category(
            brain, pronouns, "pronoun-subject",
            "Pronouns can never be standalone subjects — old parser bugs.",
        )
        _review_category(
            brain, question_typos, "question-word typo",
            "Typos of question words (waht/hwat) that became subjects.",
        )
        _review_category(
            brain, stopwords, "stopword-subject",
            "Bare stopwords (not/it/like) should never be standalone subjects.",
        )
        # Sentence-subjects and punctuation-artifacts are summary-only.
        # Too numerous and noisy for per-item review. The parser now
        # rejects subjects >4 words so new ones won't be created.
        if sentences:
            print(f"\n  {len(sentences)} sentence-subjects (summary only — old digester artifacts).")
        if punct:
            print(f"\n  {len(punct)} punctuation-artifacts (summary only).")

        s = brain.stats()
        print()
        print(f"  Cleanup complete. Brain: {s['neurons']} neurons, {s['paths']} paths.")
        print()
        return True

    return False  # Not a recognized slash command


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sara-cortex",
        description="Sara Cortex — language layer for Sara Brain. "
                    "The cortex handles natural language directly. "
                    "Ollama is consulted only when the cortex defers.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Brain database path (default: {default_db_path()})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Cortex only — never fall back to Ollama. "
             "Sara will say 'I don't know' instead of guessing.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model for fallback (default: auto-detect)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show cortex operations after each turn",
    )
    args = parser.parse_args()

    db_path = args.db or default_db_path()
    brain = Brain(db_path)
    cortex = Cortex(brain)

    # Auto-configure LLM settings if the brain has none and Ollama is running.
    # New brains start with empty settings — this saves the user from having
    # to manually configure every new .db file.
    if not args.no_llm and not brain.settings_repo.get("llm_provider"):
        try:
            from ..agent import ollama as _ollama_check
            if _ollama_check.check_health(args.url):
                model_name = args.model or "qwen2.5-coder:3b"
                brain.settings_repo.set("llm_provider", "ollama")
                brain.settings_repo.set("llm_model", model_name)
                brain.settings_repo.set("llm_api_url", args.url)
                brain.conn.commit()
        except Exception:
            pass

    # Optionally set up the llama fallback
    fallback_loop = None
    if not args.no_llm:
        try:
            from ..agent import ollama
            from ..agent.cli import _pick_model, TOOL_CAPABLE_MODELS
            from ..agent.loop import AgentLoop
            if ollama.check_health(args.url):
                models = ollama.list_models(args.url)
                if models:
                    if args.model:
                        match = [m.get("name", "") for m in models if args.model in m.get("name", "")]
                        model = match[0] if match else None
                    else:
                        model = _pick_model(models)
                    if model:
                        fallback_loop = AgentLoop(
                            brain=brain,
                            model=model,
                            base_url=args.url,
                        )
        except Exception as e:
            print(f"  (Ollama fallback unavailable: {e})", file=sys.stderr)

    stats = brain.stats()
    print()
    if args.no_llm:
        print("  Sara Cortex — rule-based mode (no LLM)")
        print(f"  Brain: {db_path} ({stats['neurons']} neurons, {stats['paths']} paths)")
        print("  LLM: none — cortex handles I/O directly")
    else:
        model_name = fallback_loop.model if fallback_loop else "unavailable"
        print("  Sara Brain — LLM as ears and mouth, Sara as the brain")
        print(f"  Brain: {db_path} ({stats['neurons']} neurons, {stats['paths']} paths)")
        print(f"  LLM: {model_name} (sensory + motor cortex)")
    print()
    print("  Type 'exit' to quit. Slash commands: /help")
    print()

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye. Sara remembers everything.")
                break

            if not user_input:
                continue
            # Slash commands always bypass both LLM and cortex
            if user_input.startswith("/"):
                if _handle_slash(brain, cortex, user_input):
                    continue
            if user_input.lower() in ("exit", "quit", "bye"):
                print("  Goodbye. Sara remembers everything.")
                break

            try:
                if fallback_loop is not None and not args.no_llm:
                    # ── LLM MODE (default) ──
                    # The LLM is the entry point (ears) and exit point (mouth).
                    # Sara Brain is the internal machinery consulted via tools.
                    # The agent loop's _sara_turn auto-teaches declarative input,
                    # the system prompt forces the LLM to defer to Sara's paths,
                    # and the LLM renders the grounded output as fluent English.
                    text = fallback_loop.turn(user_input)
                    print(f"\nsara> {text}\n")
                else:
                    # ── CORTEX-ONLY MODE (--no-llm) ──
                    # Rule-based cortex handles everything directly.
                    # Less fluent output but zero hallucination possible.
                    response = cortex.process(user_input)

                    if response.requires_disambiguation:
                        print(f"\nsara> {response.text}\n")
                        print(
                            "  How do you want to handle this?\n"
                            "    [c]orrect your spelling and re-enter\n"
                            "    [n]ew concept — keep the new term as a separate neuron\n"
                            "    [s]kip — discard this teaching\n"
                        )
                        choice = input("  > ").strip().lower()
                        if choice == "n":
                            old = cortex.strict_safety
                            cortex.strict_safety = False
                            for fact in response.parsed_turn.facts:
                                stmt = fact.original_text or user_input
                                if fact.negated:
                                    cortex.brain.refute(stmt)
                                else:
                                    cortex.brain.teach(stmt)
                            cortex.brain.conn.commit()
                            cortex.strict_safety = old
                            print("\n  Committed as a new concept.\n")
                        elif choice == "c":
                            print("  Try again with the corrected spelling.\n")
                        else:
                            print("  Skipped.\n")
                        continue

                    _print_response(response, verbose=args.verbose)
            except Exception as e:
                print(f"\n  Error: {e}\n", file=sys.stderr)

    finally:
        brain.close()


if __name__ == "__main__":
    main()
