"""CLI entry point for sara-care.

Usage:
    sara-care [--db PATH] [--model MODEL] [--url URL]
"""

from __future__ import annotations

import argparse
import sys

from ..agent import ollama
from ..agent.cli import TOOL_CAPABLE_MODELS, _pick_model
from ..config import default_db_path
from ..core.brain import Brain
from .accounts import select_account, setup_wizard
from .loop import CareLoop


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sara-care",
        description="Sara Care — Memory assistant for dementia care",
    )
    parser.add_argument(
        "-m", "--model",
        help="Ollama model name (default: auto-detect)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Brain database path (default: {default_db_path()})",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Run first-time account setup",
    )
    args = parser.parse_args()

    # 1. Check Ollama
    if not ollama.check_health(args.url):
        print("\n  Sara Care needs Ollama running locally.", file=sys.stderr)
        print("  Start it with: ollama serve", file=sys.stderr)
        print("  Then pull a model: ollama pull llama3.1", file=sys.stderr)
        sys.exit(1)

    # 2. Select model
    models = ollama.list_models(args.url)
    if not models:
        print("\n  No models available.", file=sys.stderr)
        print("  Pull one with: ollama pull llama3.1", file=sys.stderr)
        sys.exit(1)

    if args.model:
        model_names = [m.get("name", "") for m in models]
        matched = [n for n in model_names if args.model in n]
        if not matched:
            print(f"\n  Model '{args.model}' not found.", file=sys.stderr)
            sys.exit(1)
        model = matched[0]
    else:
        model = _pick_model(models)
        if model is None:
            print("\n  Could not auto-select a model.", file=sys.stderr)
            sys.exit(1)

    # 3. Open Brain
    db_path = args.db or default_db_path()
    brain = Brain(db_path)

    try:
        # 4. Setup or select account
        if args.setup or not brain.account_repo.list_all():
            accounts = setup_wizard(brain)
            if not accounts:
                print("\n  Setup cancelled.")
                return

        account = select_account(brain)
        if account is None:
            # New account requested
            print("\n  Let's set you up.")
            name = input("  Your name: ").strip()
            if not name:
                return
            print("  What is your role?")
            print("  1. I'm the person Sara helps (patient)")
            print("  2. I'm a family member")
            print("  3. I'm a doctor or caregiver")
            role_choice = input("  > ").strip()
            role_map = {"1": "reader", "2": "teacher", "3": "doctor"}
            role = role_map.get(role_choice, "teacher")

            if role == "reader":
                from .accounts import create_patient
                account = create_patient(brain, name)
            elif role == "doctor":
                from .accounts import create_doctor
                account = create_doctor(brain, name)
            else:
                patient = brain.account_repo.get_reader()
                patient_name = patient.name if patient else ""
                relationship = input(f"  Your relationship to {patient_name}: ").strip() or "family"
                from .accounts import create_family
                account = create_family(brain, name, relationship, patient_name)

        # 5. Banner
        print()
        print("  Sara Care — Memory Assistant")
        print(f"  Model: {model} @ {args.url}")
        stats = brain.stats()
        print(f"  Brain: {stats['neurons']} neurons, {stats['paths']} paths")
        print(f"  Logged in as: {account.name} ({account.role})")

        # 6. Run the care loop
        care = CareLoop(
            brain=brain,
            account=account,
            model=model,
            base_url=args.url,
        )
        care.run_interactive()

    finally:
        brain.close()


if __name__ == "__main__":
    main()
