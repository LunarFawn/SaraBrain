"""cmd.Cmd-based interactive REPL for Sara Brain."""

from __future__ import annotations

import cmd
import sys

from ..core.brain import Brain, _BUILTIN_QUESTION_WORDS
from . import commands


class SaraShell(cmd.Cmd):
    intro = "Sara Brain v0.1.0 — Path-of-thought brain simulation\nType 'help' for commands.\n"
    prompt = "sara> "

    def __init__(self, brain: Brain) -> None:
        super().__init__()
        self.brain = brain

    def do_teach(self, args: str) -> None:
        """Teach a fact: teach apples are red"""
        print(commands.cmd_teach(self.brain, args))

    def do_recognize(self, args: str) -> None:
        """Recognize from inputs: recognize red, round"""
        print(commands.cmd_recognize(self.brain, args))

    def do_why(self, args: str) -> None:
        """Show paths leading to a concept: why apple"""
        print(commands.cmd_why(self.brain, args))

    def do_trace(self, args: str) -> None:
        """Trace paths from a neuron: trace red"""
        print(commands.cmd_trace(self.brain, args))

    def do_neurons(self, args: str) -> None:
        """List all neurons"""
        print(commands.cmd_neurons(self.brain, args))

    def do_paths(self, args: str) -> None:
        """List all recorded paths"""
        print(commands.cmd_paths(self.brain, args))

    def do_stats(self, args: str) -> None:
        """Show brain statistics"""
        print(commands.cmd_stats(self.brain, args))

    def do_similar(self, args: str) -> None:
        """Show neurons similar to given neuron: similar red"""
        print(commands.cmd_similar(self.brain, args))

    def do_analyze(self, args: str) -> None:
        """Analyze path similarities across all neurons"""
        print(commands.cmd_analyze(self.brain, args))

    def do_define(self, args: str) -> None:
        """Define a new association type: define taste how"""
        print(commands.cmd_define(self.brain, args))

    def do_describe(self, args: str) -> None:
        """Register properties under an association: describe mood as happy, sad"""
        print(commands.cmd_describe(self.brain, args))

    def do_associations(self, args: str) -> None:
        """List all defined associations and their properties"""
        print(commands.cmd_associations(self.brain, args))

    def do_questions(self, args: str) -> None:
        """List all available question words"""
        print(commands.cmd_questions(self.brain, args))

    def do_categorize(self, args: str) -> None:
        """Tag a concept with a category: categorize apple item"""
        print(commands.cmd_categorize(self.brain, args))

    def do_categories(self, args: str) -> None:
        """List all categories and their members"""
        print(commands.cmd_categories(self.brain, args))

    def do_ask(self, args: str) -> None:
        """Ask a natural language question (requires LLM config): ask what does an apple taste like?"""
        if not args.strip():
            print("  Usage: ask <natural language question>")
            return

        translator = self._get_translator()
        if translator is None:
            print("  No LLM configured. Use structured commands (how, what, where, etc.)")
            print('  Or configure with: llm set <api_url> <api_key> <model>')
            return

        # Build available commands list for the LLM
        qwords = self.brain.list_question_words()
        available = ["teach <subject> is/are <property>"]
        for qword, assocs in sorted(qwords.items()):
            for assoc in sorted(assocs):
                available.append(f"{qword} <concept> {assoc}")
        available.append("categorize <concept> <category>")

        translated = translator.translate(args, available)
        if translated is None:
            print("  Could not translate query. Try using structured commands.")
            return

        print(f"  → {translated}")
        # Dispatch the translated command through the REPL
        self.onecmd(translated)

    def do_llm(self, args: str) -> None:
        """Configure LLM for natural language translation.
        llm set <api_url> <api_key> <model>
        llm status
        llm clear"""
        parts = args.strip().split()
        if not parts:
            print("  Usage: llm set <api_url> <api_key> <model>")
            print("         llm status")
            print("         llm clear")
            return

        subcmd = parts[0].lower()
        if subcmd == "set":
            if len(parts) < 4:
                print("  Usage: llm set <api_url> <api_key> <model>")
                return
            api_url, api_key, model = parts[1], parts[2], parts[3]
            self.brain.settings_repo.set("llm_api_url", api_url)
            self.brain.settings_repo.set("llm_api_key", api_key)
            self.brain.settings_repo.set("llm_model", model)
            self.brain.conn.commit()
            print(f"  LLM configured: {model} @ {api_url}")
        elif subcmd == "status":
            url = self.brain.settings_repo.get("llm_api_url")
            model = self.brain.settings_repo.get("llm_model")
            if url and model:
                print(f"  LLM: {model} @ {url}")
            else:
                print("  No LLM configured.")
        elif subcmd == "clear":
            self.brain.settings_repo.delete("llm_api_url")
            self.brain.settings_repo.delete("llm_api_key")
            self.brain.settings_repo.delete("llm_model")
            self.brain.conn.commit()
            print("  LLM configuration cleared.")
        else:
            print("  Unknown subcommand. Use: set, status, clear")

    def _get_translator(self):
        """Return an LLMTranslator if configured, else None."""
        url = self.brain.settings_repo.get("llm_api_url")
        key = self.brain.settings_repo.get("llm_api_key")
        model = self.brain.settings_repo.get("llm_model")
        if not url or not model:
            return None
        from ..nlp.translator import LLMTranslator
        return LLMTranslator(url, key or "", model)

    def do_save(self, args: str) -> None:
        """Force flush to disk"""
        self.brain.conn.commit()
        print("  Saved.")

    def do_quit(self, args: str) -> bool:
        """Exit the REPL"""
        self.brain.conn.commit()
        print("  Goodbye.")
        return True

    def do_exit(self, args: str) -> bool:
        """Exit the REPL"""
        return self.do_quit(args)

    do_EOF = do_quit

    def default(self, line: str) -> None:
        parts = line.split(None, 1)
        word = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        # Check if it's a registered question word (DB first, then builtins)
        associations = self.brain.resolve_question_word(word)
        if associations:
            print(commands.cmd_query(self.brain, word, args))
            return

        print(f'  Unknown command: "{word}". Type "help" for commands.')

    def emptyline(self) -> None:
        pass


def main() -> None:
    db_path = "sara.db"
    if len(sys.argv) > 1:
        db_path = sys.argv[1]

    print(f"  Using database: {db_path}")
    with Brain(db_path) as brain:
        shell = SaraShell(brain)
        try:
            shell.cmdloop()
        except KeyboardInterrupt:
            print("\n  Goodbye.")


if __name__ == "__main__":
    main()
