"""cmd.Cmd-based interactive REPL for Sara Brain."""

from __future__ import annotations

import cmd
import sys

from ..core.brain import Brain
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
        print(f"  Unknown command: {line.split()[0]}. Type 'help' for commands.")

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
