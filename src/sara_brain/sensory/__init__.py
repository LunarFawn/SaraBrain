"""Sara Sensory Shell — an empty processing engine over Sara Brain.

The shell has no knowledge, no weights, no training. It is a thin
processing layer that tokenizes input, feeds wavefront seeds into
Sara's graph, runs parallel propagation, and renders converging
paths as output with full provenance.

Sara Brain IS the weight store. The shell IS the transformer.
Teaching a fact is adding a weight. Every answer traces to specific
taught facts. No black box.

Usage:
    from sara_brain.core.brain import Brain
    from sara_brain.sensory import SensoryShell

    brain = Brain("sara.db")
    shell = SensoryShell(brain)

    # Process input — all answers come from Sara's paths
    response = shell.process("what has one carbon atom")
    print(response.text)  # traceable output

    # Direct query
    response = shell.query("methane")
    for src in response.sources:
        print(f"  path #{src.path_id}: {src.source_text}")
"""

from .shell import SensoryShell, ShellResponse
from .tokenizer import Tokenizer, Token
from .renderer import Renderer, SourcedLine
from .session import Session

__all__ = [
    "SensoryShell",
    "ShellResponse",
    "Tokenizer",
    "Token",
    "Renderer",
    "SourcedLine",
    "Session",
]
