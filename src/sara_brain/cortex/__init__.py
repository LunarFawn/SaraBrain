"""Sara Cortex — purpose-built language layer for Sara Brain.

The cortex is the language I/O of Sara. It does NOT hold knowledge.
It does NOT have opinions. It has two jobs:

1. SENSORY:  natural language → structured Sara operations
2. MOTOR:    Sara's grounded paths → fluent English

This module is the architectural commitment to "the LLM is the senses,
not the brain" taken to its conclusion. The cortex has no training data
about the world — only grammar, vocabulary, and templates. When asked
about something, the cortex MUST query Sara because it has no other
source of facts. Hallucination becomes structurally impossible.

Phase 1 (today): rule-based enhanced parser + template-based generator.
Phase 2 (later): tiny learned grammar model trained on pure grammar pairs.
Phase 3 (vision): replace Ollama entirely. Sara on a Pi with no external
LLM, just sara_brain + sara_cortex + whisper.cpp. A self-contained brain.

The cortex exposes a simple API:

    from sara_brain.cortex import Cortex
    from sara_brain.core.brain import Brain

    cortex = Cortex(Brain("sara.db"))
    response = cortex.process("what is the edubba")
    response = cortex.process("the edubba was a sumerian school")

The cortex returns a CortexResponse with:
    - text:       the response to show the user
    - operations: list of brain operations performed (teach/refute/query)
    - confidence: how confident the cortex is in its handling
    - delegate:   if True, the caller should fall back to a larger LLM
"""

from .parser import EnhancedParser, ParsedTurn, TurnKind
from .generator import TemplateGenerator
from .router import Cortex, CortexResponse

__all__ = [
    "Cortex",
    "CortexResponse",
    "EnhancedParser",
    "ParsedTurn",
    "TurnKind",
    "TemplateGenerator",
]
