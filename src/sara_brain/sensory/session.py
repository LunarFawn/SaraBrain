"""Session state for the sensory shell.

Tracks conversation context across multiple turns without mutating
the long-term graph. Uses Sara's ShortTerm scratchpad under the hood.
"""

from __future__ import annotations

from collections import deque


class Session:
    """Conversation context for the sensory shell.

    Tracks recent topics so follow-up questions have context.
    Does not persist — lives only for the current session.
    Does not mutate the graph.
    """

    def __init__(self, max_history: int = 20) -> None:
        self.recent_topics: deque[str] = deque(maxlen=max_history)
        self.turn_count: int = 0

    def add_turn(self, tokens: list[str]) -> None:
        """Record the tokens from one turn."""
        self.turn_count += 1
        for t in tokens:
            if t not in self.recent_topics:
                self.recent_topics.append(t)

    def context_seeds(self) -> list[str]:
        """Return recent topics as additional wavefront seeds.

        These provide conversational context: if the user asked about
        methane last turn and now says "how many hydrogen atoms",
        methane is still in the seed pool.
        """
        return list(self.recent_topics)

    def clear(self) -> None:
        """Reset session state."""
        self.recent_topics.clear()
        self.turn_count = 0
