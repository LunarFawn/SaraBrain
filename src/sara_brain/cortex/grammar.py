"""Grammar rules and vocabulary for the Sara Cortex.

This module is pure data — verb categories, question patterns, modal
markers, source markers, temporal markers. It is the raw vocabulary
that the parser and generator use to understand and produce English.

Adding a new pattern means appending a string to a frozenset. No
training, no fine-tuning, no GPU. The cortex grows by editing this
file.
"""

from __future__ import annotations

# ── Question patterns ──
# A turn starts with one of these → it is a question, not a statement.

QUESTION_PREFIXES = frozenset({
    "what", "what's", "whats",
    "who", "who's", "whos", "whom", "whose",
    "where", "where's", "wheres",
    "when", "when's", "whens",
    "why", "why's", "whys",
    "how", "how's", "hows",
    "which",
    "is", "are", "was", "were",
    "do", "does", "did",
    "can", "could", "would", "should", "may", "might", "must", "shall",
    "tell", "show", "explain", "describe", "define", "list",
})

# ── Statement-ending punctuation ──
QUESTION_MARKS = frozenset({"?"})


# ── Negation markers ──
# Words that, when they appear right after the verb, flip a teach into a refute.

NEGATION_WORDS = frozenset({
    "not", "no", "never", "n't", "nothing", "nobody", "neither", "nor",
})


# ── Statement separators ──
# Patterns that split a compound user input into multiple sub-statements.

CONJUNCTIONS = (
    " and ", " but ", " however ", " whereas ", " while ",
    "; ", ". ",
)

# Stronger sentence boundaries. We split on these even without leading space.
SENTENCE_TERMINATORS = (". ", "! ", "? ")


# ── Source / authority markers ──
# Phrases that indicate the user is citing a source. The cortex
# attaches these as provenance metadata when teaching.

SOURCE_PHRASES = (
    "according to ", "as stated in ", "as written in ",
    "from the wikipedia article on ", "wikipedia says ",
    "the textbook says ", "the paper says ", "the doctor said ",
    "i read in ", "i read that ", "i learned that ",
    "the source is ", "from ", "based on ",
)


# ── Quantifier / strength markers ──
# Modify the trust weight of an extracted statement.

QUANTIFIERS_HIGH = frozenset({
    "always", "definitely", "certainly", "absolutely", "clearly",
    "obviously", "undoubtedly", "surely", "indeed",
})

QUANTIFIERS_LOW = frozenset({
    "maybe", "perhaps", "possibly", "probably", "might", "could",
    "sometimes", "occasionally", "supposedly", "allegedly",
})

QUANTIFIERS_NEGATING = frozenset({
    "rarely", "seldom", "hardly", "barely",
})


# ── Temporal markers ──
# Phrases that anchor a statement in time. Used by the temporal layer
# (already present in core/temporal.py) to attach date neurons.

TEMPORAL_RELATIVE = frozenset({
    "today", "yesterday", "tomorrow", "tonight",
    "now", "currently", "lately", "recently",
})

TEMPORAL_ANCHORED = (
    "in 19", "in 20",
    "during the ", "in the ",
    "before ", "after ",
    "since ",
)


# ── Articles and stop words ──
# Stripped during parsing to focus on content words.

ARTICLES = frozenset({"a", "an", "the"})

STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "and", "or", "but", "this", "that", "these", "those",
    "it", "its", "i", "you", "we", "they", "them", "us", "him", "her",
    "his", "hers", "their", "our", "your", "my", "me",
})


# ── Output templates ──
# Used by the generator to render Sara's grounded paths into English.
# Each template is a function or format string. We start simple — a
# single template per relation type — and add more variety later.

TEMPLATES_BY_RELATION = {
    "is_a":          "{subject} is {object}.",
    "is_not_a":      "{subject} is not {object}.",
    "has":           "{subject} has {object}.",
    "has_color":     "{subject} is {object}.",
    "has_shape":     "{subject} is {object}.",
    "has_size":      "{subject} is {object}.",
    "has_taste":     "{subject} tastes {object}.",
    "has_texture":   "{subject} feels {object}.",
    "contains":      "{subject} contains {object}.",
    "includes":      "{subject} includes {object}.",
    "requires":      "{subject} requires {object}.",
    "follows":       "{subject} follows {object}.",
    "precedes":      "{subject} precedes {object}.",
    "excludes":      "{subject} excludes {object}.",
    "happened_on":   "{subject} happened on {object}.",
    "happened_during": "{subject} happened during {object}.",
}

# Fallback template when relation isn't in the registry
DEFAULT_TEMPLATE = "{subject} {relation} {object}."


# ── Entity disambiguation context clues ──
# Used by entity_resolver.py to determine if a term refers to a place,
# people/ethnicity, or language. When context clues are detected, the
# entity resolver qualifies the neuron label to prevent conflation.

PLACE_CLUES = frozenset({
    "in", "at", "from", "near", "of", "located",
    "region", "city", "area", "territory", "land", "country",
    "kingdom", "empire", "coast", "river", "valley", "plain",
    "mesopotamia", "geography", "border", "capital",
    "lived in", "located in", "region of", "city of", "part of",
})

PEOPLE_CLUES = frozenset({
    "people", "peoples", "civilization", "culture", "society",
    "tribe", "ethnic", "ethnicity", "population", "inhabitants",
    "citizens", "nation", "community", "group",
    "they", "their", "them", "who",
    "built", "created", "invented", "developed", "established",
    "conquered", "ruled", "governed", "migrated", "settled",
})

LANGUAGE_CLUES = frozenset({
    "language", "tongue", "dialect", "script", "writing",
    "cuneiform", "alphabet", "grammar", "vocabulary", "word",
    "spoke", "spoken", "speak", "written", "wrote", "write",
    "read", "taught", "learned", "translated", "text",
    "literature", "poem", "epic", "hymn", "prayer", "tablet",
})


# ── Multi-fact response templates ──
# When Sara has multiple paths about a topic, the generator joins them.

LIST_INTRO = "Here is what Sara knows about {topic}:"
LIST_ITEM = "  • {sentence}"
LIST_REFUTED_PREFIX = "  • [refuted, no longer believed] {sentence}"

NO_KNOWLEDGE = "Sara has no knowledge of {topic}."
NO_KNOWLEDGE_WITH_HINT = (
    "Sara has no knowledge of {topic}. "
    "Tell me about it in a sentence (e.g., '{topic} is ...') and I'll learn."
)

CONFIRM_TAUGHT = "Learned: {fact}."
CONFIRM_REFUTED = "Marked as known-to-be-false: {fact}."
CONFIRM_TAUGHT_MULTI = "Learned {count} facts."
CONFIRM_PARSE_FAIL = (
    "I couldn't parse '{text}' as a fact. "
    "Try the form 'X is Y' or 'X has Y' or 'X requires Y'."
)
