"""Prompt templates for the teacher cascade.

Mirrors docs/teacher_cascade_prompts_draft.md. Each template has a
single {sentence} placeholder for the source input, except the
integrator/synthesizer prompts which take multiple fields.

Every prompt ends with the same two-line hard rule:
  - Use ONLY content from the source.
  - If nothing matches, output the single word NONE.
"""
from __future__ import annotations


_RULES = (
    "Rules:\n"
    "- Use ONLY content from the source. Do not add, infer, summarize, "
    "or generalize.\n"
    "- If the source contains nothing matching the request, output the "
    "single word NONE."
)


# Layer 1 — Claim sensors (1B)

CLAIM_DEFINITION = f"""You are extracting factual definitions from a source sentence.
A definition says "X is Y" or "X means Y" or "X is a type of Y".

Source sentence:
{{sentence}}

Output each definition as one line in the form:
  a <thing> is a <category>
  or
  <thing> is <property>

{_RULES}"""


CLAIM_PROCESS = f"""You are extracting process claims from a source sentence.
A process claim says "X does Y" or "X produces Y" or "X includes Y".

Source sentence:
{{sentence}}

Output each process claim as one line in the form:
  <subject> <verb> <object>

Use only these verbs: produces, contains, includes, requires,
precedes, follows, causes, prevents, divides.

{_RULES}"""


CLAIM_CAUSATION = f"""You are extracting cause-and-effect claims from a source sentence.
A causation claim says "X causes Y" or "X prevents Y" or "X requires Y".

Source sentence:
{{sentence}}

Output each causation claim as one line in the form:
  <cause> causes <effect>
  <condition> requires <prerequisite>
  <agent> prevents <outcome>

{_RULES}"""


CLAIM_TEMPORAL = f"""You are extracting temporal claims from a source sentence.
A temporal claim says "X happens before Y", "X happens after Y",
"X happens during Y", or "X is a phase of Y".

Source sentence:
{{sentence}}

Output each temporal claim as one line in the form:
  <earlier> precedes <later>
  <later> follows <earlier>
  <event> during <phase>
  <phase> is a phase of <process>

{_RULES}"""


CLAIM_DATETIME = f"""You are extracting date and time claims from a source sentence.
A date/time claim attaches an absolute time reference to an event —
a year, date, era, period, or duration.

Examples of time references: "1953", "March 2024", "the Jurassic
period", "19th century", "24 hours", "3 billion years ago".

Source sentence:
{{sentence}}

Output each dated claim as one line in the form:
  <event> occurred in <year_or_period>
  <event> lasted <duration>
  <event> began in <year_or_period>
  <event> ended in <year_or_period>

Rules:
- Use ONLY content from the source. Do not add, infer, summarize,
  or generalize.
- Keep the time reference exactly as it appears in the source.
- If the source contains no absolute date, time, or duration,
  output the single word NONE."""


CLAIM_PROMPTS: dict[str, str] = {
    "definition": CLAIM_DEFINITION,
    "process": CLAIM_PROCESS,
    "causation": CLAIM_CAUSATION,
    "temporal": CLAIM_TEMPORAL,
    "datetime": CLAIM_DATETIME,
}


# Layer 1 — Prose sensors (1B)

PROSE_TOPIC = f"""You are identifying what topic a source sentence is about.
Output ONE noun phrase naming the primary topic.

Source sentence:
{{sentence}}

Rules:
- Use a noun phrase that appears in the source.
- Do not add, infer, or generalize.
- If no clear topic exists, output NONE."""


PROSE_RELATION = f"""You are identifying the main relational structure of a source
sentence. Pick one label from:

  defines | describes_process | gives_example | compares |
  lists_parts | cause_effect | sequence

Source sentence:
{{sentence}}

Rules:
- Output exactly one label from the list, nothing else.
- If none applies, output NONE."""


PROSE_CONTEXT = f"""You are identifying any scope qualifier in a source sentence —
a phrase that narrows when or where a claim applies.

Source sentence:
{{sentence}}

Output each qualifier as one line in the form:
  in <scope>
  during <phase>
  for <domain>

Rules:
- Use ONLY qualifiers that appear in the source.
- If no qualifier exists, output NONE."""


PROSE_PROMPTS: dict[str, str] = {
    "topic": PROSE_TOPIC,
    "relation": PROSE_RELATION,
    "context": PROSE_CONTEXT,
}


# Layer 2 — Integrators (3B)

CLAIM_INTEGRATOR = """You are validating candidate factual claims extracted from a
source sentence. You see the source sentence and a list of candidate
claims produced by multiple small models. Your job is to keep only
claims that (a) are supported by the source and (b) appear in at
least two candidate lists (witness consensus).

Source sentence:
{sentence}

Candidate claims from definition sensor:
{definition}

Candidate claims from process sensor:
{process}

Candidate claims from causation sensor:
{causation}

Candidate claims from temporal sensor:
{temporal}

Candidate claims from datetime sensor:
{datetime}

Output the surviving claims, one per line, in the form:
  <subject> <verb> <object>

Rules:
- Keep a claim only if it appears in at least two candidate lists
  OR if one list produced it verbatim from the source (exact
  substring match).
- Drop any claim that cannot be verified as a substring or paraphrase
  of the source.
- Do not add new claims.
- If no claim survives, output NONE."""


PROSE_INTEGRATOR = """You are integrating discourse signals from multiple prose sensors
on the same source sentence. Produce a single compact summary the
claim-integrator can use as a grounding check.

Source sentence:
{sentence}

Topic sensor output: {topic}
Relation sensor output: {relation}
Context sensor output: {context}

Output exactly three lines:
  topic: <one noun phrase, or NONE>
  relation: <one label, or NONE>
  context: <one qualifier, or NONE>

Rules:
- Only restate what the sensors already said; do not invent.
- If sensors disagree, choose the option that appears in the source
  text verbatim. If none do, use NONE."""


# Layer 3 — Final synthesizer (3B / future 7B)

SYNTHESIZER = """You are producing the final list of clean claims to teach a
knowledge graph. You receive:
- the original source sentence
- the claim-integrator's validated claim list
- the prose-integrator's grounding (topic / relation / context)

Source sentence:
{sentence}

Validated claims:
{claims}

Grounding:
  topic: {topic}
  relation: {relation}
  context: {context}

Output the final teach list, one claim per line, in the form:
  <subject> <verb> <object>

Rules:
- Each claim must be a substring or close paraphrase of the source
  sentence.
- Prefer short claims (3-8 words) over long ones.
- Use only these verbs: is, are, has, have, produces, contains,
  includes, requires, precedes, follows, causes, prevents, divides.
- If a candidate claim contradicts the grounding (e.g., claim is about
  one topic but context excludes it), drop it.
- Do not add any claim not present in the validated list.
- If nothing survives, output NONE."""
