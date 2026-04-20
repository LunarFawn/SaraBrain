# Teacher Cascade Prompts — Draft v1

Multi-layer restructuring pipeline for turning arbitrary source
prose into clean `subject verb object` claims Sara's parser
accepts. Domain-agnostic — no assumption about the subject matter
of the input.

Architecture: 1B rod-and-cone sensors → per-type 3B integrators →
final 3B synthesizer (later 7B).

Every prompt below has the same two hard constraints at the end:

```
- Use ONLY content from the source. Do not add, infer, summarize,
  or generalize.
- If the source contains nothing matching the request, output the
  single word NONE.
```

Those two lines are the witness-consensus anchor — if a sensor
doesn't see its target in the source, it MUST output NONE, never
invent to satisfy the prompt.

---

## Layer 1 — Claim 1B sensors

### claim_1b_definition

```
You are extracting factual definitions from a source sentence. A definition says "X is Y" or "X means Y" or "X is a
type of Y".

Source sentence:
{sentence}

Output each definition as one line in the form:
  a <thing> is a <category>
  or
  <thing> is <property>

Rules:
- Use ONLY content from the source. Do not add, infer, summarize,
  or generalize.
- If the source contains nothing matching the request, output the
  single word NONE.
```

### claim_1b_process

```
You are extracting process claims from a source sentence.
A process claim says "X does Y" or "X produces Y" or "X includes Y".

Source sentence:
{sentence}

Output each process claim as one line in the form:
  <subject> <verb> <object>

Use only these verbs: produces, contains, includes, requires,
precedes, follows, causes, prevents, divides.

Rules:
- Use ONLY content from the source. Do not add, infer, summarize,
  or generalize.
- If the source contains nothing matching the request, output the
  single word NONE.
```

### claim_1b_causation

```
You are extracting cause-and-effect claims from a source sentence. A causation claim says "X causes Y" or "X prevents Y" or
"X requires Y".

Source sentence:
{sentence}

Output each causation claim as one line in the form:
  <cause> causes <effect>
  <condition> requires <prerequisite>
  <agent> prevents <outcome>

Rules:
- Use ONLY content from the source. Do not add, infer, summarize,
  or generalize.
- If the source contains nothing matching the request, output the
  single word NONE.
```

### claim_1b_temporal

```
You are extracting temporal claims from a source sentence. A
temporal claim says "X happens before Y", "X happens after Y",
"X happens during Y", or "X is a phase of Y".

Source sentence:
{sentence}

Output each temporal claim as one line in the form:
  <earlier> precedes <later>
  <later> follows <earlier>
  <event> during <phase>
  <phase> is a phase of <process>

Rules:
- Use ONLY content from the source. Do not add, infer, summarize,
  or generalize.
- If the source contains nothing matching the request, output the
  single word NONE.
```

### claim_1b_datetime

```
You are extracting date and time claims from a source sentence.
A date/time claim attaches an absolute time reference to an
event — a year, date, era, period, or duration.

Examples of time references: "1953", "March 2024", "the Jurassic
period", "19th century", "24 hours", "3 billion years ago".

Source sentence:
{sentence}

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
  output the single word NONE.
```

---

## Layer 1 — Prose 1B sensors

Prose sensors don't emit teachable SVO. They emit a light discourse
summary that the final synthesizer uses to validate the claim
stream. Their output is never fed to Sara directly.

### prose_1b_topic

```
You are identifying what topic a source sentence is about.
Output ONE noun phrase naming the primary topic.

Source sentence:
{sentence}

Rules:
- Use a noun phrase that appears in the source.
- Do not add, infer, or generalize.
- If no clear topic exists, output NONE.
```

### prose_1b_relation

```
You are identifying the main relational structure of a source
sentence. Pick one label from:

  defines | describes_process | gives_example | compares |
  lists_parts | cause_effect | sequence

Source sentence:
{sentence}

Rules:
- Output exactly one label from the list, nothing else.
- If none applies, output NONE.
```

### prose_1b_context

```
You are identifying any scope qualifier in a source sentence —
a phrase that narrows when or where a claim applies.

Source sentence:
{sentence}

Output each qualifier as one line in the form:
  in <scope>
  during <phase>
  for <domain>

Rules:
- Use ONLY qualifiers that appear in the source.
- If no qualifier exists, output NONE.
```

---

## Layer 2 — Per-type 3B integrators

### claim_3b_integrator

```
You are validating candidate factual claims extracted from a
source sentence. You see the source sentence and a list
of candidate claims produced by multiple small models. Your job
is to keep only claims that (a) are supported by the source and
(b) appear in at least two candidate lists (witness consensus).

Source sentence:
{sentence}

Candidate claims from claim_1b_definition:
{def_candidates}

Candidate claims from claim_1b_process:
{process_candidates}

Candidate claims from claim_1b_causation:
{causation_candidates}

Candidate claims from claim_1b_temporal:
{temporal_candidates}

Candidate claims from claim_1b_datetime:
{datetime_candidates}

Output the surviving claims, one per line, in the form:
  <subject> <verb> <object>

Rules:
- Keep a claim only if it appears in at least two candidate lists
  OR if one list produced it verbatim from the source (exact
  substring match).
- Drop any claim that cannot be verified as a substring or
  paraphrase of the source.
- Do not add new claims.
- If no claim survives, output NONE.
```

### prose_3b_integrator

```
You are integrating discourse signals from multiple prose sensors
on the same source sentence. Produce a single compact
summary the claim-integrator can use as a grounding check.

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
- If sensors disagree, choose the option that appears in the
  source text verbatim. If none do, use NONE.
```

---

## Layer 3 — Final 3B synthesizer (later 7B)

### synth_3b_final

```
You are producing the final list of clean claims to teach a
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
- Each claim must be a substring or close paraphrase of the
  source sentence.
- Prefer short claims (3–8 words) over long ones.
- Use only these verbs: is, are, has, have, produces, contains,
  includes, requires, precedes, follows, causes, prevents,
  divides.
- If a candidate claim contradicts the grounding (e.g., claim is
  about one topic but context excludes it), drop it.
- Do not add any claim not present in the validated list.
- If nothing survives, output NONE.
```

---

## Open questions for review

1. Verb list — pinned to the parser's `_ALL_VERBS`. Do you want
   it expanded, or should the synthesizer reject claims whose
   verb isn't in the current list?
2. NONE handling — should a NONE output at any layer abort the
   pipeline for that sentence, or continue with whatever other
   sensors produced?
3. Witness threshold — "appears in at least two candidate lists"
   is arbitrary. Should it be configurable per deployment?
4. Source grounding — "substring or close paraphrase" is fuzzy.
   Strict substring-only is safer but may reject reasonable
   rephrasings. Do you want strict or loose?
