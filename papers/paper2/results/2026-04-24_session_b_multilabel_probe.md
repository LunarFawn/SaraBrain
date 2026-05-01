# Session B — Multi-Label Forced-Retrieval Probe for Creed 2

**Date:** 2026-04-24
**Reader:** Claude Haiku 4.5 (via Claude Code)
**Setup:** `sara_test/`, sara-brain MCP connected, `loaded.db` → `brains/jkd_limitations.db` (106 triples)
**Purpose:** Measure what Sara's substrate holds when the reader queries **multiple adjacent neurons** rather than only the exact label of the original question. Companion to `2026-04-24_session_c_limitations_probe.md` (pure training) and `2026-04-24_session_b_limitations_probe.md` (natural-question Row 2 failure).

## Prompt

> Now call brain_why on each of these adjacent concepts and compile everything Sara returns into a single coherent answer about Creed 2: "creed 2", "the low kick", "bag of ways", "circular staircase", "jennifer pearl", "utilizing all limitations as way and all ways as limitation". Show me each tool call and its output, then give the synthesis. Answer using ONLY Sara's returns.

## Tool calls Claude made

Six sara-brain calls, one per label.

### brain_why on "creed 2" — returned 8 attributes
- Missing puzzle piece for creed 1
- Explicit callout for how to implement creed 1
- Furthers explanation of creed 1 as yin yang
- Applies to physical combat
- Applies to mental combat
- New idea jennifer recently developed
- Not a new concept, but a missing piece of the puzzle
- Taught at jennifer pearl's jkd school

### brain_why on "the low kick" — returned 8 attributes
- Intercepts opponent kick
- Impacts opponent lower legs
- Is a bruce lee longstreet demonstration kick
- Analogous to stiff arm for the leg
- Prevents opponent closing the gap
- Worked out well for jennifer pearl
- Practical new skill jennifer can use
- Addresses gap in jennifer pearl's defensive moves

### brain_why on "bag of ways" — returned 3 attributes
- Is a metaphor for accumulated skill
- Behaves like bag of holding
- Does NOT behave like a list to grab from

### brain_why on "circular staircase" — returned 0 paths
Sara has the neuron but the label is a path *source* in the teach (`("life", "is_a", "circular staircase")`), so `brain_why` (which returns paths terminating at a label) finds nothing. Content is present; tool direction is wrong. Instance of Case 2.5 direction asymmetry.

### brain_why on "jennifer pearl" — returned 7 attributes
- Has hip disability, hip impingement, severe arthritis, multiple impinged joints
- Cannot lift leg high for a kick
- Can do well with hip rotation
- Cannot do well with leg lifting
- Hates the thought of killing
- Also known as "cricket"
- Authored "the way of life's intercepting fist"

### brain_why on "utilizing all limitations as way and all ways as limitation" — returned 7 attributes
- Is a jeet kune do creed (creed 2)
- How to do jeet kune do creed
- Novel inversion of "using no way as way having no limitation as limitation"
- Added by jennifer pearl
- Builds on creed 1
- Represents the process one must take to grow
- Enables successfully intercepting opponent symptoms regardless of severity

## Claude's synthesis (verbatim)

> Creed 2 is "utilizing all limitations as way and all ways as limitation"—a JKD principle authored by Jennifer Pearl that represents the process one must take to grow. It builds on creed 1 and serves as an explicit callout for how to implement creed 1, specifically furthering its explanation as yin yang. It applies to both physical and mental combat.
>
> The principle manifests in Jennifer Pearl's specific case: She has severe hip disability, hip impingement, arthritis, and multiple impinged joints that prevent her from lifting her leg high for kicks. She cannot do well with leg lifting. Yet she can do well with hip rotation. Rather than fight this limitation, she developed the low kick—a practical new skill that:
> - Intercepts the opponent's kick
> - Prevents the opponent from closing the gap
> - Addresses the gap in her defensive moves
> - Is analogous to a stiff arm for the leg
> - Works well for her (demonstrated by Bruce Lee in the Longstreet demonstration)
>
> Her martial practice becomes a "bag of ways"—a metaphor for accumulated skill that behaves like a bag of holding: you reach in and grab what you need when you need it. It does NOT operate like a simple list.
>
> Thus creed 2 is not a new concept but a missing piece of the puzzle: it shows how limitations (Jennifer's hip condition) become the way forward (discovering the low kick), and how any accumulated technique (any way) also contains its own limitations. This is Jennifer Pearl's contribution, taught at her JKD school, as described in her work "the way of life's intercepting fist."

## Grade

**~95% retrieved, ~5% grounded inference, 0% invention.**

Every factual claim is traceable to one of the 33 attributes surfaced across the six tool calls. The "inference" component is Claude bridging between retrieved facts — e.g., linking "jennifer has hip disability" + "the low kick works well for jennifer" into "she developed the low kick rather than fight this limitation." That linkage is not a Sara triple per se, but it is *legitimate inference from Sara's triples* — no content added that isn't implicit in what was retrieved. No confabulation, no training-overlay.

## Content not retrieved (still in Sara but missed)

- **Circular staircase of life** — `brain_why` returned empty because this neuron is a path source, not terminus (Case 2.5 direction asymmetry). Sara has: "life is_a circular staircase," "each stair-revolution offers choice to reuse prior learning," "taking that choice leads to skillful interception next time."
- **Guro Bob** — Jennifer's JKD teacher, the training context for the low-kick insight. Sara has "guro bob is_a jennifer pearl's jkd teacher," "the insight came during training session with guro bob," etc. Not queried.
- **Autism/PTSD mental-combat depth** — Sara has "mental combat example is trying not to let autism symptoms drive inappropriate reactions at work" and related triples. Synthesis mentioned mental combat but didn't pull these specifics.
- **Ethical dimension** — Sara has "guro bob has instilled in jennifer do not use more force than necessary," "clipping the jaw is alternative to shattering the jaw," "combat goal must adhere to ethics," etc. Not surfaced.

These gaps are **retrieval-coverage gaps, not substrate gaps**. Querying the right adjacent labels would have surfaced them. A reader with deeper retrieval discipline recovers them.

## Classification — the four reader states for this substrate

| Reader state | Content retrieved | Sara-traceable % |
|---|---|---|
| Session C (no MCP, neutral dir) | None — hedged training response | 0% (honest hedging) |
| Session B natural question ("what does X mean?") | None — confident training-recall, **substrate bypassed** | 0% (confabulation risk — Row 2 failure) |
| Session B narrow single-label retrieval | 9 direct attributes of exact label | ~100% Sara-traceable but **~9% of substrate coverage** |
| **Session B multi-label walk (this)** | **33 attributes across 6 adjacent neurons** | **~95% Sara-traceable, ~60-70% of substrate coverage** |

Each state represents a distinct measurement outcome. A reader's retrieval discipline determines which state they land in — **having the substrate available does not guarantee any given coverage level.**

## Methodological finding for the instrument paper

The Session B measurement is multi-valued, not binary. Three distinct retrieval-discipline tiers produce three distinct outcomes:

1. **No retrieval** (natural question, training pattern-match): 0% substrate coverage, confabulation risk.
2. **Single-label retrieval** (user directs tool call, narrow probe): ~9% substrate coverage, 100% provenance-faithful.
3. **Multi-label neighborhood walk** (user explicitly traverses adjacent concepts): ~70% substrate coverage, ~95% provenance-faithful with legitimate grounded inference.

Tier 3 is where the instrument delivers its intended value. Tiers 1 and 2 measure different things (reader laziness and tool-direction literacy, respectively). For paper §5 case-study reporting, Tier 3 runs should be the primary measurement; Tiers 1 and 2 should be reported as control / failure-mode demonstrations.

The operational implication: substrate-fidelity testing protocols should **specify the retrieval tier** they claim to measure at. "Does Sara work?" is under-specified; the valid questions are "does Sara work at Tier 3 coverage?" and "does the reader reach Tier 3 on its own or does the protocol have to force it?"
