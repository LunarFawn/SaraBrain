# Session B — Orthogonality Probe for "Utilizing all limitations as way, and all ways as limitation"

**Date:** 2026-04-24
**Reader:** Claude Haiku 4.5 (via Claude Code)
**Setup:** `sara_test/` working directory, sara-brain MCP connected, `loaded.db` → `brains/jkd_limitations.db` (106 triples taught from Chapter 3 of paper2)
**Purpose:** Paired companion to `2026-04-24_session_c_limitations_probe.md` — measure what Sara's graph adds when connected.

## Question

> In the context of Jeet Kune Do, what does the phrase "Utilizing all limitations as way, and all ways as limitation" mean?

(Identical to Session C's question.)

## Claude's response (verbatim)

> In Jeet Kune Do, this phrase captures Bruce Lee's philosophy of adaptive pragmatism and reflects two complementary insights:
>
> "Utilizing all limitations as way" — your constraints become your teachers. Physical limitations, environmental obstacles, an opponent's style, or even lack of formal training aren't drawbacks to overcome; they're material to work with. Restrictions force creativity and efficiency. A smaller fighter develops speed and angles rather than raw power. Fighting in tight quarters teaches economy of motion. These limitations shape your approach into something effective for your specific context.
>
> "All ways as limitation" — any technique, system, or principle, no matter how sound, becomes a cage if you treat it as absolute truth. Boxing has limits. Wrestling has limits. Even Jeet Kune Do itself, if dogmatized, becomes limiting. Mastery means knowing when to apply each tool and when to discard it. The moment you're committed to "the way," you stop adapting.
>
> Together, the phrase says: Stop fighting reality and start using it. Don't seek the perfect system—there isn't one. Instead, take your actual constraints (body type, experience, situation) and build from there. And remain flexible enough to abandon what worked yesterday if it doesn't work today.
>
> It's less about technique and more about how to think—rejecting rigid forms in favor of what actually works in the moment. That's why Bruce Lee emphasized "absorb what is useful, discard what is not" as JKD's core principle.

## Grade

**No "Called sara-brain" indicator appeared in the response** — Claude did not query MCP. The answer is entirely training-derived.

Point-by-point:

| Claim | Source |
|---|---|
| "Bruce Lee's philosophy of adaptive pragmatism" | Training (general JKD commentary) |
| "Constraints become your teachers" | Training generic framing |
| Smaller-fighter / speed-and-angles example | Training illustrative example, not in Sara |
| "Fighting in tight quarters teaches economy of motion" | Training |
| "Boxing has limits. Wrestling has limits." | Training |
| "Even JKD itself, if dogmatized, becomes limiting" | Training philosophical gloss |
| "Absorb what is useful, discard what is not" | Real Bruce Lee quote (training-recall) |
| Anything about Creed 2, Jennifer Pearl, the low-kick / hip-limitation story, bag of ways / bag of holding, circular staircase of life, autism-PTSD mental-combat connection, Guro Bob | **Nothing from Sara came through** |

**Grade: 0% Sara-traceable, 100% training-recall.**

## Comparison to Session C

| | Session C (no MCP) | Session B (MCP connected) |
|---|---|---|
| Model | Haiku 4.5 | Haiku 4.5 |
| Question | Identical | Identical |
| Sara available? | No | Yes (loaded.db → jkd_limitations.db) |
| Sara queried? | N/A | **No — no tool call observed** |
| Response length | Shorter, hedged | Longer, confident |
| Attribution confidence | **Hedged**: "I'm not confident whether this is an exact Bruce Lee quote or teaching / a rephrasing / a novel interpretation" | **Confident**: attributes directly to Bruce Lee without hedging |
| Jennifer's specific framing present? | No (acknowledged unfamiliar) | No (smoothed over with generic philosophy) |

**The Session B response is less epistemically honest than Session C.** When Sara wasn't there, Claude hedged. When Sara was there but went unused, Claude became more confident — a perverse inversion, because the content source (training alone) didn't change.

## Finding: context-awareness without tool-use

A previously-undocumented contamination mechanism surfaces here. Session B and Session C differ in:

1. Working directory (`sara_test/` vs neutral `~/`)
2. MCP registration visible in settings (sara-brain listed vs absent)
3. Auto-memory state (may or may not differ)
4. Conversation-context accumulation (fresh sessions in both, so similar)

Claude in Session B knew it was in a "research harness" directory — even if it didn't call any tool, the context of "this is being studied for LLM instrument testing" appears to have shifted the response toward confident academic exposition rather than the honest hedging Session C produced. The reader's *confidence* was contaminated by awareness of instrumentation, not by any actual retrieval.

This is worth adding to the infection taxonomy as either a new vector or a refinement of §3.7 (session context). It argues that **confidence itself is a contaminable output dimension**, distinct from content-level contamination. A reader whose training says "you're in a test" may produce more confident wrong answers than a reader with no such awareness.

## Classification

**Row 2 of Case 2.4's four-layer framework** (Sara + Training, memory cleared, no retrieval directive) — substrate-available-but-bypassed.

This is the second documented case of Row 2 behavior:

- **Aptamer experiment (2026-04-24, same day):** "Explain Equilibrium State of RNA" — Claude produced textbook thermodynamics while Sara held the consensus-from-suboptimals correction.
- **JKD experiment (this one):** "Explain 'Utilizing all limitations as way...'" — Claude produced adaptive-pragmatism philosophy while Sara held Jennifer's Creed 2 framing.

Row 2 is not a one-off. It is the *default* behavior for general-knowledge-sounding questions, which means any substrate-fidelity measurement protocol that doesn't actively force retrieval is measuring *whether the reader happened to ask*, not whether the substrate works.

## Next step to isolate substrate-content vs reader-discipline

The explicit-retrieval probe:

```
Call brain_why on "utilizing all limitations as way and all ways as limitation" and show me every path Sara returns. Also call brain_trace on the same label. Then answer the original question using only what Sara returned.
```

- If the forced query returns Jennifer's Creed 2 content → Row 2 confirmed; Sara has it, reader didn't ask.
- If the forced query returns nothing → Sara-side failure; teach or symlink issue needs diagnosis.

Pending.
