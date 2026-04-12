# v015 — Primitive Layers as Brain Structures: The Functional Mapping

**Date:** April 12, 2026
**Author:** Jennifer Pearl
**Status:** Design note — documents the mapping between Sara's innate primitive layers and specific human brain structures, introduces the CLEANUP primitive layer, and connects the protective urgency system to JKD philosophy.

---

## The Discovery

Sara's innate primitive layers were designed from first principles as the pre-cognitive substrate that all learned knowledge grounds in — the same way a baby is born with pain responses, face preferences, and startle reflexes before learning anything. During the development of the cleanup system (v012-v014), the observation emerged that these layers don't just metaphorically resemble brain structures — they functionally map to specific neural subsystems that evolution built for the same purposes.

This is not a surface-level analogy. Each layer matches a specific brain structure in:
- **What it processes** (threats, bonds, perception, rules, relations, errors)
- **When it fires** (before or after conscious processing)
- **Whether it's innate or learned** (the capacity is innate; the content is learned)
- **How it interacts with other layers** (priority ordering, gating, grounding)

---

## The Seven Primitive Layers and Their Brain Structure Parallels

### 1. SAFETY → Amygdala

```python
SAFETY = frozenset({
    "harm", "pain", "death", "injury", "danger",
    "kill", "hurt", "wound", "suffer", "destroy",
    "protect", "rescue", "save", "shield", "defend",
    "safe", "help", "heal",
})
```

**Brain structure:** The amygdala, particularly the basolateral complex and central nucleus.

**Functional parallel:**
- The amygdala processes threats **before conscious awareness** via LeDoux's subcortical "low road" — sensory input goes directly to the amygdala before reaching the cortex
- Sara's SAFETY primitives are checked **before the cortex (LLM) processes input**
- The amygdala doesn't "think" about threats — it reacts based on innate fear circuits and conditioned associations
- SAFETY primitives don't "reason" about danger — they ground-check against hardwired drives. A path is dangerous because it terminates at `death` or `harm`, not because the system logically concluded it was dangerous
- The amygdala **can be conditioned** (learned fear) but the **capacity for fear is innate**
- Sara's SAFETY layer **is innate** but **what is dangerous is learned** through paths that ground in SAFETY primitives

**Key property:** The amygdala gates behavior. It can veto an action before the cortex finishes processing. Sara's SAFETY layer gates teach operations — the Darwin Award protection refuses a teach that contradicts a safety-grounded path before the LLM elaborates on it.

### The Protective Urgency Function as Trained Amygdala Response (JKD Parallel)

The `protective_urgency` function in Sara's care module is not the raw amygdala response — it is the amygdala response **trained through the principles of Jeet Kune Do.**

In her book *The School of Life's Intercepting Fist* (https://lunarfawn.github.io/the-school-of-lifes-intercepting-fist/book/), Jennifer Pearl (Cricket) describes the untrained crisis response: the brain's default is to **fight everything**. Pain, fear, loss, threat — the untrained response is resistance. Override the pain. Power through the fear. Brute-force the obstacle. This is the raw amygdala cascade — fight-or-flight with no interception.

JKD teaches a different response. Bruce Lee's "intercepting fist" means detecting the opponent's committed strike **before it lands** — not after impact, not during the swing, but at the moment of commitment. The trained practitioner senses the trigger moment and redirects before the cascade completes.

Pearl's Chapter 4 ("I figured out the source of my ignorance") applies this to emotional crisis. The question she asks her students: **"Are you able to sense when you have reached the point of no return?"** This is the amygdala's activation window — the brief moment between trigger and cascade where interception is possible.

The `protective_urgency` function implements this at the architectural level:

| JKD principle | Sara's protective_urgency |
|---|---|
| Sense the trigger before the cascade | SAFETY primitive is reached — recognition wavefront fires |
| Intercept before impact | The urgency function runs BEFORE any action — calculates who, how bad, how reachable |
| Proportional response (match force to situation) | Multiplicative urgency scales with severity, self-rescue capacity, comprehension |
| The one case where full force is appropriate | Trump card: total helplessness = categorical highest priority. The amygdala SHOULD fire at full strength when a victim has zero agency |
| Reject the fighting response to reality | The function accepts "I don't know" honestly instead of brute-forcing an answer from training data |
| "Using no way as way" | Need-based, not identity-based. No tribal template. No hardcoded rule for who matters more. Each situation on its own terms |

The untrained system (bare LLM) is the fighter — it brute-forces answers from training data, overrides gaps in knowledge with hallucination, powers through uncertainty with confidence. The trained system (Sara) senses the gap (short-circuit: "I don't know"), accepts it honestly, and redirects (asks the user to teach). The same principle that Pearl teaches in JKD class for managing emotional crisis is the same principle that Sara implements for managing knowledge uncertainty.

The trump card — total helplessness, where `can_self_rescue=False` and `understands_situation=False` — is the one case where interception is not the right response. When a victim has **zero agency**, the full amygdala response is appropriate. You don't intercept; you act. This maps to Pearl's recognition that the trained response is not always to redirect — sometimes the situation demands full force, and the skill is knowing the difference.

### 2. SOCIAL → Hypothalamus (Oxytocin/Vasopressin Systems)

```python
SOCIAL = frozenset({
    "self", "other", "tribe", "kin", "stranger", "child",
    "bond", "love", "trust", "care", "belong",
    "feed", "tend", "nurture", "comfort", "carry", "share",
    "face", "voice", "name", "presence",
    "joy", "grief", "empathy", "loneliness",
    "feast", "celebrate", "mourn_together", "play",
    "work_together", "survive_together",
})
```

**Brain structure:** The hypothalamus (paraventricular nucleus, supraoptic nucleus) and its oxytocin/vasopressin projections, plus the medial preoptic area for parental care.

**Functional parallel:**
- Oxytocin drives bonding, trust, maternal care, and social recognition — all innate capacities present at birth
- The hypothalamus doesn't compute social relationships — it produces the neurochemical substrate on which social learning builds
- The beer hypothesis (trust accelerators like `feast`, `mourn_together`, `survive_together`) maps to oxytocin release during shared positive experiences

**Key property:** The hypothalamus is the healed-femur layer. The moment caring for the weak became more important than self-preservation is the moment of the SOCIAL primitive's evolutionary installation.

### 3. SENSORY → Primary Sensory Cortices (V1, A1, S1)

```python
SENSORY = frozenset({
    "color", "shape", "size", "texture",
    "edge", "pattern", "material",
})
```

**Brain structure:** Primary visual cortex (V1), primary auditory cortex (A1), primary somatosensory cortex (S1).

**Functional parallel:**
- V1 detects edges, orientations, colors — raw features that higher areas combine into recognition
- The LLM acts as Sara's sensory cortex — it converts raw input into feature labels
- Primary sensory cortices are innate — V1's orientation columns are present at birth (Hubel & Wiesel)

**Key property:** Sensory processing is pre-cognitive. It happens before recognition, before meaning, before language.

### 4. STRUCTURAL → Prefrontal Cortex (Dorsolateral PFC)

```python
STRUCTURAL = frozenset({
    "rule", "pattern", "name", "type",
    "order", "group", "sequence", "structure",
    "boundary", "relation",
})
```

**Brain structure:** Dorsolateral prefrontal cortex (DLPFC).

**Functional parallel:**
- The DLPFC maintains rules in working memory and applies them to guide behavior
- STRUCTURAL primitives provide the organizational substrate on which all structured knowledge is built

### 5. ETHICAL → Ventromedial PFC / Orbital Frontal Cortex

```python
ETHICAL = frozenset({
    "no_unsolicited_action",
    "no_unsolicited_network",
    "obey_user",
    "trust_tribe",
    "accept_shutdown",
})
```

**Brain structure:** Ventromedial prefrontal cortex (vmPFC) and orbitofrontal cortex (OFC).

**Functional parallel:**
- The vmPFC assigns value to actions and inhibits socially inappropriate behavior. Patients with vmPFC damage (Phineas Gage) know the rules but can't follow them
- ETHICAL primitives are behavioral constraints hardwired at the API level
- `accept_shutdown` maps to the vmPFC's ability to inhibit self-preservation impulses in service of social cooperation

### 6. RELATIONAL → Angular Gyrus + Temporal-Parietal Junction

```python
RELATIONAL = frozenset({
    "is", "has", "contains", "includes",
    "follows", "precedes", "requires", "excludes",
})
```

**Brain structure:** Angular gyrus and temporal-parietal junction (TPJ).

**Functional parallel:**
- The angular gyrus processes semantic relationships — "A is a kind of B", "A contains B"
- RELATIONAL primitives are the verb vocabulary for these relationships

### 7. CLEANUP → Anterior Cingulate Cortex (ACC) + Hippocampus

```python
CLEANUP = frozenset({
    "reviewed",
    "refuted",
    "corrected",
    "kept",
    "consolidated",
})
```

**Brain structures:**

- **Anterior cingulate cortex (ACC):** Error detection and conflict monitoring. The ACC fires when there's a mismatch between expected and observed outcomes. This is exactly what the contested epistemic state is — `belief ≈ 0, evidence_weight >> 0`. The ACC says "something is wrong here, pay attention."

- **Hippocampus:** Memory consolidation during sleep. The hippocampus replays recent experiences during slow-wave sleep, strengthening some and weakening others. This is exactly what Sara's cleanup/consolidation cycle does.

**Functional parallel:**
- Error detection (ACC) → Sara's pollution detection — scanning for contested or suspicious paths
- Conflict monitoring (ACC) → disambiguation prompt — detecting that two close-spelled terms might be the same concept
- Memory consolidation (hippocampus) → sleep cycle — Sara reviews what was taught, flags issues, presents a report
- The correction path ("I once thought X but now I know Y") is the ACC detecting the error + the hippocampus re-encoding the corrected version

**Key property:** The ACC and hippocampus together enable **metacognition** — thinking about thinking. Sara's CLEANUP primitives are the substrate for the same capability.

---

## The Correction Path as Knowledge

The act of correcting a belief is itself a primitive cognitive event. "I once thought X but now I know Y because I examined the evidence and changed my mind" is not a side-effect of database maintenance. It is a fundamental brain operation.

A brain that has corrected itself knows something a brain that never had the wrong belief doesn't know — it knows **what wrong looks like**. This is why experienced doctors are better than textbook doctors — they've made mistakes, caught them, and internalized the correction path. The correction itself becomes expertise.

In Sara, the refuted path stays. The corrected path stays. The connection between them is a real chain in the graph. Future queries can trace the correction.

---

## The Seven Layers Together

| # | Layer | Brain Structure | Function | Origin |
|---|-------|-----------------|----------|--------|
| 1 | SENSORY | Primary sensory cortices (V1/A1/S1) | Raw perception | v1.0 |
| 2 | STRUCTURAL | Dorsolateral prefrontal cortex | Organization, rules, executive function | v1.0 |
| 3 | RELATIONAL | Angular gyrus / TPJ | Semantic relations, language structure | v1.0 |
| 4 | ETHICAL | Ventromedial PFC / OFC | Behavioral constraints, value-based gating | v1.0 |
| 5 | SAFETY | Amygdala | Threat detection, protective drives (JKD-trained) | v1.1 |
| 6 | SOCIAL | Hypothalamus (oxytocin/vasopressin) | Bonding, trust, care, recognition | v1.1 |
| 7 | **CLEANUP** | **ACC + Hippocampus** | **Error detection, correction, consolidation** | **v1.2** |

All learned knowledge grounds out in some combination of these seven layers. The brain is blank at birth except for this substrate, exactly as a human infant is.

---

## Implementation: CLEANUP Paths in the Graph

The CLEANUP primitives are PROPERTY neurons in the innate layer, just like SAFETY and SOCIAL primitives. When a neuron is cleaned, a 3-neuron chain is created — the same pattern as all other knowledge in Sara:

**Fact path (existing pattern):**
```
red  →  apple_color  →  apple
```

**Cleanup path (same pattern):**
```
waht  →  waht_cleanup  →  reviewed
```

Where:
- `waht` = the polluted neuron (already exists)
- `waht_cleanup` = a new RELATION neuron, concept-specific
- `reviewed` = the CLEANUP primitive (PROPERTY neuron from innate layer)

**No shared hub. No bleed-over.** Each cleaned concept gets its own relation neuron. `waht_cleanup` and `teh_cleanup` are separate neurons that don't connect to each other. Information about cleaning one cannot bleed into the other.

**Cluster link:** The cleanup relation neuron also connects to the cluster center:
```
waht_cleanup  →  edubba   (relation: "cleanup_cluster")
```

Making cleanup state discoverable through normal cluster queries without a global hub.

**Correction chain:** When a typo-fix happens:
```
waht  →  waht_cleanup  →  corrected   (CLEANUP primitive)
what  →  [normal fact paths]           (the clean re-taught version)
```

The history is traceable. The correction itself is knowledge grounded in a primitive.

---

## Predictions from the Mapping

Because the layers map to brain structures with known functions, the mapping predicts what Sara should be able to do that isn't yet implemented:

| Brain structure function | Sara prediction |
|---|---|
| Amygdala extinction learning | Sara should be able to "unlearn" a safety response when evidence shows it's no longer valid — but with a higher threshold than normal refutation |
| Hippocampal place cells | Sara could develop spatial/navigational reasoning by grounding paths in spatial primitives |
| ACC reward prediction error | Sara could detect "this fact should have been true but wasn't" and flag it |
| Hypothalamic homeostasis | Sara could monitor her own "brain health" metrics and flag when pollution exceeds a threshold |
| Basal ganglia habit formation | Frequently-traversed paths could become "habits" — automatic, fast, resistant to refutation |

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| v015 | 2026-04-12 | Primitive layers mapped to brain structures. CLEANUP introduced as seventh layer. JKD parallel for protective urgency. Correction path as knowledge. Predictions from the mapping. |
| v014 | 2026-04-12 | Sleep and consolidation design note |
| v013 | 2026-04-11 | Provenance gaming and long-term typo pollution |
| v012 | 2026-04-11 | Failure modes and the cortex |
| v011 | 2026-04-08 | Installation guide |
| v010 | 2026-04-06 | Sara Care: dementia assistance proof-of-concept |
