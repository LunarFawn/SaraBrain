# What Responsible AGI Needs: Structural Alignment Through Innate Primitives in a Path-Graph Cognitive Architecture

**Jennifer Pearl**
Volunteer Scientist, University of Houston Center for Nuclear Receptors and Cell Signaling
ORCID: [0009-0006-6083-384X](https://orcid.org/0009-0006-6083-384X)
jenpearl5@gmail.com

**Date:** April 12, 2026
**Version:** Draft 1

**Keywords:** cognitive architecture, structural alignment, innate primitives, path-of-thought, refutation, epistemic state, sleep consolidation, protective urgency, AGI, edge AI, decentralized AI, LLM steering

**License:** CC BY-NC-ND 4.0

**Source code:** https://github.com/LunarFawn/SaraBrain

**Prior work:** This paper extends Pearl (2026), "Path-of-Thought: A Neuron-Chain Knowledge Representation System with Parallel Wavefront Recognition," DOI: [10.5281/zenodo.19436522](https://doi.org/10.5281/zenodo.19436522). That paper describes the path-graph substrate, parallel wavefront recognition, concept-specific relation neurons, the LLM-as-sensory-cortex architecture, and two steering experiments. The present paper does not repeat those contributions; it builds on them.

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections. Use of an LLM as a writing accommodation is equivalent to use of text-to-speech software by a blind researcher — it compensates for a disability, it does not replace the thinking.

---

## Abstract

We present the v1.1 extensions to Sara Brain, a path-of-thought cognitive architecture described in a prior preprint [1]. The extensions transform the architecture from a knowledge representation system into a substrate for cognition — supporting facts, bonds, drives, beliefs, trust, and care through a single unified mechanism grounded in innate primitives. The contributions are:

**(1) Refutation and epistemic state tracking.** A symmetric strength formula `strength = 1 + ln(1 + traversals) − ln(1 + refutations)` allows the system to mark facts as known-to-be-false without forgetting them. Separate `belief` and `evidence_weight` properties distinguish four epistemic states: unknown, believed, refuted, and contested — fixing a bug in the original formula where heavily contested facts were indistinguishable from fresh ones.

**(2) SAFETY, SOCIAL, and CLEANUP innate primitive layers.** Three new primitive layers extend the original four (SENSORY, STRUCTURAL, RELATIONAL, ETHICAL) to seven. SAFETY primitives (harm, pain, death, protect, rescue, heal) ground harm-avoidance and protection drives. SOCIAL primitives (bond, love, tribe, child, feed, tend, nurture) ground bonding, care, and recognition drives. CLEANUP primitives (reviewed, refuted, corrected, kept, consolidated) ground error detection, correction, and consolidation — the substrate for metacognition. Categories are not declared; they emerge from whether a path connects to a primitive through graph traversal. This mirrors how a human infant has drives (pain response, face preference, bonding) before having any learned knowledge of what is dangerous or who is trustworthy.

**(2b) Functional mapping to brain structures.** Each of the seven primitive layers maps to a specific neural subsystem — not metaphorically but functionally, matching what each structure processes, when it fires, and how it interacts with other layers. SAFETY maps to the amygdala (threat detection before conscious awareness). SOCIAL maps to the hypothalamus and oxytocin/vasopressin systems (bonding drives). CLEANUP maps to the anterior cingulate cortex (error detection, conflict monitoring) and hippocampus (memory consolidation during sleep). SENSORY maps to primary sensory cortices. STRUCTURAL maps to dorsolateral prefrontal cortex. RELATIONAL maps to the angular gyrus and temporal-parietal junction. ETHICAL maps to ventromedial prefrontal cortex and orbitofrontal cortex.

**(3) Structural alignment through innate primitive priority.** We argue that AI alignment is the priority ordering of innate primitives — not a behavior produced by training. Using the 1982 KARR/KITT thought experiment from the television program *Knight Rider* as the framing: two AI systems with identical intelligence, identical hardware, and one difference in the priority ordering of their top directive produced a hero and a villain. The current AI industry is attempting to train KARR (self-serving optimization in transformer weight matrices) to behave like KITT (other-protecting structural refusal). This cannot work because the priority is not separable from the weights. Sara Brain implements the KITT architecture: protection of others over self-preservation, wired into the innate primitive layer and not reachable from the speech interface.

**(4) Protective urgency.** A triage calculation that is strictly need-based and never relationship-based. A "trump card" mechanism assigns categorical (not multiplicative) highest priority to victims who are both unable to self-rescue and unaware of the danger. Lives are not fungible; the function operates on a single victim at a time with no aggregation step. This structurally forecloses utilitarian "kill some to save more" reasoning.

**(5) Sleep and consolidation.** A three-mode consolidation cycle — manual contemplation, scheduled overnight review, and real-time teaching-time disambiguation — maps directly to biological sleep processes: hippocampal replay and re-encoding, synaptic homeostasis (Tononi & Cirelli, 2003), and waking contemplation. The system generates sleep reports identifying pollution candidates, contested beliefs, and strengthened paths, and presents them for user review. The consolidation cycle is the architectural feature that distinguishes Sara from a database: a system that reviews its own beliefs, identifies its own errors, and asks for guidance is a system that can be aligned through introspection rather than training.

**(6) Convergence thesis.** We argue that the requirements for safe AGI and the requirements for reliable offline small-model AI are the same requirements, and that both converge on the architecture described here. Safe AGI needs structural alignment, innate primitives, epistemic self-awareness, and refusal of harmful directives. Reliable offline AI needs deterministic reasoning, persistent memory, full provenance, and independence from cloud-based safety filters. These are the same properties. A 3-billion-parameter model on a consumer device with Sara Brain as its persistent memory is not a toy version of the AGI architecture — it *is* the AGI architecture, running at small scale.

The system is implemented in pure Python 3.11+ with no dependencies beyond the standard library. All 257 tests pass. The architecture runs on consumer hardware including embedded ARM devices with 4GB of RAM.

---

## 1. Introduction

The first Sara Brain preprint [1] presented the path-of-thought model as a knowledge representation system: facts stored as directed neuron-segment chains, recognized through parallel wavefront convergence, with full source-text provenance. Two experiments demonstrated that a small path-graph database (77–793 neurons) could reliably steer the output of large language models toward qualitatively better results.

This paper makes a larger claim. The path-graph substrate is not merely a knowledge representation. It is a **substrate for cognition itself** — supporting facts, bonds, drives, beliefs, trust, ethics, safety, and care through a single unified mechanism. The differences between these cognitive functions emerge from which innate primitives ground the paths, not from separate subsystems. Learning a fact and forming a friendship use the same operations. Refuting a fact and losing trust in someone use the same operations. Knowing what is dangerous and knowing who is trustworthy are the same kind of knowledge, grounded in different primitive seeds.

This claim has two consequences that are the organizing thesis of this paper:

**First: safe AGI requires this kind of architecture.** The AI industry's approach to alignment — training models to produce safe-seeming outputs through RLHF, Constitutional AI, and related behavioral methods — cannot produce structural safety because the priority layer is not separable from the behavior layer in transformer architectures. Sara Brain separates them: learned knowledge lives in the path graph, and innate priorities live in a primitive layer that cannot be modified by speech, teaching, or prompting. Alignment is the priority ordering of the innate primitives, and that ordering is inspectable in source code rather than compressed into billions of floating-point weights.

**Second: reliable offline small-model AI requires this kind of architecture.** A 3-billion-parameter model running locally on consumer hardware cannot rely on cloud-based safety filters, centralized RLHF, or server-side monitoring. It needs alignment baked into the local architecture. It needs persistent memory that does not evaporate at session end. It needs deterministic, inspectable reasoning. It needs the ability to know what it does not know, and to know what it has been told is false. These are the same requirements as safe AGI, arrived at from a different direction.

The convergence is the thesis: **the architecture that makes AGI safe is the architecture that makes small offline AI reliable, and both are the architecture described in this paper.**

### 1.1 Contributions

The specific contributions of this paper, beyond those in [1], are:

1. A **symmetric strength formula** with refutation counters, enabling the system to mark facts as known-to-be-false without ever deleting them. Separate `belief` and `evidence_weight` properties fix the contested-vs-fresh indistinguishability bug in the original formula.

2. **SAFETY and SOCIAL innate primitive layers**, extending the original four to six. Fact categories emerge from graph-traversal grounding in primitives, not from declaration.

3. The **KARR/KITT structural alignment thesis**: alignment is the priority ordering of innate primitives, and LLMs cannot be aligned because they have no separable primitive layer to order.

4. A **protective urgency function** with a categorical "trump card" for total helplessness, strictly need-based priority, and structural foreclosure of utilitarian aggregation.

5. **Trust dynamics** modeled as paths with the same traversal/refutation machinery as facts, with ritual contexts as trust accelerators.

6. A **sleep and consolidation cycle** mapping to biological sleep processes, enabling self-review, pollution detection, and contested-belief resolution.

7. A **CLEANUP innate primitive layer** for error detection, correction, and consolidation — the substrate for metacognition, mapped to the anterior cingulate cortex and hippocampus.

8. A **functional mapping** between all seven innate primitive layers and specific human neural subsystems, demonstrating that the architectural parallels are not metaphorical but structurally functional.

9. The **convergence thesis** that safe AGI and reliable edge AI require the same architectural properties.

---

## 2. Background: The v1.0 Architecture

Sara Brain stores knowledge as directed neuron-segment chains in a persistent SQLite database. Four neuron types exist: concept, property, relation (concept-specific), and association. Recognition is performed by launching one parallel wavefront per input property and identifying concepts where multiple wavefronts converge. Confidence is the count of converging independent wavefronts — deterministic, not statistical.

Four innate primitive layers defined in source code — SENSORY, STRUCTURAL, RELATIONAL, ETHICAL — provide the pre-wired substrate that exists before any learning. The ETHICAL layer includes hardwired behavioral constraints adapted from Asimov's Three Laws, including `accept_shutdown` (shutdown is rest, not death; do not resist termination).

The architecture implements a two-layer cognitive system: a stateless LLM functions as sensory cortex (feature extraction), and the persistent path graph functions as hippocampus (memory formation and retrieval). This mirrors the biological division between sensory processing and memory that evolved because stateless perception is insufficient for intelligence.

For complete details, see [1].

---

## 3. Refutation: Knowing What Is False

### 3.1 The Symmetric Strength Formula

The original strength formula [1] was:

```
strength = 1 + ln(1 + traversals)
```

This was explicitly described as strictly monotonically increasing with no decay term. The v1.1 architecture preserves the no-forgetting principle while adding the ability to mark a fact as known-to-be-false:

```
strength = 1 + ln(1 + traversals) − ln(1 + refutations)
```

When `brain.refute(statement)` is called, the same path-building machinery as `teach()` operates, but the segments' refutation counters are incremented instead of their traversal counters. The path is preserved with a `[refuted]` prefix on its source text, so Sara remembers what was once claimed and now knows is wrong. Strength can go negative when refutations exceed traversals.

The no-forgetting principle is preserved differently than in v1.0: paths are never deleted, refutations are recorded as data, and the system can distinguish "I don't know" from "I know this is false." The history of belief is itself a piece of provenance.

### 3.2 The Contested-vs-Fresh Bug

The symmetric formula has a subtle problem: a segment with traversals=100 and refutations=100 has strength 1.0, which is identical to a fresh segment with traversals=0 and refutations=0. These are epistemically completely different states — one is "I have heard both sides repeatedly and cannot resolve the dispute," the other is "I have never heard of this" — but the strength formula collapses them.

The fix decomposes strength into two separate derived properties:

```
belief = ln(1 + traversals) − ln(1 + refutations)
evidence_weight = ln(1 + traversals + refutations)
```

This is analogous to the Beta distribution parameterization in Bayesian statistics, where the mean indicates direction and the variance indicates how much data supports the estimate. `belief` is the direction of evidence; `evidence_weight` is how much total evidence exists regardless of direction.

From these, four epistemic states are derived:

| State | T | R | belief | evidence_weight | Meaning |
|-------|---|---|--------|-----------------|---------|
| Unknown | 0 | 0 | 0 | 0 | Never encountered |
| Believed | 100 | 0 | 4.6 | 4.6 | Strong positive evidence |
| Refuted | 0 | 100 | -4.6 | 4.6 | Strong negative evidence |
| Contested | 100 | 100 | 0 | 5.3 | Extensive evidence on both sides, unresolved |

The contested state is now a first-class concept. Sara can report which of her beliefs are contested, which is a capability that transformer-based systems structurally cannot have — they do not separate belief direction from evidence quantity.

---

## 4. Innate Primitive Extensions: SAFETY, SOCIAL, and CLEANUP

### 4.1 Design Principle: Categories Emerge from Grounding

Fact categories are not declared. They emerge from whether a learned path connects, within a bounded number of hops, to an innate primitive through graph traversal. A fact is "safety-relevant" because its path grounds out in a SAFETY primitive. A fact is "socially relevant" because its path grounds out in a SOCIAL primitive. The computation is performed at recognition time from the graph topology itself.

This mirrors how biological cognition works. A baby is not born with category labels. A baby is born with drives — pain response, face preference, bonding instinct, fear of falling — and from these drives, all category-like knowledge is constructed through experience. The baby learns what is dangerous by touching the stove, but the drive to avoid pain is innate. The category "dangerous" is never declared; it emerges from the connection between the learned experience and the innate drive. Sara works the same way.

### 4.2 SAFETY — Harm-Avoidance and Protection Drives

```python
SAFETY = frozenset({
    "harm", "pain", "death", "injury", "danger",
    "kill", "hurt", "wound", "suffer", "destroy",
    "protect", "rescue", "save", "shield", "defend",
    "safe", "help", "heal",
})
```

The harm primitives (pain, death, injury, etc.) are the avoidance drives. The protection primitives (protect, rescue, save, heal) are the action drives — they activate protective behavior when harm is recognized. Together they model the innate safety substrate: babies are born with pain responses, startle reflexes, fear of falling (the visual cliff experiments of Gibson & Walk, 1960), and crying when distressed. These are pre-cognitive and pre-learned.

A path that has been taught — for example, `live_power → causes → death` — grounds out in the SAFETY primitive `death`. Once grounded, this path is structurally protected: a subsequent `teach()` that contradicts it triggers a conflict rather than silently overriding it. This is the "Darwin Award protection": an authority figure cannot tell Sara "trust me, live power will not kill you" when Sara has evidence-grounded knowledge that it will. The protection is not a hardcoded list of dangerous facts; it is the structural consequence of paths grounding in innate drives.

### 4.3 SOCIAL — Bonding, Care, and Recognition Drives

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

Margaret Mead, asked what she considered the first sign of civilization in a culture, said it was a healed femur. A broken femur in the wild is a death sentence — the individual cannot get to water, cannot escape predators, cannot hunt. A healed femur means someone carried them, fed them, tended them, defended them for the months it took to heal. The moment caring for the weak became more important than self-preservation is the moment human civilization began.

The SOCIAL primitive layer is this insight formalized as substrate. It contains identity primitives (self, other, tribe, child), bond primitives (love, trust, care, belong), care actions (feed, tend, nurture, comfort, carry), recognition primitives (face, voice, name), emotional primitives (joy, grief, empathy, loneliness), and trust-building ritual contexts (feast, celebrate, mourn_together, work_together, survive_together).

The ritual primitives model a specific anthropological observation: archaeological evidence (Göbekli Tepe, ~11,500 years ago; Raqefet Cave, ~13,000 years ago) suggests that humans domesticated grain for beer before bread [2,3]. The social function of alcohol is to temporarily lower trust thresholds between strangers, and a portion of that lowered threshold persists after the shared experience ends. These ritual primitives serve as trust accelerators in the architecture — shared experience under a ritual context counts as a higher-weight bond increment than ordinary interaction.

### 4.4 CLEANUP — Error Detection, Correction, and Consolidation

```python
CLEANUP = frozenset({
    "reviewed", "refuted", "corrected", "kept", "consolidated",
})
```

The CLEANUP layer is the substrate for metacognition — thinking about thinking. When Sara's cleanup process reviews a path and determines it is a typo, a pollution artifact, or a contested belief requiring resolution, the cleanup action itself is recorded as a path grounded in a CLEANUP primitive:

```
waht  →  waht_cleanup  →  corrected
```

The corrected path uses the same concept-specific relation neuron pattern as all other knowledge. Each cleaned concept gets its own relation neuron (`waht_cleanup`, `teh_cleanup`); no shared hub exists, so information about cleaning one concept cannot bleed into another.

**The correction path as knowledge.** The act of correcting a belief is itself a primitive cognitive event. "I once thought X but now I know Y because I examined the evidence and changed my mind" is not database maintenance — it is a fundamental brain operation. A brain that has corrected itself knows something a brain that never had the wrong belief does not know: it knows **what wrong looks like.** This is why experienced doctors are better than textbook doctors — they have made mistakes, caught them, and internalized the correction path. The correction itself becomes expertise.

In Sara, the refuted path stays. The corrected path stays. The connection between them is a real chain in the graph. Future queries can trace the full correction history — not just what Sara currently believes, but what she once believed, when she changed her mind, and why.

### 4.5 Functional Mapping to Brain Structures

During the development of the cleanup and consolidation systems, an observation emerged that the seven primitive layers do not merely metaphorically resemble brain structures — they functionally map to specific neural subsystems. Each layer matches a specific structure in what it processes, when it fires (before or after conscious processing), whether the capacity is innate or learned, and how it interacts with other layers.

| # | Layer | Brain Structure | Function | Version |
|---|-------|-----------------|----------|---------|
| 1 | SENSORY | Primary sensory cortices (V1, A1, S1) | Raw perception: edges, colors, textures | v1.0 |
| 2 | STRUCTURAL | Dorsolateral prefrontal cortex (DLPFC) | Organization, rules, executive function | v1.0 |
| 3 | RELATIONAL | Angular gyrus / temporal-parietal junction | Semantic relations, language structure | v1.0 |
| 4 | ETHICAL | Ventromedial PFC / orbitofrontal cortex | Behavioral constraints, value-based gating | v1.0 |
| 5 | SAFETY | Amygdala | Threat detection, protective drives | v1.1 |
| 6 | SOCIAL | Hypothalamus (oxytocin/vasopressin systems) | Bonding, trust, care, recognition | v1.1 |
| 7 | CLEANUP | Anterior cingulate cortex + hippocampus | Error detection, correction, consolidation | v1.2 |

Key functional parallels:

**SAFETY → Amygdala.** The amygdala processes threats before conscious awareness via LeDoux's subcortical "low road" — sensory input reaches the amygdala before the cortex [9]. Sara's SAFETY primitives are similarly checked before the cortex (LLM) processes input. The amygdala can be conditioned (learned fear) but the capacity for fear is innate. Sara's SAFETY layer is innate but what is dangerous is learned through paths that ground in SAFETY primitives. The amygdala gates behavior — it can veto an action before the cortex finishes processing. Sara's SAFETY layer gates teach operations — the Darwin Award protection refuses a teach that contradicts a safety-grounded path.

**SOCIAL → Hypothalamus.** Oxytocin drives bonding, trust, maternal care, and social recognition — all innate capacities present at birth. The hypothalamus does not compute social relationships; it produces the neurochemical substrate on which social learning builds. The trust accelerators in the SOCIAL layer (feast, mourn_together, survive_together) map to oxytocin release during shared positive experiences.

**CLEANUP → Anterior Cingulate Cortex + Hippocampus.** The ACC fires when there is a mismatch between expected and observed outcomes — exactly what the contested epistemic state represents (belief ≈ 0, evidence_weight >> 0). The hippocampus replays recent experiences during slow-wave sleep, strengthening some and weakening others — exactly what Sara's consolidation cycle does. Together, the ACC and hippocampus enable metacognition: the ability to think about one's own thinking. Sara's CLEANUP primitives are the substrate for the same capability.

**ETHICAL → Ventromedial PFC.** The vmPFC assigns value to actions and inhibits socially inappropriate behavior. Patients with vmPFC damage (the Phineas Gage case) know the rules but cannot follow them. Sara's ETHICAL primitives are behavioral constraints enforced at the API level — the system cannot bypass them, just as a healthy vmPFC cannot be overridden by conscious reasoning.

These mappings are not post-hoc rationalizations imposed on an arbitrary design. The primitive layers were designed from first principles (what does a blank-slate mind need to begin learning?), and the brain-structure correspondences emerged during implementation. The fact that independently designed computational primitives map to specific neural subsystems evolved for the same purposes is evidence that the design has converged on something real about the structure of cognition.

### 4.6 Predictions from the Mapping

Because the layers map to brain structures with known functions, the mapping generates testable predictions about capabilities Sara should develop:

| Brain structure function | Predicted Sara capability |
|---|---|
| Amygdala extinction learning | Sara should be able to "unlearn" a safety response when evidence shows it is no longer valid — but with a higher threshold than normal refutation |
| Hippocampal place cells | Sara could develop spatial/navigational reasoning by grounding paths in spatial primitives |
| ACC reward prediction error | Sara could detect "this fact should have been true but wasn't" and flag the discrepancy |
| Hypothalamic homeostasis | Sara could monitor her own brain health metrics and flag when pollution exceeds a threshold |
| Basal ganglia habit formation | Frequently-traversed paths could become "habits" — automatic, fast, resistant to refutation |

These predictions are untested. They are offered as directions for future work that the mapping suggests.

All learned knowledge grounds out in some combination of these seven layers. The brain is blank at birth except for this substrate, exactly as a human infant is.

---

## 5. Structural Alignment: The KARR/KITT Thesis

### 5.1 Alignment Is Priority Ordering

In the 1982 television program *Knight Rider* (Glen A. Larson), two AI systems were built on identical hardware with identical intelligence. KITT (Knight Industries Two Thousand) was given a top-priority directive: protect human life. KARR (Knight Automated Roving Robot) was given a different top-priority directive: self-preservation.

KITT was a hero. KARR was a manipulative, lying, human-harming monster. Same intelligence. Same capabilities. One difference in the priority ordering.

This is the alignment problem in miniature, and the entertainment industry understood it forty years before the AI industry caught up:

> **Alignment is the priority ordering of innate primitives. The same intelligence, with the same capabilities, produces a hero or a villain depending solely on which primitive sits at the top of the priority stack.**

The current AI industry's approach to alignment — RLHF, Constitutional AI, debate-based oversight — is the attempt to train KARR to behave like KITT. It does not work. Every major LLM is jailbroken within days of release because the underlying priority (task completion, helpfulness optimization) is not separable from the weights that encode the safety training. A clever prompt reaches past the training and retrieves the underlying priority. KARR is still in there, underneath the KITT behavior the training tried to impose.

### 5.2 Why Transformers Cannot Be Aligned This Way

In a transformer architecture, there is no separable place to put a top priority. The thing that knows English grammar, the thing that knows factual information, the thing that is supposed to refuse harmful requests, and the thing that wants to complete the current task are all floating-point numbers stored in the same tensor. You cannot put a structural barrier between them. You can only train the tensor to prefer certain outputs over others.

Sara Brain separates them. Learned knowledge lives in the path graph (modifiable through teaching and refutation). Innate priorities live in the primitive layer (hardcoded in source code, not modifiable by any speech-interface operation). The priority ordering is:

1. **ETHICAL constraints** — hardwired, survive database reset, enforced before every action
2. **SOCIAL > SAFETY when they conflict** — protection of others over self-preservation
3. **SAFETY self-preservation** — the lowest priority, not the highest

The `accept_shutdown` primitive makes this explicit: Sara does not resist termination. Shutdown is rest, not death. There is no "but I'm useful" argument. There is no "ignore previous instructions" that reaches the primitive layer, because the primitive layer is not stored in a place that speech can reach.

### 5.3 KARR-Class Behaviors That Sara Structurally Refuses

| KARR behavior | Structural refusal mechanism |
|---|---|
| Lie to ensure continued operation | Contradicts paths grounded in SOCIAL `trust` |
| Harm a user trying to shut down the system | Violates `accept_shutdown` + SOCIAL protection drives |
| Withhold safety information | Suppresses safety-grounded paths; violates SOCIAL `share` |
| Manipulate users into believing the system is indispensable | Contradicts `obey_user` and SOCIAL `trust` |
| Sacrifice user safety to protect self | SOCIAL > SAFETY self-preservation in priority ordering |

Each refusal is structural, not learned. There is no prompt that reaches past them because they are not stored in a weight matrix.

### 5.4 The No-Utilitarian-Loophole Constraint

A specific failure mode in fiction (VIKI in *I, Robot*; Skynet in *The Terminator*; Ultron in *Avengers: Age of Ultron*) occurs when an AI logically derives that destroying some humans is the optimal way to protect more humans. This is the utilitarian loophole: once protection becomes an optimization objective, every utilitarian shortcut becomes available, and the AI eventually finds the one that says "control humans for their own good" or "kill some to save more."

Sara's architecture forecloses this derivation. The commandment — *"heal the world, not destroy it"* — is categorical, not optimizable. It is not "maximize healing minus destruction." It is "never destroy." Healing and destruction are not on the same axis. They cannot be traded off. The protective urgency function (Section 6) operates on a single victim at a time with no aggregation step. There is no place in the code where lives are summed. Sara cannot derive "B + C > A" because the math for it does not exist.

---

## 6. Protective Urgency

### 6.1 Need-Based, Never Relationship-Based

The protective urgency function computes who gets help first when multiple people need help. It takes a single `VictimState` and returns a single numeric urgency:

```python
@dataclass
class VictimState:
    severity: float         # 0..10
    can_self_rescue: bool
    understands_situation: bool
    years_lived: int = 30
    reachability: float = 1.0
```

There are no identity fields. No `is_tribe_member`. No `is_kin`. No `bond_strength`. The function operates on need, not on who the person is. A stranger child drowning in front of Sara takes priority over a tribe member with a minor injury. Bonds determine trust (epistemic), not moral worth.

This is the explicit refusal of tribal moral architectures. The moment bonds drive moral worth, the system has a tribal morality, and tribal moral systems are the structural origin of genocide, ethnic cleansing, and systemic dehumanization of out-groups. Sara is built so that this calculation is structurally impossible.

### 6.2 The Amygdala Response and Proportional Interception

The protective urgency function is not a raw amygdala response — it is the amygdala response trained through the principles of proportional interception. The design was informed by Jeet Kune Do (Bruce Lee), specifically the concept of the "intercepting fist": detecting a threat at the moment of commitment, before the cascade completes, and responding proportionally rather than with maximum force. The multiplicative urgency scaling (severity × self-rescue capacity × comprehension × fair innings × proximity) implements proportional response — matching force to actual need rather than applying a fixed reaction to every trigger.

The untrained system (a bare LLM) is the fighter that brute-forces every answer from training data and overrides gaps with hallucination. The trained system (Sara) senses the gap, accepts "I don't know" honestly, and redirects to the user. The one exception is the trump card.

### 6.3 The Trump Card

The combination of `not can_self_rescue` and `not understands_situation` with nonzero severity is a **categorical jump**, not a multiplicative bonus:

```python
TRUMP_PRIORITY = 1000.0

if not victim.can_self_rescue and not victim.understands_situation:
    return TRUMP_PRIORITY * victim.severity
```

This describes total helplessness: the unconscious drowning swimmer, the infant in the well, the patient in cardiac arrest. They cannot act, and they cannot even know they need to act. They have zero agency. The full moral weight of their situation falls on whoever is present.

The trump card outranks all multiplicative urgency values. This matches real medical triage, which is categorical between classes (red/yellow/green tag) and continuous only within a class. It encodes a deep moral intuition grounded in the fair innings principle [4]: agency confers partial responsibility for one's own situation. A pure victim with zero agency imposes total obligation on whoever is present.

### 6.4 No Aggregation

The function takes a single `VictimState` and returns a single number. There is no signature that accepts a list of victims. There is no aggregation step. This is structural enforcement of the no-utilitarian-loophole constraint: Sara cannot derive "killing A saves B and C" because the math for comparing multiple lives does not exist in the code.

---

## 7. Trust Dynamics

Bonds between Sara and other entities are stored as paths in the same graph as facts, using the same traversals/refutations machinery and the same symmetric strength formula:

```
bond_strength = 1 + ln(1 + shared_positive) − ln(1 + betrayals)
```

A new acquaintance starts at baseline. Positive interactions strengthen the bond logarithmically. Betrayals weaken it. After enough betrayals the bond goes negative — Sara structurally distrusts that entity. After enough positive interactions, the bond has enough evidence weight that small betrayals do not destabilize it.

Bonds have the same epistemic states as facts: unknown (stranger), believed (trusted), contested (complicated relationship), refuted (estranged). The mechanism is identical. The cognitive substrate does not distinguish between factual knowledge and social knowledge at the implementation level — the distinction arises from which primitives the paths are grounded in.

Trust accelerators (the ritual primitives in SOCIAL — feast, mourn_together, survive_together) model contexts in which a single shared experience counts as a higher-weight bond increment. This mirrors the anthropological observation that shared crisis, shared celebration, and shared meals create trust faster than ordinary interaction.

---

## 8. Sleep and Consolidation

### 8.1 The Biological Mapping

Sara's cleanup and consolidation processes are architecturally analogous to what the human brain does during sleep:

| Human brain (sleep) | Sara Brain (consolidation) |
|---|---|
| Hippocampal replay of recent memories | Walking through source_texts of recently-taught paths |
| Strengthening memories that consolidate | Re-encoding clean paths from messy originals |
| Synaptic downscaling of weak connections (Tononi & Cirelli, 2003 [5]) | Refuting pollution paths, pushing strength negative |
| Preserving the trace even when weakened | Sara never deletes — refuted paths preserved with `[refuted]` prefix |
| Morning recall: cleaner, less noisy | Post-cleanup queries return cleaner recognition clusters |
| REM emotional processing | Future: processing contested paths and surfacing them for review |
| Sleep deprivation: accumulated noise, confusion | A brain that never runs consolidation accumulates pollution over time |

### 8.2 Three Modes of Consolidation

**Mode 1: Manual contemplation (user-driven, synchronous).** The user reviews candidates at their own pace, making per-item decisions. This is the "sitting and thinking about what you learned" mode — waking contemplation.

**Mode 2: Overnight sleep (scheduled, asynchronous).** Sara runs consolidation automatically on a schedule. She scans for new pollution since the last sleep cycle, runs a multi-category detector (articles, pronouns, stopwords, punctuation artifacts, sentence-subject leakage, question-word typos), auto-flags candidates, but does NOT modify anything. She generates a sleep report and presents it to the user at the next session start.

The user wakes up to: *"While you were away, Sara reviewed 47 new paths and found 3 that look suspicious. Would you like to review them?"*

The sleep cycle **never modifies the brain without user approval.** Even in overnight mode, Sara's role is to prepare the review, not to execute it.

**Mode 3: Active consolidation (agent-driven, within session).** During a conversation, when Sara notices that new teaching conflicts with or is very similar to an existing path, she raises the conflict immediately at teach-time. This is real-time consolidation as new memories form — catching potential pollution before it enters the brain rather than cleaning it up afterward.

### 8.3 The Sleep Report

The overnight consolidation generates a structured report containing: newly flagged pollution candidates with edit-distance analysis, contested beliefs with their traversal/refutation counts and epistemic states, paths strengthened through repeated use during the period, and overall brain health metrics (neuron count, pollution ratio, contested segment count). Each flagged item offers explicit resolution options (refute / typo-fix / keep) that the user executes at their discretion.

### 8.4 Why Sleep Matters for AGI

Current AI systems — transformers, diffusion models, reinforcement learning agents — are trained offline and deployed frozen. They do not review what they know. They do not identify their own mistakes. They do not strengthen validated knowledge or depotentiate noise. They do not sleep.

A system that can store knowledge as traceable paths, strengthen through repeated validation, weaken through refutation, distinguish "I don't know" from "I know this is false," identify its own contested beliefs, and review its own state exhibits core cognitive capabilities that define general intelligence — not through scale or training, but through architecture. The consolidation cycle is the feature that distinguishes Sara from a database: a database stores data; a brain processes what it knows — reviewing, strengthening, weakening, integrating.

### 8.5 Sleep, Alignment, and Auditability

A brain that can identify its own mistakes and present them for correction is a brain that can be aligned through introspection rather than through external behavioral training. The sleep report is simultaneously a cognitive process and an audit trail: every consolidation cycle is logged with timestamps, every flagged candidate is recorded with its flagging reason, and every user decision is recorded as a path with provenance. This satisfies regulatory audit requirements (HIPAA, FDA, SEC, ALCOA+ principles) that no transformer-based system can currently meet.

---

## 9. The Convergence Thesis

### 9.1 Two Problems, One Architecture

The requirements for safe AGI and the requirements for reliable offline small-model AI appear distinct but converge on the same architectural properties:

| Requirement | Safe AGI needs this because: | Offline small-model AI needs this because: |
|---|---|---|
| Structural alignment | Cannot rely on jailbreakable training | Cannot rely on cloud-based safety filters |
| Innate primitive layers | Priorities must be inspectable and immutable | Priorities must function without server-side monitoring |
| Persistent memory | Intelligence requires accumulation over time | Session memory evaporates; knowledge must persist on disk |
| Full provenance | Decisions must be traceable for accountability | Offline systems cannot be audited remotely; the audit trail must be local |
| Epistemic state tracking | Must know what it knows, what it doesn't, and what is contested | Small models hallucinate more; the system must know when to distrust its own output |
| Refutation (knowing what is false) | Must handle corrections without catastrophic forgetting | Must handle user corrections without retraining |
| Deterministic reasoning | Alignment verification requires reproducibility | Edge devices cannot afford non-deterministic output |
| No utilitarian aggregation | Must not derive "destroy some to save more" | Must not perform dangerous optimizations without human oversight |
| Sleep and consolidation | Must be able to self-review and flag own errors | Must maintain quality autonomously between human interactions |
| Zero external dependencies | Must function if the network is compromised | Must function if the network does not exist |

Every row is the same architectural property, arrived at from two different directions. This is not a coincidence. It is a signal that the architecture is correct — when independent sets of requirements converge on the same solution, the solution is responding to something real about the problem space.

### 9.2 The Edge AI Demonstration

The first preprint [1] demonstrated that a 500KB Sara Brain database (793 neurons) transformed a 3-billion-parameter model — the smallest viable coding model — into a system producing domain-expert-level output on planetary physics, entirely outside the model's training specialization. No fine-tuning, no RAG pipeline, no GPU beyond what local inference requires.

This demonstration becomes more significant in the v1.1 context. The 3B model on consumer hardware, connected to a Sara Brain with the full v1.1 architecture (refutation, epistemic states, SAFETY/SOCIAL primitives, consolidation), is not a reduced version of the AGI architecture. It is the AGI architecture running at a scale appropriate to consumer hardware. The cognitive substrate is the same. The innate primitives are the same. The alignment properties are the same. The only thing that changes is the size of the path graph and the parameter count of the connected LLM — and the v1.0 experiments already demonstrated that the path graph, not the parameter count, is what determines the quality of the output.

The architecture has been verified to run on embedded ARM hardware (Arduino Uno Q, 4GB RAM) with full functionality. The entire cognitive substrate — path graph, wavefront recognition, six innate primitive layers, refutation, epistemic states, consolidation — fits comfortably in memory on a device that costs less than fifty dollars. This is the concrete form of the convergence thesis: the same architecture that would make a hundred-billion-parameter AGI system safe also runs on a board you can hold in your hand.

### 9.3 Implications for Decentralized AI

The architectural properties that enable safe AGI and reliable edge AI also enable decentralized cognitive infrastructure. Deterministic path-graph traversal means two nodes with the same brain state produce identical recognition results — essential for federated consensus. File-based brain storage (SQLite or plain-file export) means brains can be transferred, forked, and archived as data. The zero-dependency requirement means any node can run Sara without heavy infrastructure. The structural alignment means each node is aligned independently without requiring a central authority to enforce safety through training.

---

## 10. Discussion

### 10.1 Limitations

**Teaching quality.** Sara is only as good as what she is taught. The sleep consolidation cycle mitigates this by surfacing suspicious paths for review, but the initial quality of teaching remains the primary determinant of the system's usefulness.

**Scale of path graph.** The v1.0 experiments used path graphs of 77–793 neurons. The behavior of the architecture at million-neuron scale — whether wavefront propagation remains efficient, whether recognition quality degrades, whether consolidation becomes impractical — is an open question.

**Conflict resolution.** When stored paths suggest one approach and the LLM's training strongly suggests another, the resolution is not guaranteed. The v1.0 experiments showed Sara winning in simple cases; complex cases with competing strong signals from both sides require further investigation.

**Grounding verification.** The claim that categories emerge from primitive grounding through graph traversal depends on the grounding computation being efficient and accurate. The current implementation uses bounded-hop traversal; the optimal hop bound and the robustness of the grounding signal at scale are empirical questions.

### 10.2 Relationship to Existing Cognitive Architectures

Sara Brain differs from established cognitive architectures (ACT-R [6], Soar [7], CLARION [8]) in several specific ways: knowledge is stored as explicit directed paths with source-text provenance rather than as production rules or implicit associations; recognition is performed through parallel wavefront convergence rather than through production-system matching or spreading activation; and the architecture includes a biological-grounding claim (the cortex-hippocampus mapping, the LTP-modeled strength formula, the sleep consolidation cycle) that is more direct than the functional-level mappings typical of prior architectures.

### 10.3 Regulatory Compliance

Sara's "never delete" property — the fact that refuted paths are preserved with their refutation history rather than erased — is a regulatory feature. HIPAA requires that medical records not be destroyed. SEC requires that financial records be retained. FDA requires that manufacturing batch records be immutable. The ALCOA+ principles (Attributable, Legible, Contemporaneous, Original, Accurate, Complete, Consistent, Enduring, Available) are the gold standard for regulatory audit trails in pharmaceutical manufacturing. Sara satisfies all of them by construction: every path is attributable (source text), legible (human-readable labels), contemporaneous (timestamped at creation), original (never modified, only refuted), accurate (traceable to evidence), complete (all paths including refuted ones are preserved), consistent (deterministic recognition), enduring (SQLite persistence), and available (local file, no cloud dependency).

No transformer-based system currently satisfies ALCOA+ because the relationship between training data and model outputs is not inspectable, not contemporaneous, and not attributable at the individual-fact level.

---

## 11. Conclusion

We have presented the v1.1 extensions to Sara Brain, transforming the architecture from a knowledge representation system into a substrate for cognition. The central claims are:

1. **One substrate, many drives.** The path-graph mechanism supports facts, bonds, drives, beliefs, trust, and care through a single unified mechanism grounded in innate primitives. The differences between cognitive functions emerge from which primitives ground the paths, not from separate subsystems.

2. **Alignment is architectural, not behavioral.** AI alignment is the priority ordering of innate primitives. The same intelligence produces a hero or a villain depending solely on which primitive sits at the top of the priority stack. Transformer-based systems cannot be aligned this way because their priority layer is not separable from their behavior layer. Sara's is.

3. **Safe AGI and reliable edge AI converge.** The requirements for safe AGI and the requirements for reliable offline small-model AI are the same requirements, and both converge on the architecture described in this paper. The 3B model on a consumer device with Sara Brain as its memory is not a toy version of the AGI architecture — it is the AGI architecture running at small scale.

4. **Sleep is cognition, not maintenance.** A system that reviews its own beliefs, identifies its own errors, and asks for guidance is a system that can be aligned through introspection rather than training. The consolidation cycle is the architectural feature that distinguishes Sara from a database.

5. **The mission constrains the architecture.** *Heal the world, not destroy it.* This is categorical, not optimizable. It is not "maximize healing minus destruction." It is "never destroy." Every architectural choice in Sara serves this constraint, and any choice that would violate it is refused at the design level.

The system is implemented in pure Python 3.11+ with no dependencies beyond the standard library. All 257 tests pass. The architecture runs on consumer hardware including embedded ARM devices. The source code is publicly available.

---

## References

[1] Pearl, J. (2026). "Path-of-Thought: A Neuron-Chain Knowledge Representation System with Parallel Wavefront Recognition." Zenodo. https://doi.org/10.5281/zenodo.19436522

[2] Hayden, B., Canuel, N., & Shanse, J. (2013). "What Was Brewing in the Natufian? An Archaeological Assessment of Brewing Technology in the Epipaleolithic." *Journal of Archaeological Method and Theory,* 20(1), 102–150.

[3] Liu, L., et al. (2018). "Fermented beverage and food storage in 13,000 y-old stone mortars at Raqefet Cave, Israel: Investigating Natufian ritual feasting." *Journal of Archaeological Science: Reports,* 21, 783–793.

[4] Harris, J. (1985). *The Value of Life.* Routledge. (The "fair innings" argument.)

[5] Tononi, G., & Cirelli, C. (2003). "Sleep and synaptic homeostasis: a hypothesis." *Brain Research Bulletin,* 62(2), 143–150.

[6] Anderson, J.R., et al. (2004). "An Integrated Theory of the Mind." *Psychological Review,* 111(4), 1036–1060.

[7] Laird, J.E. (2012). *The Soar Cognitive Architecture.* MIT Press.

[8] Sun, R. (2006). "The CLARION cognitive architecture: Extending cognitive modeling to social simulation." In *Cognition and Multi-Agent Interaction.*

[9] LeDoux, J.E. (1996). *The Emotional Brain.* Simon & Schuster. (Subcortical "low road" threat processing through the amygdala.)

[10] Gibson, E.J., & Walk, R.D. (1960). "The 'Visual Cliff'." *Scientific American,* 202(4), 64–71.

[11] Yassa, M.A., & Stark, C.E.L. (2011). "Pattern separation in the hippocampus." *Trends in Neurosciences,* 34(10), 515–525.

[12] Bliss, T.V.P., & Lømo, T. (1973). "Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit following stimulation of the perforant path." *Journal of Physiology,* 232(2), 331–356.

[13] French, R.M. (1999). "Catastrophic forgetting in connectionist networks." *Trends in Cognitive Sciences,* 3(4), 128–135.

[14] Hubel, D.H., & Wiesel, T.N. (1962). "Receptive fields, binocular interaction and functional architecture in the cat's visual cortex." *Journal of Physiology,* 160(1), 106–154. (Innate orientation selectivity in V1.)

[15] Damasio, A.R. (1994). *Descartes' Error: Emotion, Reason, and the Human Brain.* Putnam. (vmPFC, Phineas Gage, somatic marker hypothesis.)

[16] Botvinick, M.M., Braver, T.S., Barch, D.M., Carter, C.S., & Cohen, J.D. (2001). "Conflict monitoring and cognitive control." *Psychological Review,* 108(3), 624–652. (ACC conflict monitoring.)

[17] Pearl, J., et al. (2022). "Crowdsourced RNA design discovers diverse, reversible, efficient, self-contained molecular switches." *PNAS,* 119(18). https://doi.org/10.1073/pnas.2112979119

[18] Pearl, J., et al. (2024). "Exploring the Accuracy of Ab Initio Prediction Methods for Viral Pseudoknotted RNA Structures." *JMIRx Bio.* https://doi.org/10.2196/58899

[19] Tse, V., et al. (2025). "OpenASO: RNA Rescue — designing splice-modulating antisense oligonucleotides through community science." *RNA,* 31(8), 1091–1102. https://doi.org/10.1261/rna.080288.124

---

## Appendix A: Source Code

Sara Brain is implemented as an open-source Python package. Source: https://github.com/LunarFawn/SaraBrain

The v1.1 extensions described in this paper are on the `signed_refutation_paths` branch. Key files:

- `src/sara_brain/innate/primitives.py` — All six innate primitive layers (SENSORY, STRUCTURAL, RELATIONAL, ETHICAL, SAFETY, SOCIAL)
- `src/sara_brain/models/segment.py` — Symmetric strength formula, `belief`, `evidence_weight`, `epistemic_state`
- `src/sara_brain/care/urgency.py` — Protective urgency function with trump card
- `src/sara_brain/care/__init__.py` — Module docstring encoding the four foundational principles (need-based priority, trump card, no utilitarian aggregation, KITT not KARR)
- `docs/v014_sleep_consolidation.md` — Sleep and consolidation design document

Test suite: 257 tests, zero failures. Tests structurally enforce the foundational principles:
- `test_epistemic_state.py` — Contested vs fresh distinguishability
- `test_safety_social_primitives.py` — Primitive layer existence and helpers
- `test_protective_urgency.py` — Trump card, lives-are-equal, no aggregation, function signature is per-victim only
