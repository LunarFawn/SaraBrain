# Sara Brain v003 — Perception via Claude Vision

> Sara Brain learns to see the way a child does: observe, guess wrong, get corrected, learn what distinguishes one thing from another.

---

## Table of Contents

1. [The Cognitive Development Model](#the-cognitive-development-model)
2. [How Perception Works](#how-perception-works)
3. [The Tribal Trust Model](#the-tribal-trust-model)
4. [The Correction & Teaching Mechanism](#the-correction--teaching-mechanism)
5. [The "Taught What to Observe" Principle](#the-taught-what-to-observe-principle)
6. [Security](#security)
7. [REPL Commands](#repl-commands)

---

## The Cognitive Development Model

### The child analogy

A newborn knows nothing but has senses — touch, taste, smell, sight. It experiments and tests until it understands. Then it sees something new and asks a parent "what is that?" and is taught. Later it sees a red ball and yells "apple!" — wrong, because it only knew "red" and "round." The parent corrects: "no, that's a ball." Now the child needs to learn what distinguishes a ball from an apple (bouncy vs crunchy, rubber vs organic, hollow vs solid). Over time, it accumulates enough distinguishing properties to tell them apart — not by memorizing every ball in existence, but by understanding the *characteristics* that define "ball-ness" vs "apple-ness."

This is exactly how Sara Brain's perception works.

### Why misidentification is expected and useful

A brain with only `color` and `shape` associations WILL confuse apples and balls — both are red and round. This is not a bug; it's how learning works. The confusion drives the brain to seek more distinguishing properties. After being corrected, Sara knows that both apple and ball share red + round, so next time it MUST find more properties (seams, texture, material) to tell them apart.

### Generalization from characteristics, not memorization

Sara doesn't memorize every image of a ball it has ever seen. Instead, it accumulates properties that define "ball-ness" — round, bouncy, rubber, seams. When it sees a new object with those properties, it recognizes it as a ball through converging paths, not through pixel matching.

### The role of the tribe

Homo sapiens flourished by relying on each other. A child doesn't figure out the entire world alone — parents, siblings, and community teach it. Sara's "tribe" is the user who corrects mistakes, points out missed details, and teaches new concepts. The tribe accelerates learning by providing corrections and observations Sara would otherwise have to discover on its own.

---

## How Perception Works

### The perception loop

```
Phase 1 — INITIAL OBSERVATION (the baby looks)
  → Claude Vision freely describes everything it sees
  → Sara teaches itself each observation as a fact
  → Sara tries to recognize from all observed properties
  → Result: best guess (may be wrong — that's expected)

Phase 2 — DIRECTED INQUIRY (the baby reaches out)
  → Sara checks what associations it knows (color? texture? taste?)
  → For each not yet observed, asks Claude specific questions
  → New observations are taught and recognition re-runs
  → Repeats until convergence or max rounds

Phase 3 — SUSPICION VERIFICATION (the baby checks)
  → Sara looks up properties of its top guess
  → For properties not yet observed, asks Claude to verify
  → Confirmed properties strengthen the recognition

Phase 4 — CORRECTION (the parent says "no, that's a ball")
  → User corrects: `no ball`
  → Sara retains all original observations
  → Teaches the correct concept the observed properties
  → Both concepts now share some paths — next time Sara must
    find more distinguishing properties

Phase 5 — PARENT TEACHES (the parent points out)
  → User notices something Sara missed: `see seams`
  → Sara learns this property for the image
  → User can then teach: `teach ball is seams`
  → Over time, distinguishing properties accumulate
```

### Claude Vision as Sara's senses

Claude Vision acts as Sara's eyes. When Sara perceives an image:

1. **Claude freely reports everything** — colors, shapes, textures, patterns, materials, markings, distinguishing features. Claude volunteers novel observations (e.g., "has seams") even if Sara has no existing association for seams.
2. **Sara doesn't filter** — all observations are taught as facts, creating permanent neuron chains.
3. **Sara's existing knowledge directs deeper inquiry** — if Sara knows about "taste" and "texture" associations, it asks Claude about those specifically.

The key distinction: Claude is a sense organ that freely reports what it sees. Sara is the brain that learns from those reports and tries to recognize what it's looking at.

### Observations become permanent neuron chains

Every observation follows the existing teach pipeline:

```
observe "red" → teach "img_photo is red"
  → creates: red → img_photo_color → img_photo
```

This is the same 3-neuron chain as any taught fact. The image concept (`img_photo`) is a standard CONCEPT neuron — no new types needed. Each observation creates a permanent path with full provenance.

### How existing knowledge directs inquiry

If Sara has defined associations for color, shape, taste, and texture, the directed inquiry phase asks Claude about each one. But if Sara has zero associations defined, it still gets the initial free-form observations from Claude. Associations don't gate perception — they deepen it.

---

## The Tribal Trust Model

### Sara trusts its tribe but retains its own observations

When a parent corrects Sara:
- If the parent gives a reason → Sara listens and learns
- If the parent says "trust me" → Sara listens and learns
- Sara **never deletes** its own observations — it retains both what it saw and what it was told

### Corrections add knowledge, they never erase

When `no ball` is issued:
1. Sara's original observations (red, round, smooth) remain as facts about the image
2. The correct identity is taught: `img_photo is ball`
3. All observed properties are transferred: `ball is red`, `ball is round`, etc.
4. Now both apple and ball share red + round paths

Nothing is erased. The original "apple" guess remains in the path history. The correction adds new knowledge on top.

### The human dilemma

Sara CAN be lied to. If the parent says "that red round thing is a car," Sara will learn it. This is the human dilemma — you need to trust your tribe, but reality is reality. If Sara later encounters contradictory information (e.g., observes a car that is definitely not red and round), the conflicting paths will be present in the brain.

### Reassessment (future: deep self-thought)

When Sara encounters contradictory information — taught "X is Y" but observes "X is not Y" — it examines ALL paths that lead to conflicting conclusions. Sara reports the conflict ("I was taught X but I observe Y — N paths say X, M paths say Y") rather than auto-resolving. The tribe helps sort out the conflict.

---

## The Correction & Teaching Mechanism

### `no <correct_label>` — correcting a misidentification

After perceiving an image that was guessed as "apple":

```
sara> no ball
  Corrected: not apple, this is ball.
  Taught ball: red, round, smooth, shiny, small
  (Original observations retained — Sara never erases.)
```

This teaches:
- `img_photo is ball` (identity)
- `ball is red`, `ball is round`, etc. (property transfer)

### `see <property>` — parent points out what Sara missed

```
sara> see seams
  Taught img_photo_abc123 is seams.
```

The parent noticed seams that Claude didn't report (or that Sara didn't ask about). This is how the tribe helps: "see how it has seams — an apple doesn't have that."

The parent can then build distinguishing knowledge:
```
sara> teach ball is seams
sara> teach ball is bouncy
sara> teach ball is rubber
```

### Worked example: apple vs ball confusion

1. Sara perceives a red ball → observes red, round, smooth
2. Sara recognizes "apple" (only concept it knows with red + round)
3. Parent corrects: `no ball` → ball now has red, round, smooth
4. Parent teaches: `see seams`, `teach ball is bouncy`, `teach ball is rubber`
5. Next time Sara sees something red + round:
   - If it also has seams → ball (3 converging paths vs 2 for apple)
   - If it also is crunchy → apple (3 converging paths vs 2 for ball)
6. The more distinguishing properties Sara learns, the better it gets

---

## The "Taught What to Observe" Principle

### Why associations matter for directed perception

The directed inquiry phase only fires for known associations. If Sara has no "taste" association defined, it won't ask Claude "what does this taste like?" But Claude's free-form initial observation works regardless.

### How a naive brain still perceives

A brand-new brain with zero associations defined:
- Still gets initial observations from Claude (color, shape, texture, etc.)
- These are taught as facts with whatever property type the taxonomy assigns
- Recognition works on these observations
- No directed inquiry fires (nothing to ask about)

### How associations unlock deeper perception

Once you define associations:
```
sara> define material what
sara> describe material as rubber, plastic, wood, metal, organic
```

Now the directed inquiry phase will ask Claude about material, giving Sara richer observations to work with.

---

## Security

### Sara is a brain that recognizes, never an executor

**Critical constraint**: Sara Brain NEVER executes code. The perception pipeline is strictly:

```
image → Claude Vision → property labels → teach/recognize
```

### All vision output is sanitized

The `VisionObserver._sanitize()` method strips all Claude output to simple lowercase property labels:
- Only `[a-z0-9_ -]` characters allowed
- Maximum 40 characters per label
- URLs, code keywords (`import`, `def`, `class`, `print(`), and special characters are rejected
- Multi-sentence instructions are broken into individual labels and filtered

### No prompt injection from images

Images could contain hidden text, watermarks, or adversarial prompts. The sanitization layer ensures that no matter what Claude responds with, only simple property labels reach the brain. No raw Claude output is ever passed to command dispatchers, eval, or exec.

---

## REPL Commands

### `perceive <path> [label]` — full perception loop

```
sara> perceive /photos/fruit.jpg
  [initial]
    Observed: red, round, smooth, shiny, small, stem
    Taught 6 facts.
    Recognition: apple (2 converging paths)
      red → img_fruit_abc123_color → img_fruit_abc123
      round → img_fruit_abc123_shape → img_fruit_abc123
  [directed-1]
    Observed: sweet
    Taught 1 fact.
    Recognition: apple (3 converging paths)
  [verification]
    Observed: crunchy
    Taught 1 fact.
    Recognition: apple (4 converging paths)
  Perception of img_fruit_abc123:
    Image: /photos/fruit.jpg
    Total observations: 8
    Total facts taught: 8
    Final recognition: apple (4 converging paths)
```

### `no <correct_label>` — correct a misidentification

```
sara> no ball
  Corrected: not apple, this is ball.
  Taught ball: red, round, smooth, shiny, small, stem, sweet, crunchy
  (Original observations retained — Sara never erases.)
```

### `see <property>` — parent points out a missed property

```
sara> see seams
  Taught img_fruit_abc123 is seams.
```
