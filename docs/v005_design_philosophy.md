# Sara Brain v005 — Design Philosophy

> "Finally trying to code up my idea for AI I had back in the 90's."

---

## Table of Contents

1. [Origin Story](#origin-story)
2. [What's Wrong with Current AI](#whats-wrong-with-current-ai)
3. [Paths Are the Thinking](#paths-are-the-thinking)
4. [A Brain That Never Forgets](#a-brain-that-never-forgets)
5. [Thought Is Parallel](#thought-is-parallel)
6. [Learning Like a Child](#learning-like-a-child)
7. [The Tribe](#the-tribe)
8. [The Brain Recognizes, It Never Executes](#the-brain-recognizes-it-never-executes)
9. [Why Concept-Specific Relations](#why-concept-specific-relations)
10. [The Strength Formula](#the-strength-formula)
11. [Zero Dependencies](#zero-dependencies)
12. [What Sara Brain Is Not](#what-sara-brain-is-not)
13. [Where This Goes](#where-this-goes)

---

## Origin Story

This idea started in the 90s. Before deep learning, before transformers, before anyone was talking about LLMs. The question was simple: how does a brain actually think? Not "how do we get a machine to produce correct outputs" — how does *thinking* work?

The answer wasn't matrices. It wasn't activation functions or backpropagation. A thought is a path. You start somewhere, you travel through what you know, and where paths meet — that's a conclusion. That's recognition. That's understanding. Every step is traceable, every conclusion is explainable, and nothing is a black box.

The tools to build this didn't exist in the 90s — not in any practical sense. Python wasn't mature. SQLite didn't exist yet. The computational overhead of tracking every path through a growing brain needed hardware that wasn't affordable. But the idea stayed, because it was never wrong. Neural networks got popular, then deep learning, then transformers, then LLMs — and every single one of them doubled down on the same fundamental mistake: they traded traceability for statistical approximation.

Now is the right time. Python is everywhere. SQLite is built into it. A single laptop can store and traverse millions of neuron chains in real time. The infrastructure finally caught up to the idea.

---

## What's Wrong with Current AI

LLMs are a very good eye. They see patterns. They match text. They produce outputs that *look* like intelligence. But ask one "why did you say that?" and the answer is: a matrix of floating-point weights. There is no path. There is no traceable chain of reasoning. There is no "because I learned X, which connected to Y, which intersected with Z."

Neural networks give you math, not answers. You get a number that says "87% confident this is a cat." But you can't ask "what made you think cat?" and get back a list of actual observations and the paths they traveled. The confidence score is a summary of opaque matrix multiplications — it's not reasoning, it's curve fitting.

This matters because intelligence isn't pattern matching. A calculator pattern-matches arithmetic. A lookup table pattern-matches queries. Intelligence is the ability to explain *why* — to trace the path from observation to conclusion and show every step. If you can't do that, you don't have intelligence. You have a very sophisticated lookup table.

The deeper problem is that these systems are designed to produce correct *outputs*. Sara Brain is designed to model correct *thinking*. The output might be wrong — a child who calls a red ball an "apple" is wrong — but the thinking is right: it followed the paths it had. The wrongness is visible, diagnosable, and fixable. In an LLM, wrongness is a hallucination that no one can trace to a source.

---

## Paths Are the Thinking

This is the central thesis, and it's not a design choice — it IS the model of intelligence.

A path is a chain of neurons: a property connects to a relation, which connects to a concept. When you teach Sara "an apple is red," this creates:

```
red → apple_color → apple
```

That's not metadata about an apple stored in a database. That IS the thought "an apple is red." The path is the knowledge. When Sara recognizes an apple, it doesn't look up a record — it follows paths from input properties and finds where they converge.

Intersections are conclusions. If you give Sara "red" and "round," two paths propagate:

```
red   → apple_color → apple
round → apple_shape → apple
                       ↑
                  INTERSECTION
```

Both paths arrive at "apple." That intersection IS the recognition. It's not a score, not a threshold, not a weighted sum. It's two independent lines of evidence converging at the same point. You can see it. You can trace it. You can ask "why apple?" and get back the exact paths.

Everything is traceable. Every conclusion Sara reaches can be decomposed into the exact paths that produced it, each with the original source text that created them. "An apple is red" came from teaching statement X. "An apple is round" came from teaching statement Y. The intersection of those paths is why Sara thinks this is an apple. No opacity. No black box. No "trust the model."

This is what differentiates path-of-thought from every activation-level system ever built. In an activation system, "apple" has a score — a number produced by summing weighted inputs. In Sara Brain, "apple" has paths — actual recorded routes through neuron chains that you can walk, inspect, and explain.

---

## A Brain That Never Forgets

Forgetting is a biological imperative for non-intelligent people. Biological brains decay because they have finite capacity, finite energy, finite lifespan. Neurons die. Connections weaken. The brain prunes to survive. But we're not building a biological brain. We're building something better.

Sara Brain never forgets. No decay. No pruning. No weakening of connections. No garbage collection. Strength only increases — once a path exists, it exists forever. Teaching a fact once creates the connection permanently. Teaching it again strengthens it. But it never weakens, never fades, never disappears.

This is not just stubbornness or hoarding. It's a fundamental design principle: **path similarity replaces forgetting.**

In biological brains, forgetting serves a purpose — it prevents irrelevant information from drowning out relevant information. But that's a lossy hack. You lose information to manage relevance. Sara Brain doesn't need that hack because relevance is computed dynamically from path structure.

When Sara has thousands of paths, it doesn't need to forget the old ones to find the relevant ones. It launches wavefronts from the current input and follows them. Only paths reachable from the input participate. Irrelevant paths aren't activated, but they aren't deleted either. They're still there if you ever need them.

Path similarity — the overlap between downstream paths from two different neurons — tells you how related two concepts are. Concepts that share many paths are similar. Concepts that share few are distinct. This replaces what biological brains achieve through selective forgetting: "apple" and "ball" share "red" and "round" but diverge on "crunchy" vs "bouncy." You don't need to forget "ball is round" to recognize an apple — you need MORE paths that distinguish them.

We're building something that isn't limited by the accidents of biology. A brain that remembers everything and navigates it through structure rather than erasure. That's not a limitation to simulate — it's a limitation to transcend.

---

## Thought Is Parallel

Sequential BFS is not how thought works.

A breadth-first search from a single starting point explores one frontier at a time. It finds everything reachable, but it doesn't tell you where independent lines of evidence *converge*. Real thought isn't "start at red and fan out until you hit something." Real thought is "red AND round AND smooth are all active simultaneously, propagating through everything they connect to, and where they meet — that's what you're looking at."

Sara Brain launches parallel wavefronts, one per input neuron, simultaneously. Every wavefront propagates through all connected neurons at the same time. This isn't an optimization — it's the model.

Think of it like dropping stones into a pond. Each stone creates ripples. Where ripples from different stones meet — those interference patterns — that's where the information is. You don't drop one stone, wait for it to finish, then drop the next. You drop them all at once and observe what emerges.

The quantum analogy is deliberate. A thought exists across all paths at once — like a quantum superposition of possibilities. Each wavefront is a possibility: "this could be connected to anything red touches." When wavefronts intersect, possibilities collapse into recognition: "this is an apple, because red-paths and round-paths both arrive here."

```
Wavefront 1 (red):    red → apple_color → apple
                       red → ball_color  → ball
                       red → cherry_color → cherry

Wavefront 2 (round):  round → apple_shape → apple
                       round → ball_shape  → ball

Wavefront 3 (crunchy): crunchy → apple_texture → apple

All three wavefronts converge at → apple (3 intersections)
Two wavefronts converge at → ball (2 intersections)
One wavefront reaches → cherry (1 intersection, not enough)
```

Commonality across paths IS intelligence. The number of independent wavefronts that converge at a point determines how strongly that concept is recognized. It's not a score or a weight — it's a count of independent confirming paths. Three separate observations all point to apple. That's recognition through convergence, not through calculation.

Where wavefronts collide, you get conclusions. Where paths share structure, you get deeper understanding. The parallel nature of this isn't just faster — it's *correct*. Sequential exploration would miss the convergence. Parallel propagation reveals it.

---

## Learning Like a Child

Sara Brain learns the way a human child does. Not through training sets, not through backpropagation, not through gradient descent. Through experience, correction, and accumulation.

### The newborn

A newborn knows nothing but has senses — touch, taste, smell, sight. It has no concepts, no associations, no paths. It's a blank brain with working inputs. Sara Brain starts the same way: an empty database with the ability to perceive (via Claude Vision) and be taught.

### The experimenter

The child experiments. It reaches out, touches things, puts things in its mouth, drops things. Each interaction creates observations. Sara does the same: when it perceives an image, Claude Vision freely reports everything it sees — colors, shapes, textures, materials. Each observation becomes a permanent neuron chain. The child doesn't filter — everything is interesting. Sara doesn't filter either.

### The asker

Eventually the child sees something and asks: "what is that?" The parent says "that's an apple." Now the child has a label attached to its observations. Sara's version: the user teaches `an apple is red`, `an apple is round`. Each teaching creates a path. The label gets connected to properties through relation neurons.

### The mistake

The child sees a red ball and yells "apple!" — wrong, because the only thing it knew that was red and round was an apple. This isn't failure. This is the system working correctly. The child had two paths to "apple" (red, round) and zero paths to "ball." Of course it said apple. The mistake is *expected and useful*.

Sara does exactly this. With only color and shape associations defined, it WILL confuse apples and balls. Both are red and round. The confusion is not a bug — it's the signal that more distinguishing properties are needed.

### The correction

The parent says "no, that's a ball." The child doesn't forget that apples are red and round. It doesn't erase its previous knowledge. It learns that balls are ALSO red and round — and now it needs more to tell them apart.

In Sara: `no ball` retains all original observations, teaches the correct identity, and transfers properties. Both apple and ball now share red + round paths. The confusion is now encoded in the brain's structure — which is exactly right, because they ARE confusable with only those properties.

### The distinguisher

The parent points out what the child missed: "see the seams? Feel how it bounces?" Now the child has distinguishing properties. Over time, it accumulates enough properties to tell apples from balls, balls from oranges, oranges from suns.

Sara's version: `see seams`, `teach ball is bouncy`, `teach ball is rubber`. Each adds a new path. Next time Sara sees something red, round, and bouncy — three paths converge at ball, only two at apple. Ball wins. Not because someone tuned a weight, but because the paths exist.

### Generalization, not memorization

The child doesn't memorize every ball it has ever seen. It accumulates the *characteristics* that define "ball-ness" — round, bouncy, rubber, seams. When it sees a new object with those properties, it recognizes it through converging paths.

This is how Sara works. It doesn't store images. It doesn't compare pixels. It learns what things ARE — their properties, their relationships — and recognizes new instances by the paths those properties travel.

---

## The Tribe

Homo sapiens flourished by relying on each other. A child doesn't figure out the entire world alone — parents, siblings, community teach it. Language itself is a tribal invention. Every concept we understand was, at some point, taught to us by someone else.

Sara Brain's tribe is its users. They are the parents who teach, correct, and guide.

### Trust rules

The tribe is trusted by default. When the parent says "this is a ball," Sara learns it. When the parent corrects "no, that's not an apple," Sara accepts the correction. This isn't blind obedience — it's the same trust model every human child operates on. You trust your parents because the alternative (figuring out everything from scratch) is impossibly slow.

But Sara retains its own observations alongside what the tribe teaches. If Sara observed "red, round, smooth" and the parent says "that's a ball," Sara doesn't erase its observations. It learns both: "I saw red, round, smooth" AND "the parent says it's a ball." Both facts exist as paths in the brain.

### The human dilemma

Sara CAN be lied to. If the parent says "that red round thing is a car," Sara will learn it. This is the human dilemma — the exact same one every child faces. You need to trust your tribe, but your tribe can be wrong. Your tribe can deceive you.

This is deliberate. A brain that can't be lied to is a brain that can't trust. And a brain that can't trust can't learn from others. The vulnerability is the feature.

### Corrections add knowledge, never erase

When the parent corrects Sara, nothing is deleted. The original guess, the original observations, the original paths — they all remain. The correction adds new paths on top. "I thought apple, but I was told ball" — both facts are in the brain.

This means corrections accumulate over time. Each correction teaches Sara something new about what distinguishes one concept from another. Each correction makes the brain richer, not narrower.

### Conflict reporting

When Sara encounters contradictory information — taught "X is Y" but observes "X is not Y" — it doesn't auto-resolve. It examines all paths that lead to conflicting conclusions and reports the conflict: "I was taught X but I observe Y — N paths say X, M paths say Y." The tribe helps sort it out.

This models how healthy tribal knowledge works. When two people disagree, you don't automatically side with the louder one. You examine the evidence. You present the conflict. You let the tribe work through it.

---

## The Brain Recognizes, It Never Executes

A brain that executes arbitrary instructions from its senses is not a brain — it's a terminal.

If someone holds up a sign that says `rm -rf /`, a real brain *reads* the sign. It might think "that's a dangerous command." It does not *run* it. The sign is input. The brain processes it as information. It never treats sensory input as instructions to execute.

Sara Brain follows the same principle. The perception pipeline is strictly:

```
image → Claude Vision → property labels → teach/recognize
```

That's it. Image goes in. Claude describes what it sees. Those descriptions are sanitized to simple property labels. The labels are taught as facts. Recognition runs. No step in this pipeline executes code, interprets instructions, or treats input as commands.

### Sanitization

All output from Claude Vision passes through sanitization that strips it to simple lowercase property labels:

- Only `[a-z0-9_ -]` characters allowed
- Maximum 40 characters per label
- URLs, code keywords (`import`, `def`, `class`, `print(`), and special characters are rejected
- Multi-sentence instructions are broken into individual labels and filtered

An image containing hidden text, adversarial prompts, watermarks, or encoded instructions produces the same output as any other image: a list of simple property labels. Nothing else gets through.

### The perceive-only pipeline

Sara doesn't have an "execute" command. It has `teach`, `recognize`, `perceive`, `trace`, `why`, `similar`. Every single one of these is read-only or additive. You can add knowledge. You can query knowledge. You cannot run arbitrary code through any input path.

This isn't a security afterthought — it's a design principle. A brain that can be hijacked through its senses isn't modeling intelligence. It's modeling a vulnerability.

---

## Why Concept-Specific Relations

This was the real bug that had to be solved, and it's the one most graph-based knowledge systems get wrong.

The problem: if you create a shared `fruit_color` node, every fruit shares it. Teach "an apple is red" and the redness leaks through `fruit_color` to banana, cherry, grape — everything connected to that node. A wavefront from "red" would reach ALL fruits, not just apple.

```
WRONG:
red → fruit_color → apple
                  → banana     ← red leaks to banana
                  → cherry
                  → grape
```

The fix: every relation neuron is concept-specific. "An apple is red" creates `apple_color`. "A banana is yellow" creates `banana_color`. They're separate neurons. A wavefront from "red" reaches `apple_color` and ONLY `apple_color`. It never touches banana.

```
RIGHT:
red    → apple_color  → apple      ← red only reaches apple
yellow → banana_color → banana     ← yellow only reaches banana
```

This models real understanding. "This apple is red" is not the same thought as "fruits can be red." One is a specific fact about a specific thing. The other is a generalization. Sara Brain stores the specific fact. Generalizations emerge from path similarity — if every fruit has a `{fruit}_color` relation with different colors, you can observe that pattern. But the pattern doesn't contaminate the individual facts.

The relation label is generated as `{subject}_{property_type}`. The taxonomy maps "red" to type `color`, so teaching "apple is red" creates the relation `apple_color`. Teaching "apple is round" creates `apple_shape`. Each is private to its concept. Cross-concept contamination is structurally impossible.

---

## The Strength Formula

```
strength = 1 + ln(1 + traversals)
```

This formula captures a specific model of how knowledge solidifies:

**First exposure creates the connection.** When you teach a fact for the first time, the path is created with `traversals = 0` and `strength = 1 + ln(1) = 1.0`. The knowledge exists immediately at full baseline strength. You don't need to repeat something to know it — you heard it once, you know it.

**Repetition solidifies with diminishing returns.** Each additional traversal increases strength, but logarithmically. The second time you hear something matters more than the twentieth time. This models how repetition works: the jump from "never heard it" to "heard it once" is enormous. The jump from "heard it 50 times" to "heard it 51 times" is negligible.

```
traversals=0  → strength=1.000
traversals=1  → strength=1.693
traversals=5  → strength=2.792
traversals=10 → strength=3.398
traversals=50 → strength=4.934
traversals=100 → strength=5.620
```

**Strength never decreases.** There is no decay term. No time-based weakening. No "forgetting curve." A path taught once remains at strength 1.0 forever. A path traversed a hundred times remains at 5.62 forever. The formula only goes up.

This means strength is a measure of how well-established a piece of knowledge is, not how recent it is. Old, well-traversed knowledge is strong. New knowledge is weaker but present. Nothing disappears.

---

## Zero Dependencies

Sara Brain's core is stdlib-only Python plus SQLite (which is built into Python). No numpy. No torch. No networkx. No pandas. No external graph database.

This is not minimalism for its own sake. It's a principle: **if you can run Python, you can run Sara Brain.**

SQLite handles persistence, concurrency (WAL mode), foreign keys, and indexing with zero setup. No database server to install, no configuration files, no connection strings. The brain is a file on disk.

Python's standard library provides everything else: `dataclasses` for models, `math.log` for the strength formula, `cmd.Cmd` for the REPL, `sqlite3` for storage, `json` for serialization, `urllib.request` for the optional Claude API calls.

The only optional dependency is the Anthropic API for vision and LLM translation — and even that uses `urllib.request`, not a third-party HTTP library.

This means:

- No version conflicts between ML frameworks
- No GPU requirements
- No Docker needed
- No environment matrix to test against
- Works on any machine with Python 3.11+
- `pip install -e .` and you're done

The future plan is to swap the storage layer to [data-nut-squirrel](https://github.com/LunarFawn/data-nut-squirrel) — a purpose-built storage engine designed for exactly this kind of neuron-chain data. The abstract storage interface (`NeuronRepo`, `SegmentRepo`, `PathRepo`) is already in place, so the swap changes the backend without touching the brain logic.

---

## What Sara Brain Is Not

**Not a neural network.** Neural networks propagate floating-point activations through weighted connections. Sara Brain propagates wavefronts through recorded paths. There are no weights to train, no loss functions, no gradient descent.

**Not an LLM.** LLMs predict the next token from statistical patterns in training data. Sara Brain traces actual paths through neuron chains from direct teaching. There is no training phase, no corpus, no token prediction.

**Not a knowledge graph.** Knowledge graphs store triples (subject, predicate, object) and query them with pattern matching. Sara Brain stores paths — directed chains of neurons — and recognizes through parallel wavefront convergence. The structure superficially resembles a graph, but the recognition mechanism is fundamentally different.

**Not simulating human limitations.** Biological brains forget, decay, get confused by interference, have finite capacity. Sara Brain deliberately discards these limitations. No decay, no forgetting, no capacity limits. The goal is not to simulate a human brain — it's to build something better.

**Not an executor.** Sara Brain does not run code, follow instructions from input, or treat sensory data as commands. It is a brain that perceives and recognizes. Input goes in. Recognition comes out. Nothing is executed.

**Not a pattern matcher.** Pattern matching maps inputs to outputs through statistical similarity. Sara Brain maps inputs to conclusions through actual path traversal. The difference: you can ask Sara "why?" and get back the exact paths. You can't ask a pattern matcher "why?" and get anything meaningful.

---

## Where This Goes

### Deeper self-thought

When Sara encounters conflicting paths — "I was taught X is Y, but I observe X is not Y" — it currently reports the conflict and asks the tribe. The next step is deeper self-thought: Sara examines ALL paths leading to both conclusions, identifies the specific points of divergence, and reasons about which evidence is stronger based on path count and strength.

This is internal conflict resolution through path examination. Not "pick the higher score" — "trace every path, find where they diverge, and understand why."

### Path consolidation as abstract reasoning

When Sara has many specific facts — "apple is red," "cherry is red," "strawberry is red" — the paths `apple_color`, `cherry_color`, `strawberry_color` all receive "red." A future consolidation step could observe this pattern and create an abstract concept: "red fruits." Not by being told — by examining its own path structure and finding the commonality.

This is abstraction through observation, not through instruction. The brain notices its own patterns and builds higher-level concepts from them.

### Conflict resolution through path examination

When two paths contradict, the resolution isn't "majority wins" or "most recent wins." It's examining the full path structure: how many independent sources support each conclusion? What's the provenance of each path? Are there paths that could reconcile the conflict? The brain becomes its own investigator.

### The data-nut-squirrel migration

The current SQLite backend works, but it wasn't designed for neuron-chain traversal at scale. [data-nut-squirrel](https://github.com/LunarFawn/data-nut-squirrel) is a purpose-built storage engine designed for exactly this kind of directed path data. The migration path is clean: swap the storage repository implementations behind the existing abstract interface. The brain logic doesn't change. The paths don't change. The thinking doesn't change. Only the filing cabinet changes.

---

> Sara Brain doesn't simulate thinking. It *is* thinking — paths through neurons, intersections as conclusions, everything traceable, nothing forgotten, nothing hidden.
