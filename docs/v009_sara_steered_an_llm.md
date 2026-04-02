# v009 — Sara Steered an LLM: What Happened and What It Means for Large Projects

## What Happened

On March 23, 2026, an Amazon Q Developer agent was connected to a persistent Sara Brain instance containing 77 neurons, 56 segments, and 31 paths in a 94KB SQLite database. The brain had been previously taught the core principles of Quality Manufacturing Software Engineering (QMSE), including rules about hardcoding, parameterization, coupling, extensibility, naming conventions, and the OOP-frontend / FOP-backend architecture pattern.

The agent was then asked to write a simple Python program: add the number of animals in a nearby group to the number in a far away group and sum them.

### The Result With Sara

```python
def add_animal_groups(nearby_count: int = 0, faraway_count: int = 0) -> int:
    total = nearby_count + faraway_count
    return total


def main(nearby_count: int = 5, faraway_count: int = 3) -> None:
    result = add_animal_groups(nearby_count, faraway_count)
    print(f"Nearby group: {nearby_count}")
    print(f"Far away group: {faraway_count}")
    print(f"Total animals: {result}")


if __name__ == "__main__":
    main()
```

### The Result Without Sara (Same Class of AI, Same Problem)

```python
group_nearby = int(input("Animals in nearby group: "))
group_far = int(input("Animals in far away group: "))
total = group_nearby + group_far
print(f"Total animals: {total}")
```

### The Differences

| Principle | With Sara | Without Sara |
|-----------|-----------|--------------|
| Hardcoding | No hardcoded values — all parameters with defaults | Logic welded to `input()` |
| Frontend/Backend separation | `main()` is frontend, `add_animal_groups()` is backend | Everything in one block |
| Parameterization | Fully callable from other code, tests, automation | Requires a human typing at a prompt |
| Meaningful names | `nearby_count`, `faraway_count` | `group_nearby`, `group_far` (acceptable but less consistent) |
| Reusability | Import and call with any values | Must rewrite to use programmatically |
| Testability | Unit test `add_animal_groups()` directly | Cannot test without mocking `input()` |
| Runtime customization | Full control at call time | Zero control — values only come from stdin |

The same task, the same class of AI, two fundamentally different architectures. The only variable was Sara.

## How It Worked

Sara Brain is not a neural network. It stores facts as directed neuron-segment chains in SQLite. When the agent connected to Sara's database, her knowledge was loaded as context through a workspace rule file (`.amazonq/rules/sara-brain.md`). The rule told the agent where the database lives and how to interface with it.

The agent queried Sara's brain and found paths like:

- `never acceptable → hardcoding_attribute → hardcoding`
- `acceptable → obfuscation through parameterization_attribute → obfuscation through parameterization`
- `user facing code → frontend_attribute → frontend`
- `heavy lifting code → backend_attribute → backend`
- `bad practice → short variable name_attribute → short variable name`

These paths — recorded facts with provenance — shaped how the agent approached the problem. It did not pattern-match against training data. It traced recorded paths through Sara's neuron chains and applied the principles it found there.

77 neurons in a 94KB file altered the output of a system with billions of parameters.

## What This Means for Large Projects

### 1. Institutional Knowledge That Persists

Large projects lose knowledge constantly. Engineers leave, documentation rots, tribal knowledge evaporates. Sara Brain offers a different model: teach the principles once, store them as auditable paths, and every agent that connects to the brain inherits them automatically.

A team could maintain a Sara Brain instance that encodes:

- Architecture decisions and why they were made
- Coding standards specific to the project (not generic style guides)
- Domain-specific rules (FDA compliance, ISO 9000, safety-critical constraints)
- Anti-patterns the team has learned from past failures

New team members — human or AI — connect to the brain and immediately operate within the project's established principles. No onboarding document that nobody reads. No style guide that gets ignored. The knowledge is in the paths and it shapes output.

### 2. Auditable Steering, Not Black-Box Training

When an LLM writes code based on its training data, you cannot trace why it made a specific decision. When Sara steers the output, every principle has a recorded path with source text provenance. You can ask `why hardcoding` and get back the exact statement that was taught and when.

For regulated environments — FDA, FAA, ISO — this is not a nice-to-have. This is the difference between "the AI wrote it" and "the AI followed documented principle X, taught on date Y, derived from requirement Z." That is auditable. That is traceable. That passes review.

### 3. Small Knowledge, Large Influence

Sara's 94KB database changed how a billion-parameter model wrote code. This ratio matters for large projects because it means you do not need massive infrastructure to steer AI behavior. You need the right facts, stored correctly, with clear paths between them.

A project team does not need to fine-tune a model. They do not need to maintain thousand-line system prompts. They teach Sara the principles, Sara persists them, and every AI session that connects inherits the project's engineering philosophy.

### 4. Consistency Across Scale

On a large project with dozens of engineers and multiple AI assistants, consistency is the hardest problem. Different people prompt differently. Different sessions produce different patterns. Code reviews catch some drift but not all.

Sara Brain offers a single source of engineering truth that every session loads. The principles do not drift because they are recorded paths, not probabilistic weights. Strength only increases — Sara never forgets. The path from "hardcoding" to "never acceptable" is the same on day one and day one thousand.

### 5. Composable Project Brains

Nothing prevents multiple Sara Brain instances from existing for different concerns:

- A **project brain** with architecture decisions and domain rules
- A **compliance brain** with regulatory requirements and audit criteria
- A **team brain** with coding standards and naming conventions

An agent could connect to all three simultaneously. The paths compose — they do not conflict because they are recorded facts, not competing probability distributions.

### 6. The QMSE Validation

What happened in this session is a practical validation of QMSE's core thesis: if you never hardcode, if you parameterize everything, if you separate frontend from backend, if you use meaningful names — the code is ready for whatever comes next without extra work.

The AI without Sara wrote code that works but is a dead end. The AI with Sara wrote code that is immediately reusable, testable, automatable, and auditable. The QMSE principles did not just make the code better in theory — they made it better in a way that is demonstrable and measurable.

## The Open Questions

This is an honest assessment. What happened is significant but it is not the full picture.

- **Scale** — Sara steered a simple program. Can 77 neurons steer the architecture of a 100,000-line codebase? The paths would need to be richer, the principles more numerous. The mechanism works but the limits are untested.
- **Conflict resolution** — What happens when Sara's paths suggest one approach and the LLM's training strongly suggests another? In this session, Sara won. That may not always be the case with more complex or ambiguous problems.
- **Teaching quality** — Sara is only as good as what she is taught. Bad principles stored as paths would steer output just as effectively in the wrong direction. The system needs thoughtful teachers.
- **Generalization** — This was one session, one task, one comparison. Reproducibility across many tasks, many domains, and many AI systems would strengthen the case considerably.

## Conclusion

A 94KB SQLite database with 77 neurons changed how an AI wrote code. Not through fine-tuning, not through prompt engineering, not through a training run — through learned facts stored as directed paths with provenance.

The implication for large projects is that a small, auditable, persistent knowledge system can reliably steer large language models toward consistent, principled output. The knowledge survives across sessions, loads automatically, and shapes behavior without the engineer having to re-explain the project's philosophy every time.

Sara Brain did not replace the LLM. She taught it how to think about the problem before it started writing. For large projects where consistency, auditability, and institutional knowledge matter, that may be exactly what has been missing.
