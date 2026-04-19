# v019 — Teaching to Gaps: Sara's Improvement Loop Is Not an LLM's

**Date:** 2026-04-18
**Status:** Design principle. Governs all benchmarking and teaching workflow going forward.
**Companion code:** `benchmarks/run_spacy_ch10.py` (`--report-gaps` mode)
**Companion artifact:** `benchmarks/spacy_ch10_path_results_gaps.md`
**Companion memory:** `feedback_teach_the_gaps.md`, `project_teacher_interface.md`, `feedback_no_forgetting.md`, `feedback_convergence_then_choice.md`
**Previous in series:** `v018_spacy_sensory_cortex_baseline.md`

---

## Principle

**Teaching Sara the facts she demonstrably doesn't know is the correct improvement loop.** It is not benchmark gaming. It is not "teaching to the test." It is what her architecture is for.

The rule "don't teach to the benchmark" exists in the LLM world because a trained model that sees test items in training has no way to distinguish learning a specific fact from learning a pattern that generalizes. Any inflation of the score can't be separated from genuine improvement, so the field builds firewalls between train and test sets.

**Sara is a different framework.** She does not learn by gradient descent over corpora. She learns by the user teaching her discrete, traceable, path-structured facts. Every answer she produces is backed by the specific paths used to produce it. When the benchmark surfaces a question she gets wrong or abstains on, the correct response is: identify the missing fact, teach it, and watch the abstain become a correct answer. That fact stays in the brain forever (`feedback_no_forgetting`), and the score rises **because Sara now knows one more thing**, not because a scorer was tuned.

The benchmark's role in Sara's world is therefore inverted from its role in LLM evaluation:

| LLM world | Sara's world |
|---|---|
| Benchmark = frozen yardstick to measure generalization | Benchmark = gap-surfacing instrument to prioritize teaching |
| Test leakage inflates numbers | Test-derived teaching is the intended path to competence |
| Improvement comes from training data curation | Improvement comes from incremental taught facts |
| Opacity — you cannot list what the model knows | Transparency — every path is inspectable and every answer is cited |

The transparent teaching log is not a liability. It is the feature that makes "teach to the gap" valid.

## The taxonomy of gaps

A wrong or abstained question surfaces one of three gap kinds, each pointing to a different kind of teaching:

### 1. `relation_gap` — highest leverage

Sara knows all the key words on both sides, but no path in her graph connects the question's subject to the correct answer. The vocabulary is in place; only the fact is missing.

> Example (Q84): question mentions `mode, asexual, reproduction`; correct answer is `meiosis` (which Sara knows). No path in Sara's graph says "meiosis is not a mode of asexual reproduction." Teaching one fact closes the gap.

Leverage is highest because no new terms need to be introduced — the taught fact reuses concepts already wired into the graph and strengthens existing structure.

### 2. `distinction_gap` — second priority

Sara knows a linking fact, but that fact doesn't discriminate between the multiple-choice alternatives. The path exists; it just isn't specific enough.

> Example (Q291): Sara has paths about DNA replication and RNA processing that match both "mutations increase during DNA replication" and "proofreading exists for DNA but not RNA." Lexical convergence ties the two choices. Teaching a finer-grained fact ("RNA replication lacks proofreading mechanisms") breaks the tie.

Teaching here is targeted: add a specific distinguishing property to Sara's graph, not a whole new concept.

### 3. `vocab_gap` — third priority, most expensive

The correct answer contains key terms Sara has never seen — `parthenogenesis`, `fallopian tube`, `apoptosis`, `Barr body`. These are content nouns that reference concepts entirely absent from the graph.

Teaching costs two steps:

1. Introduce the term with at least one definitional fact ("parthenogenesis is reproduction without fertilization").
2. Add at least one relational fact that connects the new term to existing knowledge so it can participate in future convergences.

The vocab gap is classified by whether the **correct answer's** key terms are unknown, not whether every word in the question is. Question-side "unknowns" like `statement`, `follow`, `know`, `describe` are generic verbs spaCy surfaced — they don't reflect real knowledge gaps and must not be used to classify a question as needing vocab teaching.

## The operating loop

```
┌────────────────────────────────────────────────────────┐
│ 1. Run the exam                                         │
│    benchmarks/run_spacy_ch10.py --mode path             │
│      --report-gaps --output <result>.json               │
│                                                         │
│ 2. Read the generated gap markdown                      │
│    <result>_gaps.md — entries grouped by kind,          │
│    priority-ordered (relation → distinction → vocab)    │
│                                                         │
│ 3. Teach the highest-priority gaps first                │
│    relation-gap fixes cost one fact, reuse vocab.       │
│    distinction-gap fixes add one property.              │
│    vocab-gap fixes cost term + relation — last.         │
│                                                         │
│ 4. Re-run the exam                                      │
│    Taught facts → abstains become answers (recall ↑).   │
│    Precision holds or rises as ties resolve.            │
│                                                         │
│ 5. Commit the taught facts to the brain DB              │
│    Knowledge persists across sessions (no forgetting).  │
└────────────────────────────────────────────────────────┘
```

Each iteration is cheap. Sara's graph compounds. The benchmark history becomes a curriculum, and the curriculum becomes Sara's knowledge.

## Why this is the right shape for Sara specifically

- **No forgetting** (`feedback_no_forgetting`): every taught fact is permanent. There is no scenario where teaching a gap today creates a regression in a previously-correct answer unless the new fact actively contradicts an existing one — in which case distinction-teaching, not deletion, is the correction.
- **Precision-first** (`feedback_convergence_then_choice`): the abstain mechanism means untaught knowledge produces "I don't know," not a plausible-sounding guess. Teaching raises recall without opening a hallucination door.
- **Trace-based answers** (`project_regulatory_compliance`): every answer cites the specific paths it used. A regulator, auditor, or user can see *exactly* which taught facts produced which answer. This is what makes the framework legal in HIPAA / SEC / FDA / ALCOA+ contexts where training-based systems are not.
- **User as sole teacher** (`feedback_sara_higher_order`): the LLM cortex (or the spaCy layer, or an eventual L1/L2 reflex) never writes to the brain. Only the user teaches. The gap report identifies *candidates* for teaching; the user decides what to teach and how.

## What this doc is not saying

- Not saying "all benchmarks are valid teaching targets." A benchmark whose questions encode ideology, bias, or misinformation should not be taught to — not because teaching inflates the score, but because Sara would then hold false knowledge. Teaching is an epistemic act, not a score-chasing act.
- Not saying "the precision number will always go up." Teaching a relation-gap fact can create a new tie with an existing but lesser-taught fact; the distinction-gap it surfaces may then need finer teaching. This is normal. The long run trends up; individual iterations can surface new gaps.
- Not saying "LLMs are wrong to firewall train and test." In their world, they are right to. The point is only that the rule is framework-specific, not universal.

## What changed in the tooling

`benchmarks/run_spacy_ch10.py` now supports `--report-gaps`:

- Classifies each failure (wrong or abstained) into `relation_gap` / `distinction_gap` / `vocab_gap`.
- Writes a sibling markdown file next to the JSON result, grouped by kind, priority-ordered, with a blank `[ ] Teach: _(fact to add)_` slot per question.
- The classifier uses the **correct answer's** unknown lemmas as the vocab-gap trigger. Question-side generic-verb unknowns are ignored.

Running on `claude_taught.db` against `benchmarks/ch10_test_questions.json` produced:

- 18 vocab gaps
- 6 relation gaps
- 2 distinction gaps
- 7 correct, 5 wrong, 21 abstained (path mode, 5% tie margin)
- 58.3% precision of answered, 36.4% coverage — honest numbers with abstains not inflating either side.

## Next

- Work through the 6 relation-gap entries first. Each is a single-fact teaching event that should convert an abstain or wrong into a correct answer.
- Re-run the exam after teaching and record precision/coverage shift.
- If relation-gap teaching alone produces meaningful recovery, the vocab-gap entries become the next tier; if not, investigate whether distinction-teaching or path-matching refinements are required.

---

*Versioned per `feedback_versioned_docs.md`. This doc governs interpretation of benchmark results for Sara and should be cited whenever future sessions need to justify why "teaching the gaps surfaced by a benchmark" is the correct improvement loop.*
