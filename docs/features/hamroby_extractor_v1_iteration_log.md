# hamroby_extractor_v1 — iteration log (aug2 → aug9)

How the extractor went from "broken on conjunctions and terse acronym
subjects" to "58/58 on the extended battery." Eight retrains over one
session. The architecture didn't change at the model layer — every fix
was in the training pipeline (data structure, feature source, parser
choice). The lessons are about *where* the bugs were, not *how big* the
model needed to be.

This doc is a retrospective. The canonical state is documented in
[hamroby_extractor_v1.md](hamroby_extractor_v1.md). For the formal
plans driving the work, see:

- `~/.claude/plans/we-need-to-build-golden-toast.md` — rule-stub fixes
- `~/.claude/plans/lexical-orbiting-matsumoto.md` — round 1 (aug, aug2, aug3)
- `~/.claude/plans/lets-plan-the-fix-smooth-snowglobe.md` — rounds 2 & 3 (aug4–aug9)

## Where it started

The trained head was a grammar-feature-only transformer (POS, dep,
offset, funcword IDs per word → BIO span tags). It worked well on most
inputs but had 4 named persistent failures:

1. `K_d for the binding is 1.2nM.` → garbled (2 wrong triples)
2. `Marker theory predicts kdoff with p<0.05.` → verb swallowed into subject
3. `The 5'3' static stem provides stability.` → rule stub only failure
4. `Noticing a limitation shows the way.` → subject truncated to "Noticing"

The rule stub had its own four failures (mostly overlapping). Both
extractors are different code paths; fixing one doesn't fix the other.

## The iteration table

| run | change | result | takeaway |
|---|---|---|---|
| aug | baseline: synthetic-only, spaCy-sm, one-Pair-per-conjunct | 4 named failures + 5 PRON+conj failures | starting point |
| aug2 | added `t_with_oblique`, `t_conjoined_subject`, `t_intj_subject_copular`; funcword PROPN gating | Marker theory, John+Mary, Bruce Lee fixed. K_d partial. 5'3' regressed. | first augmentation pass; targeted but uneven |
| aug3 | added `t_intj_subject_pp_copular` (PP-modified INTJ subject); PP_MODIFIER_PROB=0.30 | K_d fully fixed. Marker theory regressed. 5'3' restored. | each retrain plays whack-a-mole; variance dominates with only 13 test sentences |
| aug4 | added `t_conjoined_object` (bare-noun conjuncts), `t_compound_oblique` (forced 2-word subjects), `by` to oblique preps | intj_pp 6/6, compound_oblique 5/5, conj_subject 4/4. PRON+conj still NO TRIPLE. | extended battery to 47 sentences — the wins crystallized |
| aug5 | added pronoun-subject variants to `t_conjoined_object` (30% of emissions) | no change on PRON+conj — model still emitted NO TRIPLE | pronoun *frequency* wasn't the issue |
| aug6 | added spaCy-NOUN filter on `t_conjoined_object` (drop ~27% pairs where conj got tagged PROPN) | PRON+conj now emit ONE conjunct (always the second), not both | feature noise wasn't the whole story; some structural issue lurked |
| aug7 | plumbed **gold UD features** through `real_prose_pairs.py`: new `pre_parsed: ParsedSentence` field on `Pair`, UD-native delex, `_ud_to_parsed` builder | conj_object behavior barely changed | gold features fix the *input* but not the *supervision* |
| aug8 | added `additional_object_spans` to `Pair`; multi-B-O label generation; UD triples grouped by (subject, relation); `t_conjoined_object`/`t_list_object_two`/`t_list_object_three` emit ONE Pair per scene | **conj_object 12/12 emit BOTH conjuncts**. 2 over-extractions remain (`similarity`, `second`). DNA/RNA still fails. | structural fix at the supervision layer; the bug was in how we taught, not what we taught |
| aug9 | `ud_triple_extractor.py`: `obl` is an object only when no `obj` exists. spaCy cascade: sm primary + trf fallback when no VERB/AUX in parse. | **58/58 on the extended battery** (1 intransitive correctly returns NO TRIPLE) | last two failures fixed by parser cascade + obl-handling — both at inference boundary |

## The bug we kept missing

Across aug2–aug7 we poured energy into the **input side**: better
templates, more variety, cleaner features, gold UD annotations,
synthetic POS filters, pronoun-subject coverage. Each retrain shifted
results but no run cleanly handled conjoined objects with pronoun
subjects.

The actual bug was on the **supervision side**. The synthetic templates
and `real_prose_pairs.py` both emitted **one `Pair` per conjunct**:

```
"She bought apples and oranges."
  → Pair A: prose, subject="She", relation="bought", object="apples"
  → Pair B: prose, subject="She", relation="bought", object="oranges"
```

During training, the same prose appeared in two examples with
contradictory BIO labels:

- Example A: `She/B-S, bought/B-R, apples/B-O, and/O, oranges/O`
- Example B: `She/B-S, bought/B-R, apples/O,   and/O, oranges/B-O`

The loss pulled in opposite directions on the same input. The model
converged on "pick one." Multi-B-O output was emergent, not trained —
it worked on some checkpoints by seed luck and broke on others.

We *touched* this bug repeatedly: every new template inherited the
per-conjunct emission shape from `t_list_object_two`. None of our
"improvements" addressed the structural conflict.

The fix in aug8 was small: a single Pair carries the full multi-object
label set. `char_spans_to_word_bio` produces multiple B-O spans for
multiple conjuncts. The model now trains directly on "tag every
conjunct" — no competing examples.

This unlocked 10 of 12 sentences in the extended `conj_object`
category in one retrain.

## Lessons

### 1. When iterations don't converge, the bug isn't where you're looking

aug2 → aug7 each made principled changes to the input side and each
produced *different* output (sometimes better, sometimes worse, never
cleanly fixed). When five iterations of careful tuning don't converge,
the bug isn't in what you're tuning. **Look one layer deeper.** For us,
that meant the label generation, not the feature extraction.

### 2. "Emergent good behavior" is unstable

In aug3/aug4, `The system processes books and papers.` correctly
emitted both `books` and `papers` as separate triples. We treated this
as a working state. It wasn't — it was a coincidence of the loss
landscape, and it broke under later changes. **If a behavior isn't
explicitly supervised, it isn't reliable.**

The conj_object behavior was emergent because the per-conjunct Pair
labels accidentally balanced. As soon as new training data shifted the
balance, the emergence collapsed.

### 3. Gold features ≠ multi-object supervision

aug7 was a real architectural improvement: real-prose pairs now carry
gold UD features instead of running spaCy on delexicalized text. This
matters — measured noise on the spaCy-on-delex path was 46% POS
mismatch in conj positions, vs. ~0% for gold UD.

But aug7 didn't fix the conj_object failure. The supervision-side bug
dominated. Gold features delivered the right INPUT distribution; the
LABEL distribution was still wrong.

Both fixes are needed; neither alone is sufficient.

### 4. Cascade > single-model parser

aug9 introduced `_CascadeNLP`: spaCy `en_core_web_sm` as primary, with
`en_core_web_trf` only as fallback when the primary parse has no VERB
or AUX. Most sentences (99%+) parse fine with sm at ~5ms; degenerate
parses (terse all-caps subjects like `DNA and RNA share...`)
transparently retry on trf at ~30ms.

Single-model trf is 5-10x slower than sm on CPU. Single-model sm
fails on `DNA and RNA share base pairing.` and similar. The cascade
gives most of trf's accuracy at most of sm's speed.

Pattern is reusable: a cheap-but-imperfect primary + an expensive-but-
accurate fallback + a fast triage check on the primary's output.

### 5. Per-category test batteries beat single-sentence diagnostics

The original failure list was 4 sentences. We could "fix" any of them
by chance with a retrain. Extending to 47 sentences (then 58) across 13
categories — `intj_pp`, `compound_oblique`, `conj_subject`,
`conj_object`, `intj_bare`, `weird_token_no_pp`, `gerund_subject`,
`propn_aux`, `svo_basic`, `particle_verb`, `copular_simple`,
`intransitive`, `pron_svo` — made retrain variance distinguishable
from real improvement.

A category at 3/4 was always mid-fix. A category at 6/6 was reliable.
The honest progress measure was per-category pass rates, not "the four
original failures." Promotion gates were defined on the categories.

### 6. The user-side rule: "don't ship partial as won"

Across this session, the assistant repeatedly framed partial fixes as
wins ("3/4 conj_subject!"). The user pushed back and forced honest
accounting. The promotion criterion was tightened after aug3 to require
*all* targeted-category sentences to produce sensible triples, with
explicit known-hard exemptions documented and the underlying parser
limitation traced. That discipline prevented multiple premature
promotions.

## The final architecture (aug9, canonical)

Files touched, from data layer down to parser layer:

| layer | file | change |
|---|---|---|
| training data shape | [synthetic_pairs.py](../../src/sara_brain/cortex/transformer/v2/synthetic_pairs.py) | `Pair.additional_object_spans` + `Pair.pre_parsed`; conj-emission templates produce one Pair per scene with multi-object label sets |
| training data source | [real_prose_pairs.py](../../src/sara_brain/cortex/transformer/hamroby_extractor_v1/real_prose_pairs.py) | UD-native delex (1:1 token alignment), `_ud_to_parsed` for gold features, triples grouped by (subj, rel) |
| triple extraction from UD | [ud_triple_extractor.py](../../src/sara_brain/cortex/transformer/hamroby_extractor_v1/ud_triple_extractor.py) | `obl` is object only when no `obj` exists; copular predicates always skip `obl` |
| BIO label generation | [synthetic_features.py](../../src/sara_brain/cortex/transformer/hamroby_extractor_v1/synthetic_features.py) | `char_spans_to_word_bio` accepts a list of object spans; each emits its own B-O span; `pair_to_example` honors `pre_parsed` |
| inference parser | [feature_extractor.py](../../src/sara_brain/cortex/transformer/hamroby_extractor_v1/feature_extractor.py) | `_CascadeNLP` wraps sm + trf; `load_domain_nlp` returns cascade by default |
| eval harness | [scripts/compare_checkpoints.py](../../scripts/compare_checkpoints.py) | 58-sentence battery, 13 category tags, per-category summary |
| diagnostic | [scripts/diagnose_failure_battery.py](../../scripts/diagnose_failure_battery.py) | uses cascade nlp; includes the round-3 follow-up failure cases |

The model class, feature ID vocabulary, decoder, and BIO tag set
didn't change. Same 13MB checkpoint shape from aug to aug9.

## What we'd do differently next time

- **Audit the supervision shape before tuning the data.** When the data
  generator emits multiple training examples sharing a prose, draw the
  full BIO label set for each and check if they conflict. We could have
  caught the per-conjunct-Pair conflict by looking at five sample
  proses before launching aug2.
- **Define the test battery upfront and freeze the promotion gates.**
  Adding categories and sentences mid-iteration was useful but slowed
  diagnosis. With a stable battery and a fixed gate, retrain variance
  becomes the only noise source.
- **Don't bake spaCy choice into training without a cascade plan.** sm
  is fast and wrong sometimes; trf is slow and accurate. The right
  default is the cascade — it should have shipped in aug.

## Post-promotion: wiring into ingest and the hybrid discovery

After aug9 was promoted to canonical, we tried the extractor end-to-end
on a real source — the JKD book (`papers/paper2/paper2_full.txt`,
~35k words, 3343 clauses). Three rounds of work followed:

### Round 1: wire the trained head into `cli_teach_book`

The CLI was hardcoded to call the rule stub. A new module
[inference.py](../../src/sara_brain/cortex/transformer/hamroby_extractor_v1/inference.py)
exposes `extract_triples(clause, nlp) -> list[Triple]` matching the
rule stub's signature, so `cli_teach_book` can swap extractors without
other changes:

```
python -m sara_reader.cli_teach_book BOOK --extractor trained --brain X.db
python -m sara_reader.cli_teach_book BOOK --extractor rules   --brain Y.db
python -m sara_reader.cli_teach_book BOOK --extractor hybrid  --brain Z.db
```

The trained-head extractor uses the canonical `hamroby_extractor_v1.pt`
plus the sm+trf cascade nlp. Committed in `2b71ed7`.

### Round 2: comparing rule stub vs trained head on real prose

Ran the same book through both extractors. Then counted actual term
mentions in the source with `grep -oiw` and compared to brain edge
counts:

| term | actual mentions | rule-stub edges | trained-head edges |
|---|---:|---:|---:|
| water | 34 | 32 | 26 |
| ptsd | 5 | 4 | 1 |
| autism | 18 | 45 | 19 |
| bruce lee | 35 | 49 | 35 |
| jeet kune do | 53 | 116 | 36 |
| jkd | 15 | 36 | 0 |
| flow | 12 | 53 | 9 |
| philosophy | 7 | 20 | 5 |
| meditation | 1 | 5 | 4 |

(Edge counts are direct segments touching the term, not the transitive
neighborhood — the latter expands by 10-100× and is misleading for
comparison.)

The two extractors produce qualitatively different brains:

- **Trained head** edge counts track actual term-frequency density.
  `water` (34 mentions → 26 edges), `ptsd` (5 → 1), `bruce lee`
  (35 → 35), `jeet kune do` (53 → 36) all roughly match the source.
- **Trained head misses POBJ / compound-NP / oblique mentions.** `jkd`
  (15 mentions → 0 edges) is the smoking gun: every JKD mention in
  the book is buried as "to JKD", "of JKD", "JKD concepts", "JKD
  journey". The BIO tagger only extracts direct subject/object roles.
- **Rule stub catches the buried mentions** via `part_of` NP
  decomposition: emits `jkd part_of "the foundations of jkd"`,
  `jkd part_of "the natural progression of a person who has dedicated
  their life to jkd"`. Inflated counts (`jeet kune do` 53 mentions →
  116 edges) but those extra edges are real contexts the term appears
  in, not phantom data.

The user reframed it: **"noise is where the information is."** For
their writing style (long run-on sentences, JKD as POBJ, idiomatic
"flow", autobiographical pronoun-heavy prose), the rule stub's `part_of`
decomposition captures the discourse context the trained head leaves
on the floor.

### Round 3: hybrid extractor

Added `--extractor hybrid` to `cli_teach_book`. Runs BOTH extractors
per clause and commits all triples. `brain.teach_triple` is idempotent
on identical edges, so duplicates collapse.

Coverage with hybrid (direct edge counts):

| term | rules-only | trained-only | **hybrid** |
|---|---:|---:|---:|
| jkd | 36 | 0 | **36** |
| jeet kune do | 116 | 36 | **143** |
| bruce lee | 49 | 35 | **65** |
| autism | 45 | 19 | **55** |
| water | 32 | 26 | **84** |
| flow | 53 | 9 | **61** |
| philosophy | 20 | 5 | **25** |

Hybrid wins on every probed term. 4,275 triples total (vs 1,941 rules,
2,340 trained), 130s vs 48s/83s. Committed in `b186511`.

Sample of what hybrid captures for `flow` (the contextual richness the
trained head alone missed):

```
flow part_of "the flow"
flow part_of "the flow of the manuscript"
flow part_of "a large amount of reflection of the past in this manuscript"
flow part_of "the flow of this turbulent raging river"
flow part_of "some flow of something that you are moving against"
flow part_of "the flow of my thoughts"
flow part_of "going with the flow"
flow_attribute describes flow
"the situation" with flow_attribute
"the flow" help with that_attribute
"the severity of my autism" reduce that_attribute
"bruce lee's expression of his own unique mind" be that_attribute
```

These are real contexts. The rule stub's `part_of` flood IS useful
information when the underlying prose uses terms as buried modifiers
rather than direct verb arguments.

### Round 4 finding: chat routing can't surface the brain's content

Tried `cli_stateless_chat --brain X --cortex-router --explore-first
--strict-sara`. The LLM router routinely called `brain_define(concept)`
and got "No definitional edges found" — even on the hybrid brain with
36-143 edges per term. The router then either reported "no info" or
filled in from general LLM knowledge.

**Cause:** `brain_define` looks for `X is Y` definitional triples. Our
extractors produce verb-form triples (`jeet kune do achieve...`) and
`part_of` triples — neither matches the definitional pattern. The
routing layer needs to use `brain_explore` / `brain_neighborhood` as a
fallback (or as the primary tool for these prose-derived brains), not
just `brain_define`.

The brain content is good. Surfacing it via chat is a separate fix.

### Lessons added by this phase

7. **Test extraction on a real source before declaring victory.** The
   58-sentence synthetic battery said "58/58 clean." But running the
   trained head on real prose immediately exposed the POBJ /
   compound-NP gap. Synthetic tests verify the model does what its
   training data trained it to do; real-prose tests verify the
   training data covered the right patterns. We had the former; we
   needed the latter.

8. **Edge inflation isn't always noise.** The rule stub's `part_of`
   pattern produces 5-10× the edges per term vs the trained head. We
   initially framed this as "noise dressed as breadth." The user
   pushed back: those extra edges encode the COMPOUND NPs each term
   appears in, which IS information about how the term is used in
   discourse. Calling it noise was wrong; it's a different *kind* of
   information than verb-driven SVO.

9. **Hybrid pipelines are cheap and powerful when extractors are
   complementary.** Two extractors that produce different kinds of
   triples (verb-driven SVO vs NP-decomposition) can both feed the
   same brain. `brain.teach_triple` idempotency makes the merge
   automatic. Coverage strictly improves; cost is only the slower of
   the two extractors.

10. **A "fixed" extractor doesn't mean a "useful" brain.** The trained
    head passed every synthetic test; the routing-layer-on-strict-mode
    behavior shows it doesn't matter if the brain has content the
    query layer can't reach. Extractor accuracy and brain-query
    surfacing are independent problems.

## Pointers

- Canonical state and architectural overview: [hamroby_extractor_v1.md](hamroby_extractor_v1.md)
- Retrain recipe: [scripts/train_hamroby_extractor_aug9.sh](../../scripts/train_hamroby_extractor_aug9.sh)
- Evaluation: `BASELINE=<old.pt> CANDIDATE=<new.pt> .venv/bin/python scripts/compare_checkpoints.py`
- Diagnostic: `HAMROBY_CHECKPOINT=<x.pt> .venv/bin/python scripts/diagnose_failure_battery.py`
- Ingest with hybrid: `python -m sara_reader.cli_teach_book BOOK --extractor hybrid --brain X.db`
