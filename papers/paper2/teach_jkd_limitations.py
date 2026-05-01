"""Teach Sara the "Utilizing all limitations as way, and all ways as limitation"
concept from Chapter 3 of Jennifer Pearl's JKD paper (paper2).

Targeted teach for the Session B orthogonality test. Focus is the Chapter 3
Creed 2 section — the novel inversion of Bruce Lee's original maxim and the
growth-through-noticing-limitations interpretation.

Triples drawn from paper2_full.txt lines 663-1082, per-fact judgment
sentence-by-sentence per feedback_hand_teach_not_bulk and
feedback_teach_faithfully_sentence_by_sentence.

Output: jkd_limitations.db (fresh brain). Load via
    cd ../../../sara_test && ./load_brain.sh jkd_limitations
"""
from __future__ import annotations

import time
from pathlib import Path

from sara_brain.core.brain import Brain


SOURCE = "paper2_jkd_chapter3_creed2"


TRIPLES: list[tuple[str, str, str]] = [

    # =====================================================================
    # The concept itself — Jennifer's novel Creed 2
    # =====================================================================

    ("utilizing all limitations as way and all ways as limitation", "is_a", "jeet kune do creed"),
    ("utilizing all limitations as way and all ways as limitation", "is_a", "how to do jeet kune do creed"),
    ("utilizing all limitations as way and all ways as limitation", "is_a", "creed 2"),
    ("utilizing all limitations as way and all ways as limitation", "novel_inversion_of", "using no way as way having no limitation as limitation"),
    ("utilizing all limitations as way and all ways as limitation", "added_by", "jennifer pearl"),
    ("utilizing all limitations as way and all ways as limitation", "builds_on", "creed 1"),

    # Creed 1 (Bruce Lee's original) — context
    ("using no way as way having no limitation as limitation", "is_a", "bruce lee jkd principle"),
    ("using no way as way having no limitation as limitation", "is_a", "creed 1"),
    ("using no way as way having no limitation as limitation", "has_essence_of", "yin yang"),
    ("creed 1", "comparable_to", "yin yang"),

    # Relationship between Creed 1 and Creed 2
    ("creed 2", "is_missing_puzzle_piece_for", "creed 1"),
    ("creed 2", "is_explicit_callout_for_how_to_implement", "creed 1"),
    ("creed 2", "furthers_explanation_of", "creed 1 as yin yang"),

    # =====================================================================
    # What the concept MEANS — Jennifer's interpretation
    # =====================================================================

    ("utilizing all limitations as way and all ways as limitation", "represents", "the process one must take to grow"),
    ("utilizing all limitations as way and all ways as limitation", "enables", "successfully intercepting opponent symptoms"),
    ("utilizing all limitations as way and all ways as limitation", "enables_interception_regardless_of", "symptom severity"),
    ("growth through creed 2", "produces", "ability to intercept any opponent symptom"),

    # The central growth-through-limitations idea
    ("noticing a limitation", "shows", "the way"),
    ("the way", "reveals", "new limitations when applied"),
    ("new limitations exposed by applying a way", "drive", "learning new ways around them"),
    ("this pattern", "is_a", "circular growth"),
    ("life", "is_a", "circular staircase"),
    ("each time around the staircase", "offers", "choice to use the way and limitations learned prior"),
    ("next time around", "allows", "skillful interception"),

    # =====================================================================
    # How Jennifer discovered the concept — the Guro Bob low kick story
    # =====================================================================

    # Origin of the insight
    ("the insight", "came_during", "training session with guro bob"),
    ("the insight", "came_from", "learning new low kicks from guro bob"),
    ("guro bob", "is_a", "jennifer pearl's jkd teacher"),

    # The low kick specifics
    ("the low kick", "intercepts", "opponent kick"),
    ("the low kick", "impacts", "opponent lower legs"),
    ("the low kick", "is_a", "bruce lee longstreet demonstration kick"),
    ("the low kick", "analogous_to", "stiff arm for the leg"),
    ("the low kick", "prevents", "opponent closing the gap"),
    ("the low kick when it lands", "allows", "taking back combat lead"),

    # Jennifer's hip limitation
    ("jennifer pearl", "has", "hip disability"),
    ("jennifer pearl", "has", "hip impingement"),
    ("jennifer pearl", "has", "severe arthritis"),
    ("jennifer pearl", "has", "multiple impinged joints"),
    ("jennifer pearl", "cannot", "lift leg high for a kick"),
    ("jennifer pearl age", "is", "45"),

    # The realization
    ("for the low kick specifically", "not_lifting_leg_higher_is", "a good thing"),
    ("the low kick execution", "emphasizes", "rotate the hip more than lift the leg"),
    ("jennifer pearl", "can_do_well", "hip rotation"),
    ("jennifer pearl", "cannot_do_well", "leg lifting"),
    ("the hip limitation", "revealed_itself_through", "the low kick application"),
    ("the low kick", "worked_out_well_for", "jennifer pearl"),
    ("the low kick", "is_a", "practical new skill jennifer can use"),
    ("the low kick", "addresses_gap_in", "jennifer pearl's defensive moves"),
    ("prior defense against kicks", "limited_to", "using arms to block"),

    # The reframe as acceptance
    ("not trying to do a powerful high kick at this moment", "is_not", "refusal to do something hard"),
    ("not trying to do a powerful high kick at this moment", "is", "acceptance of limits"),
    ("not trying to do a powerful high kick at this moment", "is", "acceptance of defeat of the 20 year old self"),
    ("focus on the low kick", "respects", "the limitations of the moment"),
    ("focus on the low kick", "is", "executing that which is efficient for jennifer in combat"),

    # =====================================================================
    # Bag of ways — the core metaphor for accumulated skill
    # =====================================================================

    ("bag of ways", "is_a", "metaphor for accumulated skill"),
    ("bag of ways", "behaves_like", "bag of holding"),
    ("bag of ways", "does_not_behave_like", "a list to grab from"),
    ("a list of combat moves", "conjures", "images of thinking mid-combat"),
    ("thinking in combat", "leads_to", "getting hit"),
    ("thinking in combat", "is_a", "dangerous failure mode"),

    # How the bag works
    ("conditioning the soul", "prepares", "the bag of ways"),
    ("a conditioned bag of ways", "responds_to", "feelings and emotions"),
    ("a conditioned bag of ways", "retrieves", "the trained skill when reached into"),
    ("reaching into a conditioned bag of ways", "happens_in_a_manner_that_is", "calm and flowing"),

    # Observable effects
    ("calmness during combat", "is", "observable to others"),
    ("calmness during combat", "diffuses", "tense situations"),
    ("others sensing confidence and honest soul expression", "produces", "peaceful diffusion of tension"),

    # The chained moves idea
    ("combat moves", "can_be_chained", "if studied well"),
    ("combat moves", "have_no", "preplanned combos"),
    ("after executing a leg check", "options_include", "stepping back"),
    ("after executing a leg check", "options_include", "devastating chain attack"),
    ("signaling to stop their advance", "is_a_valid", "combat goal"),
    ("relentless advance to halt opponent", "is_a_valid", "combat response"),

    # =====================================================================
    # Ethical dimension — proper minimum escalation
    # =====================================================================

    ("guro bob", "has_instilled_in_jennifer", "do not use more force than necessary"),
    ("clipping the jaw", "is_alternative_to", "shattering the jaw"),
    ("combat goal", "is_not_just", "succeed in combat"),
    ("combat goal", "must_adhere_to", "ethics"),
    ("jennifer pearl", "hates_the_thought_of", "killing"),
    ("killing in self defense", "would_make_jennifer", "feel bad about herself"),
    ("breaking an arm when required to stop threat", "would_make_jennifer", "feel good about herself and her decisions"),
    ("feeling good about combat decisions", "comes_from", "doing what had to be done and nothing more"),

    # =====================================================================
    # Physical combat vs mental combat — applying to autism/PTSD
    # =====================================================================

    ("creed 2", "applies_to", "physical combat"),
    ("creed 2", "applies_to", "mental combat"),
    ("mental combat example", "is", "trying not to let autism symptoms drive inappropriate reactions at work"),
    ("mental combat example", "is", "coworker interactions triggering autism symptoms"),
    ("all combat", "is", "interception of symptoms"),
    ("the interception question", "is", "how and when to intercept"),

    # The interception mechanic itself
    ("intercepting a symptom", "involves", "mad dash to bag of ways"),
    ("intercepting a symptom", "hopes", "what comes out of bag is useful"),
    ("a properly conditioned bag", "responds_to", "current emotion"),
    ("a properly conditioned bag", "produces", "trained skill"),

    # =====================================================================
    # The circular-staircase life framing
    # =====================================================================

    ("life feeling stale", "arises_when", "same situation recurs"),
    ("life feeling stale", "reframed_as", "circular staircase"),
    ("each stair-revolution", "provides", "choice to reuse way and limitations from prior revolution"),
    ("taking the choice to reuse prior learning", "leads_to", "preparation for next occurrence"),
    ("preparation for next occurrence", "means", "skillful interception and surviving with ethics and soul intact"),

    # =====================================================================
    # Closing — why the concept is NEW but also a missing piece
    # =====================================================================

    ("creed 2", "is", "new idea jennifer recently developed"),
    ("creed 2", "is_not_so_much_a_new_concept_as", "a missing piece of the puzzle"),
    ("creed 2", "is_essentially_explicit_callout_for", "how to implement creed 1"),
    ("creed 2", "is_taught_at", "jennifer pearl's jkd school"),
    ("jennifer pearl's jkd school", "opened_in", "early 2026"),

    # =====================================================================
    # Author provenance
    # =====================================================================

    ("jennifer pearl", "also_known_as", "cricket"),
    ("jennifer pearl", "authored", "the way of life's intercepting fist"),
    ("the way of life's intercepting fist", "is_a", "personal work on jkd and autism"),
    ("the way of life's intercepting fist", "subtitle", "my progression through jeet kune do that reduced the severity of my autism and ptsd symptoms"),
    ("the way of life's intercepting fist", "copyright_year", "2025"),
    ("the way of life's intercepting fist", "is", "1st edition"),
]


def main() -> None:
    db_path = Path("jkd_limitations.db")
    if db_path.exists():
        raise FileExistsError(
            f"{db_path} exists — delete or rename before re-teaching"
        )

    brain = Brain(str(db_path))
    print(f"Fresh brain: {db_path}")
    print(f"{len(TRIPLES)} triples to teach (JKD Chapter 3, Creed 2 concept)\n")

    t0 = time.time()
    for i, (s, r, o) in enumerate(TRIPLES, 1):
        result = brain.teach_triple(s, r, o, source_label=SOURCE)
        ok = "OK " if result else "FAIL"
        if i % 10 == 0 or not result:
            print(f"[{i:3d}/{len(TRIPLES)}] {ok} | {s[:45]!r} --[{r}]--> {o[:45]!r}")

    elapsed = time.time() - t0

    neuron_count = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    segment_count = brain.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    path_count = brain.conn.execute("SELECT COUNT(*) FROM paths").fetchone()[0]

    print()
    print(f"elapsed: {elapsed:.1f}s")
    print(f"triples taught: {len(TRIPLES)}")
    print(f"neurons: {neuron_count}  segments: {segment_count}  paths: {path_count}")


if __name__ == "__main__":
    main()
