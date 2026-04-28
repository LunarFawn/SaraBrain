"""Additive teach for §9.2.1.4 (KDOFF) and §9.2.1.5 (KDON) — per-sublab
numerical bindings the prior pass left at the qualitative level.

Opens existing aptamer_full.db and adds the specific numbers (1125, 850,
24-26%, 1500, etc.) plus the dual-mode (nominal vs super-performing)
paragraph. Walked sentence-by-sentence per
feedback_teach_faithfully_sentence_by_sentence.

Run from sara_brain/ (where aptamer_full.db lives), then copy result
to sara_test/brains/aptamer_full.db.
"""
from __future__ import annotations

import time
from pathlib import Path

from sara_brain.core.brain import Brain


SOURCE = "aptamer_paper_rev1"


TRIPLES: list[tuple[str, str, str]] = [

    # =====================================================================
    # §9.2.1.4 KDOFF
    # =====================================================================

    # --- SSNG1 KDOFF (line 1428) ---
    # "the highest KDOFF value of approximately 1125 is associated with a
    #  static nucleotide ratio of 20%."
    ("ssng1 highest kdoff", "value", "approximately 1125"),
    ("ssng1 highest kdoff", "at_ratio", "20 percent"),

    # "This ratio is at the upper limit for the ratio and is one of the
    #  two ratios that exhibit 100 Total Eterna Score designs."
    ("20 percent", "is", "upper limit for ssng1 ratio"),
    ("ssng1 20 percent ratio", "exhibits", "100 total eterna score designs"),

    # "The second highest maximum KDOFF observed is 850, which corresponds
    #  to a ratio value of 18%, the other ratio with 100 Total Eterna Scores."
    ("ssng1 second highest kdoff", "value", "850"),
    ("ssng1 second highest kdoff", "at_ratio", "18 percent"),
    ("ssng1 18 percent ratio", "exhibits", "100 total eterna score designs"),

    # "The highest KDOFF values at 20% and 18% ratios are associated with
    #  ensemble groups with fewer than 4000 structures."
    ("ssng1 highest kdoff at 20 percent", "associated_with", "ensemble groups fewer than 4000 structures"),
    ("ssng1 highest kdoff at 18 percent", "associated_with", "ensemble groups fewer than 4000 structures"),

    # "maintaining a static nucleotide ratio of at least 18% is crucial
    #  for achieving higher KDOFF values."
    ("ssng1 minimum ratio for higher kdoff", "value", "at least 18 percent"),

    # "Low relative KDOFF values are associated with ratios less than 18%."
    ("ssng1 low kdoff", "at_ratios", "less than 18 percent"),

    # --- SSNG2 KDOFF (line 1446) ---
    # "the highest KDOFF values are associated with a ratio range of
    #  approximately 24% to 26%."
    ("ssng2 highest kdoff", "at_ratio_range", "24 to 26 percent"),

    # "This range corresponds to the upper limits for the ratio in the
    #  SSNG2 configuration."
    ("24 to 26 percent", "is", "upper limit for ssng2 ratio"),

    # "The highest KDOFF values are linked to structure groups with fewer
    #  than 8000 structures."
    ("ssng2 highest kdoff", "associated_with", "structure groups fewer than 8000 structures"),

    # "maintaining a static nucleotide ratio of at least 24% is essential
    #  for achieving higher KDOFF values."
    ("ssng2 minimum ratio for higher kdoff", "value", "at least 24 percent"),

    # "Low relative KDOFF values are associated with ratios less than 24%."
    ("ssng2 low kdoff", "at_ratios", "less than 24 percent"),

    # --- SSNG3 KDOFF (line 1460) ---
    # "the highest KDOFF value is associated with a static nucleotide
    #  ratio of 19%."
    ("ssng3 highest kdoff", "at_ratio", "19 percent"),

    # "This ratio is just below the upper limit for the ratio, which is
    #  20%, and is also associated with designs that achieve a Total
    #  Eterna Score of around 90 or higher."
    ("ssng3 upper ratio limit", "value", "20 percent"),
    ("ssng3 19 percent ratio", "associated_with", "total eterna score 90 or higher"),

    # "The next highest KDOFF values for the configuration are found at
    #  ratios of approximately 17% and 20%."
    ("ssng3 next highest kdoff", "at_ratios", "17 percent and 20 percent"),

    # "Low relative KDOFF values are associated with ratios below
    #  approximately 10%."
    ("ssng3 low kdoff", "at_ratios", "below approximately 10 percent"),

    # "maintaining a static nucleotide ratio of at least 10% is essential
    #  for achieving higher KDOFF values."
    ("ssng3 minimum ratio for higher kdoff", "value", "at least 10 percent"),

    # "as the ratio of static nucleotides increases from 0% to 20%, the
    #  maximum KDOFF value observed at each progressive ratio increases
    #  linearly."
    ("ssng3 kdoff", "increases_linearly_with_ratio_from", "0 percent to 20 percent"),

    # =====================================================================
    # §9.2.1.5 KDON
    # =====================================================================

    # --- SSNG1 KDON (line 1478) ---
    # "the lowest KDON values are observed with the greatest delta from
    #  their observed KDOFF values at 18% and 20% ratios"
    ("ssng1 lowest kdon", "at_ratios", "18 percent and 20 percent"),

    # "These low KDON values, which do not exceed approximately 175 for
    #  the 20% ratio and approximately 75 for the 18% ratio, are part of
    #  the less than 8000 and less than 4000 structure ensemble groups."
    ("ssng1 kdon at 20 percent", "max_value", "approximately 175"),
    ("ssng1 kdon at 18 percent", "max_value", "approximately 75"),
    ("ssng1 lowest kdon", "in_ensemble_groups", "less than 8000 and less than 4000 structures"),

    # "These values are associated with the highest fold changes
    #  observed for this configuration, as well as the 100 Eterna Total
    #  Scores, indicating the strongest aptamer switches"
    ("ssng1 lowest kdon", "associated_with", "highest fold changes"),
    ("ssng1 lowest kdon", "associated_with", "100 eterna total scores"),
    ("ssng1 lowest kdon configuration", "indicates", "strongest aptamer switches"),

    # "the highest KDON values are observed in the greater than 12000
    #  and greater than 8000 structure ensemble groups, with the maximum
    #  value found to be approximately 680 and the next lower values
    #  ranging from approximately 380 to 500."
    ("ssng1 highest kdon", "in_ensemble_groups", "greater than 12000 and greater than 8000 structures"),
    ("ssng1 highest kdon", "max_value", "approximately 680"),
    ("ssng1 next lower kdon values", "range", "approximately 380 to 500"),

    # --- SSNG2 KDON (line 1488) ---
    # "the lowest KDON values are observed with the greatest delta from
    #  their observed KDOFF values at 24% and 26% ratios"
    ("ssng2 lowest kdon", "at_ratios", "24 percent and 26 percent"),

    # "These low KDON values, which do not exceed approximately 100 for
    #  the 24% ratio and approximately 550 for the 26% ratio, are part of
    #  the less than 8000 and less than 4000 structure ensemble groups."
    ("ssng2 kdon at 24 percent", "max_value", "approximately 100"),
    ("ssng2 kdon at 26 percent", "max_value", "approximately 550"),
    ("ssng2 lowest kdon", "in_ensemble_groups", "less than 8000 and less than 4000 structures"),

    # "These values are associated with the highest fold changes
    #  observed for this configuration, as well as one of the ranges for
    #  100 Eterna Total Score"
    ("ssng2 lowest kdon", "associated_with", "highest fold changes"),
    ("ssng2 lowest kdon", "associated_with", "100 eterna total score range"),
    ("ssng2 lowest kdon configuration", "indicates", "strongest aptamer switches"),

    # "the highest KDON values are observed in the greater than 12000
    #  structure ensemble group, with the maximum value found to be
    #  approximately 800."
    ("ssng2 highest kdon", "in_ensemble_groups", "greater than 12000 structures"),
    ("ssng2 highest kdon", "max_value", "approximately 800"),

    # --- SSNG2 dual-mode paragraph (line 1496) — KEY ---
    # "this sublab has two modes of KDON values: nominal performing
    #  designs and super-performing designs."
    ("ssng2", "has_kdon_mode", "nominal performing designs"),
    ("ssng2", "has_kdon_mode", "super-performing designs"),
    ("nominal performing designs", "is_a", "ssng2 kdon mode"),
    ("super-performing designs", "is_a", "ssng2 kdon mode"),

    # "The nominal mode, seen across SSNG1, SSNG2, and SSNG3
    #  configurations, has a high KDOFF of around 750 and a KDON of
    #  around 100."
    ("nominal mode", "seen_in", "ssng1"),
    ("nominal mode", "seen_in", "ssng2"),
    ("nominal mode", "seen_in", "ssng3"),
    ("nominal mode", "kdoff_value", "around 750"),
    ("nominal mode", "kdon_value", "around 100"),

    # "The super-performing mode, unique to SSNG2, has a KDOFF greater
    #  than 1500 and a relatively low KDON around less than 500."
    ("super-performing mode", "unique_to", "ssng2"),
    ("super-performing mode", "kdoff_range", "greater than 1500"),
    ("super-performing mode", "kdon_range", "less than 500"),
    ("ssng2 super-performing mode kdoff", "value", "greater than 1500"),
    ("ssng2 super-performing mode kdon", "value", "less than 500"),

    # "Both modes are considered successful based on their ratios and
    #  magnitudes, but the difference in KDOFF suggests a difference in
    #  the performance of the unbound state."
    ("nominal mode", "is", "successful"),
    ("super-performing mode", "is", "successful"),
    ("difference in kdoff between modes", "suggests", "difference in unbound state performance"),

    # "This dual-mode observation is crucial for understanding the
    #  performance dynamics of the SSNG2 molecular snare."
    ("dual-mode observation", "crucial_for_understanding", "ssng2 molecular snare performance dynamics"),

    # --- SSNG2 theory paragraph (line 1500) ---
    # "Previous analysis of SSNG2 metrics indicates that this
    #  configuration allows for stronger, more powerful switches when
    #  optimized."
    ("ssng2 configuration", "allows", "stronger more powerful switches when optimized"),

    # "The theory is that the 5'3' static stem being integrated into the
    #  molecular snare stem is what gives this performance boost."
    ("5'3' static stem integrated into molecular snare stem", "causes", "ssng2 performance boost"),

    # "This integration results in a natural limit raise of both KDOFF
    #  and KDON values."
    ("5'3' static stem integration into molecular snare stem", "raises_natural_limit_of", "kdoff"),
    ("5'3' static stem integration into molecular snare stem", "raises_natural_limit_of", "kdon"),

    # "the SSNG2 configuration is much more permissive and allows for
    #  stronger aptamer switches compared to SSNG1 and SSNG3."
    ("ssng2 configuration", "more_permissive_than", "ssng1"),
    ("ssng2 configuration", "more_permissive_than", "ssng3"),
    ("ssng2 configuration", "allows", "stronger aptamer switches"),

    # --- SSNG3 KDON (line 1506) ---
    # "the lowest KDON values are observed with the greatest delta from
    #  their observed KDOFF values at a 20% ratio of static nucleotides."
    ("ssng3 lowest kdon", "at_ratio", "20 percent"),

    # "These low KDON values, which do not exceed approximately 300, are
    #  part of the less than 8000 structure ensemble groups."
    ("ssng3 lowest kdon", "max_value", "approximately 300"),
    ("ssng3 lowest kdon", "in_ensemble_groups", "less than 8000 structures"),

    # "the highest KDON values are observed in the greater than 12000
    #  structure ensemble groups, with the maximum value found to be
    #  approximately 625 and the next lower value at around 440."
    ("ssng3 highest kdon", "in_ensemble_groups", "greater than 12000 structures"),
    ("ssng3 highest kdon", "max_value", "approximately 625"),
    ("ssng3 next lower kdon", "value", "approximately 440"),

    # "The SSNG3 configuration exhibits two modes of operation for KDON
    #  values which are associated with the less than 4000 structure
    #  ensemble groups."
    ("ssng3", "has_kdon_modes_in", "less than 4000 structure ensemble groups"),

    # "One mode is characterized by high KDOFF and low relative KDON
    #  values."
    ("ssng3 kdon mode 1", "characterized_by", "high kdoff and low relative kdon"),

    # "The other mode is characterized by KDON values that are not
    #  relatively low, indicating a less efficient transition from the
    #  unbound to the bound state."
    ("ssng3 kdon mode 2", "characterized_by", "kdon values not relatively low"),
    ("ssng3 kdon mode 2", "indicates", "less efficient unbound to bound transition"),
]


def main() -> None:
    db_path = Path("aptamer_full.db")
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found — run teach_full_paper.py first"
        )

    brain = Brain(str(db_path))
    print(f"Opening existing brain: {db_path}")

    n0 = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    s0 = brain.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    print(f"before: {n0} neurons, {s0} segments")
    print(f"{len(TRIPLES)} additive triples to teach\n")

    t0 = time.time()
    for i, (s, r, o) in enumerate(TRIPLES, 1):
        result = brain.teach_triple(s, r, o, source_label=SOURCE)
        ok = "OK " if result else "FAIL"
        if i % 25 == 0 or not result:
            print(f"[{i:4d}/{len(TRIPLES)}] {ok} | {s[:40]!r} --[{r}]--> {o[:40]!r}")

    elapsed = time.time() - t0

    n1 = brain.conn.execute("SELECT COUNT(*) FROM neurons").fetchone()[0]
    s1 = brain.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]

    print()
    print(f"elapsed: {elapsed:.1f}s")
    print(f"after:  {n1} neurons (+{n1-n0}), {s1} segments (+{s1-s0})")


if __name__ == "__main__":
    main()
