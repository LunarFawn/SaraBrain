"""Teach Sara the Executive Summary of the aptamer paper — LLM-parsed triples.

Teacher-surrogate (Claude) walks each of the 40 Executive-Summary
sentences, decides what (subject, relation, object) claims the sentence
asserts, and hands them to Sara via Brain.teach_triple. No parser in
the loop — compound terms ("molecular snare", "Marker Theory", etc.)
land in the graph verbatim.

Follows feedback_llm_parses_sara_stores (LLM is the parser),
feedback_hand_teach_not_bulk (per-fact judgment), and
feedback_teach_faithfully_sentence_by_sentence (walk every sentence,
no curation).

Source: papers/aptamer_rev1/aptamer_paper_full.txt, §2 Executive
Summary (lines 257–290).
"""
from __future__ import annotations

import time
from pathlib import Path

from sara_brain.core.brain import Brain


SOURCE = "aptamer_paper_rev1_exec_summary"


# Triples grouped by paragraph and sentence. Each triple is a faithful
# rendering of a claim the source sentence makes. Novel compound terms
# the paper coins are preserved verbatim as neuron labels.
TRIPLES: list[tuple[str, str, str]] = [

    # =====================================================================
    # Paragraph 1 — framing
    # =====================================================================

    # P1 S1: "This paper explores the integration of computational models
    # and experimental data to optimize RNA aptamer designs, emphasizing
    # the role of mechanical forces in RNA stability and functionality."
    ("the paper", "integrates", "computational models"),
    ("the paper", "integrates", "experimental data"),
    ("the paper", "aims_to_optimize", "rna aptamer design"),
    ("rna aptamer design", "emphasizes", "mechanical forces"),
    ("mechanical forces", "role_in", "rna stability"),
    ("mechanical forces", "role_in", "rna functionality"),

    # P1 S2: "The study highlights the innovative use of crowdsourcing
    # through Eterna Game Labs to solve RNA folding problems, and
    # introduces key theories such as Marker Theory, Switch Acceptance
    # Theory, and the Theory of Mechanics of RNA Folding Model."
    ("the study", "uses", "crowdsourcing"),
    ("crowdsourcing", "done_through", "eterna game labs"),
    ("eterna game labs", "solves", "rna folding problems"),
    ("the study", "introduces", "marker theory"),
    ("the study", "introduces", "switch acceptance theory"),
    ("the study", "introduces", "theory of mechanics of rna folding model"),
    ("marker theory", "is_a", "theory"),
    ("switch acceptance theory", "is_a", "theory"),
    ("theory of mechanics of rna folding model", "is_a", "theory"),

    # P1 S3: "These theories provide a comprehensive framework for
    # predicting RNA performance and optimizing RNA aptamer designs."
    ("marker theory", "predicts", "rna performance"),
    ("switch acceptance theory", "predicts", "rna performance"),
    ("theory of mechanics of rna folding model", "predicts", "rna performance"),
    ("marker theory", "optimizes", "rna aptamer design"),
    ("switch acceptance theory", "optimizes", "rna aptamer design"),
    ("theory of mechanics of rna folding model", "optimizes", "rna aptamer design"),

    # P1 S4: "The Serena RNA Analysis Tool is presented as a sophisticated
    # instrument for evaluating RNA structures, offering metrics like
    # Ensemble Variation, Local Minima Variation, Weighted Structures,
    # and Comparison Structures."
    ("serena rna analysis tool", "is_a", "instrument"),
    ("serena rna analysis tool", "evaluates", "rna structures"),
    ("serena rna analysis tool", "offers_metric", "ensemble variation"),
    ("serena rna analysis tool", "offers_metric", "local minima variation"),
    ("serena rna analysis tool", "offers_metric", "weighted structures"),
    ("serena rna analysis tool", "offers_metric", "comparison structures"),
    ("ensemble variation", "is_a", "metric"),
    ("local minima variation", "is_a", "metric"),
    ("weighted structures", "is_a", "metric"),
    ("comparison structures", "is_a", "metric"),

    # P1 S5: "The paper also delves into the molecular snare mechanism,
    # the importance of axial forces, and the dynamics of the fold
    # signal region."
    ("the paper", "describes", "molecular snare mechanism"),
    ("the paper", "describes", "axial forces"),
    ("the paper", "describes", "fold signal region"),
    ("molecular snare mechanism", "involves", "molecular snare"),
    ("axial forces", "important_to", "rna aptamer design"),

    # =====================================================================
    # Paragraph 2 — 5'3' Static Stem Mechanics of Materials Hypothesis
    # =====================================================================

    # P2 S1: "The 5'3' Static Stem Mechanics of Materials Hypothesis
    # provides a detailed explanation of the mechanical forces at play
    # within the 5'3' Static Stem of RNA aptamers."
    ("5'3' static stem mechanics of materials hypothesis", "is_a", "hypothesis"),
    ("5'3' static stem mechanics of materials hypothesis", "explains", "mechanical forces"),
    ("mechanical forces", "act_within", "5'3' static stem"),
    ("5'3' static stem", "part_of", "rna aptamer"),

    # P2 S2: "This hypothesis outlines a step-by-step process that
    # describes how the cumulative negative axial forces generated by
    # each nucleotide pair contribute to the overall stability and
    # functionality of the RNA structure."
    ("5'3' static stem mechanics of materials hypothesis", "describes_process", "cumulative negative axial forces"),
    ("cumulative negative axial forces", "generated_by", "nucleotide pair"),
    ("nucleotide pair", "generates", "negative axial force"),
    ("cumulative negative axial forces", "contribute_to", "rna stability"),
    ("cumulative negative axial forces", "contribute_to", "rna functionality"),
    ("negative axial force", "is_a", "axial force"),

    # P2 S3: "By understanding these steps, researchers can gain insights
    # into the mechanical behavior of RNA aptamers and optimize their
    # designs for enhanced performance."
    ("researchers", "study", "mechanical behavior"),
    ("mechanical behavior", "property_of", "rna aptamer"),

    # P2 S4: "The hypothesis emphasizes the importance of optimizing the
    # length of the 5'3' Static Stem to enhance the stability and
    # functionality of RNA aptamers."
    ("5'3' static stem length", "optimized_for", "rna stability"),
    ("5'3' static stem length", "optimized_for", "rna functionality"),
    ("5'3' static stem mechanics of materials hypothesis", "recommends", "optimize 5'3' static stem length"),

    # P2 S5: "This mechanical stability is crucial for the RNA aptamer
    # to maintain its functionality during state transitions."
    ("mechanical stability", "crucial_to", "rna aptamer functionality"),
    ("rna aptamer", "undergoes", "state transitions"),
    ("state transitions", "requires", "mechanical stability"),

    # =====================================================================
    # Paragraph 3 — 5'3' Static Stem Thermodynamics Hypothesis
    # =====================================================================

    # P3 S1: "The 5'3' Static Stem Thermodynamics Hypothesis provides a
    # comprehensive explanation of how the principles of thermodynamics
    # are inherently linked to the mechanical behavior of RNA structures."
    ("5'3' static stem thermodynamics hypothesis", "is_a", "hypothesis"),
    ("5'3' static stem thermodynamics hypothesis", "links", "thermodynamics"),
    ("5'3' static stem thermodynamics hypothesis", "links", "mechanical behavior"),
    ("thermodynamics", "linked_to", "mechanical behavior"),

    # P3 S2: "By examining the relationship between moment forces,
    # energy, and entropy, researchers can gain a deeper understanding
    # of the stability and functionality of RNA aptamers."
    ("5'3' static stem thermodynamics hypothesis", "examines", "moment forces"),
    ("5'3' static stem thermodynamics hypothesis", "examines", "energy"),
    ("5'3' static stem thermodynamics hypothesis", "examines", "entropy"),
    ("moment forces", "related_to", "energy"),
    ("moment forces", "related_to", "entropy"),

    # P3 S3: "The hypothesis posits that the moment forces from the RNA
    # bending to straighten out add energy to the bonds, which fights
    # the bonds that were formed."
    ("rna bending", "produces", "moment forces"),
    ("moment forces", "add_energy_to", "bonds"),
    ("moment forces", "fights", "formed bonds"),
    ("formed bonds", "is_a", "bonds"),

    # P3 S4: "This relationship highlights the importance of balancing
    # these forces to maintain the stability of the RNA structure."
    ("rna stability", "requires", "balanced forces"),
    ("balanced forces", "is_a", "force balance"),

    # P3 S5: "The hypothesis underscores the critical role of entropy
    # and enthalpy in RNA behavior."
    ("entropy", "role_in", "rna behavior"),
    ("enthalpy", "role_in", "rna behavior"),
    ("5'3' static stem thermodynamics hypothesis", "emphasizes", "entropy"),
    ("5'3' static stem thermodynamics hypothesis", "emphasizes", "enthalpy"),

    # =====================================================================
    # Paragraph 4 — SSNG1 / SSNG2 / SSNG3 comparative analysis
    # =====================================================================

    # P4 S1: "The comparative analysis of SSNG1, SSNG2, and SSNG3 static
    # stem plots using the Eterna Total Score reveals the optimal static
    # stem nucleotide ratios for stability and performance."
    ("ssng1", "is_a", "sublab"),
    ("ssng2", "is_a", "sublab"),
    ("ssng3", "is_a", "sublab"),
    ("ssng1", "has", "static stem plot"),
    ("ssng2", "has", "static stem plot"),
    ("ssng3", "has", "static stem plot"),
    ("eterna total score", "is_a", "metric"),
    ("ssng1", "scored_by", "eterna total score"),
    ("ssng2", "scored_by", "eterna total score"),
    ("ssng3", "scored_by", "eterna total score"),
    ("static stem nucleotide ratio", "determines", "rna stability"),
    ("static stem nucleotide ratio", "determines", "rna performance"),

    # P4 S2: "The analysis shows that higher ratios of static stem
    # nucleotides correlate with better performance, providing valuable
    # insights for RNA aptamer design."
    ("higher static stem nucleotide ratio", "correlates_with", "better performance"),
    ("higher static stem nucleotide ratio", "is_a", "static stem nucleotide ratio"),

    # P4 S3: "The study highlights the importance of understanding the
    # relationship between static stem nucleotide ratios and RNA
    # stability."
    ("static stem nucleotide ratio", "related_to", "rna stability"),

    # P4 S4: "This analysis is crucial for optimizing RNA aptamer
    # designs to achieve the desired performance and functionality."
    ("ssng comparative analysis", "optimizes", "rna aptamer design"),

    # P4 S5: "The findings underscore the significance of static stem
    # nucleotide ratios in RNA aptamer design."
    ("static stem nucleotide ratio", "significant_in", "rna aptamer design"),

    # =====================================================================
    # Paragraph 5 — Molecular Snare Mechanics Hypothesis
    # =====================================================================

    # P5 S1: "The molecular snare mechanics hypothesis offers a detailed
    # description of the dynamic process by which RNA aptamers detect
    # and bind target molecules."
    ("molecular snare mechanics hypothesis", "is_a", "hypothesis"),
    ("molecular snare mechanics hypothesis", "describes", "molecular snare"),
    ("rna aptamer", "detects", "target molecule"),
    ("rna aptamer", "binds", "target molecule"),
    ("molecular snare", "part_of", "rna aptamer"),
    ("molecular snare", "function", "detect and bind target molecule"),

    # P5 S2: "The interaction between the static stem and the binding
    # sites is crucial for achieving high fold changes and maintaining
    # stability."
    ("static stem", "interacts_with", "binding sites"),
    ("static stem", "part_of", "rna aptamer"),
    ("binding sites", "part_of", "rna aptamer"),
    ("static stem binding sites interaction", "produces", "high fold change"),
    ("static stem binding sites interaction", "maintains", "rna stability"),
    ("high fold change", "is_a", "fold change"),
    ("fold change", "is_a", "metric"),

    # P5 S3: "This hypothesis emphasizes the role of tension, axial
    # forces, and structural shifts in the binding process."
    ("molecular snare mechanics hypothesis", "emphasizes", "tension"),
    ("molecular snare mechanics hypothesis", "emphasizes", "axial forces"),
    ("molecular snare mechanics hypothesis", "emphasizes", "structural shifts"),
    ("tension", "role_in", "binding process"),
    ("axial forces", "role_in", "binding process"),
    ("structural shifts", "role_in", "binding process"),

    # P5 S4: "The molecular snare mechanics hypothesis provides a
    # comprehensive framework for understanding the structural and
    # mechanical changes that occur during the binding process."
    ("molecular snare mechanics hypothesis", "explains", "structural changes"),
    ("molecular snare mechanics hypothesis", "explains", "mechanical changes"),
    ("structural changes", "occur_during", "binding process"),
    ("mechanical changes", "occur_during", "binding process"),

    # P5 S5: "The findings highlight the importance of achieving
    # mechanical stability in RNA aptamer design."
    ("mechanical stability", "required_for", "rna aptamer design"),

    # =====================================================================
    # Paragraph 6 — Molecular Snare Thermodynamics Hypothesis
    # =====================================================================

    # P6 S1: "The molecular snare thermodynamics hypothesis provides a
    # unified theory of RNA mechanics and thermodynamics, explaining
    # the energy changes that occur during the binding process."
    ("molecular snare thermodynamics hypothesis", "is_a", "hypothesis"),
    ("molecular snare thermodynamics hypothesis", "unifies", "rna mechanics"),
    ("molecular snare thermodynamics hypothesis", "unifies", "thermodynamics"),
    ("molecular snare thermodynamics hypothesis", "explains", "energy changes"),
    ("energy changes", "occur_during", "binding process"),

    # P6 S2: "By understanding the principles of thermodynamics,
    # researchers can gain deeper insights into the stability and
    # functionality of RNA aptamers."
    ("thermodynamics", "explains", "rna aptamer stability"),
    ("thermodynamics", "explains", "rna aptamer functionality"),

    # P6 S3: "The hypothesis underscores the importance of energy
    # conservation and structural resemblance in optimizing RNA aptamer
    # performance."
    ("molecular snare thermodynamics hypothesis", "emphasizes", "energy conservation"),
    ("molecular snare thermodynamics hypothesis", "emphasizes", "structural resemblance"),
    ("energy conservation", "optimizes", "rna aptamer performance"),
    ("structural resemblance", "optimizes", "rna aptamer performance"),

    # P6 S4: "The molecular snare thermodynamics hypothesis provides a
    # comprehensive framework for understanding the dynamic process by
    # which RNA aptamers detect and bind target molecules."
    ("molecular snare thermodynamics hypothesis", "explains", "target molecule detection"),
    ("molecular snare thermodynamics hypothesis", "explains", "target molecule binding"),

    # P6 S5: "The findings highlight the importance of achieving an
    # optimal energy balance in RNA aptamer design."
    ("optimal energy balance", "required_for", "rna aptamer design"),

    # =====================================================================
    # Paragraph 7 — Fold Signal Region
    # =====================================================================

    # P7 S1: "The fold signal region is a dynamic and crucial component
    # of RNA aptamers, playing a key role in the detection and binding
    # of target molecules."
    ("fold signal region", "is_a", "component"),
    ("fold signal region", "part_of", "rna aptamer"),
    ("fold signal region", "is", "dynamic"),
    ("fold signal region", "function", "detect target molecule"),
    ("fold signal region", "function", "bind target molecule"),

    # P7 S2: "The interplay between static and dynamic elements, the
    # reinforcement provided by static stems and loops, and the
    # importance of energy conservation all contribute to the function
    # of the fold signal region."
    ("fold signal region", "contains", "static elements"),
    ("fold signal region", "contains", "dynamic elements"),
    ("static stems", "reinforce", "fold signal region"),
    ("loops", "reinforce", "fold signal region"),
    ("energy conservation", "contributes_to", "fold signal region function"),

    # P7 S3: "The study highlights the significance of understanding
    # the mechanical movement constraints and the role of static loops
    # as fulcrums in overcoming these constraints."
    ("fold signal region", "has", "mechanical movement constraints"),
    ("static loops", "act_as", "fulcrums"),
    ("fulcrums", "overcome", "mechanical movement constraints"),
    ("static loops", "is_a", "loops"),

    # P7 S4: "The findings underscore the importance of achieving a
    # balance between stability and flexibility in the fold signal
    # region."
    ("fold signal region", "requires_balance_of", "stability"),
    ("fold signal region", "requires_balance_of", "flexibility"),

    # P7 S5: "The study provides valuable insights for optimizing the
    # performance of RNA aptamers in various applications."
    ("fold signal region study", "optimizes", "rna aptamer performance"),

    # =====================================================================
    # Paragraph 8 — Holistic synthesis
    # =====================================================================

    # P8 S1: "Taken together, these insights enable a holistic
    # understanding of how RNA aptamers fold and function."
    ("the paper", "provides", "holistic understanding of rna aptamer"),
    ("rna aptamer", "has_process", "folding"),
    ("rna aptamer", "has_process", "function"),

    # P8 S2: "This comprehensive approach allows researchers to target
    # the design of aptamers at an accelerated pace, leveraging the
    # principles of mechanics and thermodynamics to achieve optimal
    # performance."
    ("the paper", "accelerates", "rna aptamer design"),
    ("rna aptamer design", "leverages", "mechanics"),
    ("rna aptamer design", "leverages", "thermodynamics"),
    ("mechanics and thermodynamics", "achieves", "optimal performance"),

    # P8 S3: "By integrating computational models, experimental data,
    # and innovative tools like the Serena RNA Analysis Tool and Eterna
    # Game Labs, the study paves the way for significant advancements
    # in RNA engineering."
    ("serena rna analysis tool", "is_a", "tool"),
    ("eterna game labs", "is_a", "tool"),
    ("the paper", "advances", "rna engineering"),

    # P8 S4: "The findings underscore the importance of energy
    # conservation, structural resemblance, and mechanical stability in
    # RNA aptamer design."
    ("energy conservation", "important_in", "rna aptamer design"),
    ("structural resemblance", "important_in", "rna aptamer design"),
    ("mechanical stability", "important_in", "rna aptamer design"),

    # P8 S5: "This integrated approach offers valuable insights for
    # future research and development in the field, ultimately leading
    # to more effective and reliable RNA-based tools."
    ("the paper", "guides", "rna research"),
    ("the paper", "guides", "rna development"),
    ("rna-based tools", "benefit_from", "integrated approach"),

    # =====================================================================
    # Thesis framing — from title and §1 "The Knob"
    # =====================================================================

    # Title: "Design rules for short-length RNA aptamer engineering
    # observed in a published Massive Open Laboratory dataset that may
    # give us the ability to fine tune RNA aptamers with the turn of a
    # knob, so as to render disease inert in the body."
    ("the paper", "identifies", "design rules"),
    ("design rules", "govern", "short-length rna aptamer engineering"),
    ("design rules", "observed_in", "massive open laboratory dataset"),
    ("massive open laboratory dataset", "is_a", "dataset"),
    ("the knob", "fine_tunes", "rna aptamer"),
    ("the knob", "is_a", "design metaphor"),
    ("rna aptamer", "can_render_inert", "disease"),
    ("fine-tuned rna aptamer", "therapeutic_target", "disease"),
]


def main() -> None:
    db_path = Path("aptamer_exec.db")
    if db_path.exists():
        raise FileExistsError(
            f"{db_path} exists — delete or rename before re-teaching"
        )

    brain = Brain(str(db_path))
    print(f"Fresh brain: {db_path}")
    print(f"{len(TRIPLES)} triples to teach (Executive Summary + thesis)\n")

    t0 = time.time()
    for i, (s, r, o) in enumerate(TRIPLES, 1):
        result = brain.teach_triple(s, r, o, source_label=SOURCE)
        ok = "OK " if result else "FAIL"
        print(f"[{i:3d}/{len(TRIPLES)}] {ok} | {s!r} --[{r}]--> {o!r}")

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
