"""Teach Sara the full aptamer paper (Pearl 2026c) via teach_triple.

Supersedes teach_exec_summary.py. Covers §1 Knob framing through §15,
with per-fact judgment applied to every sentence in order per
feedback_teach_faithfully_sentence_by_sentence. Produces a richer
substrate than the Executive Summary alone, including the §8 MOP /
150x / flexible-loop detail that was the catalyst for expanding
coverage.

Output: aptamer_full.db (fresh brain). Load into sara_test/ via
load_brain.sh aptamer_full.
"""
from __future__ import annotations

import time
from pathlib import Path

from sara_brain.core.brain import Brain


SOURCE = "aptamer_paper_rev1"


# Triples grouped by section and sentence. Novel compound terms are
# preserved verbatim as neuron labels.
TRIPLES: list[tuple[str, str, str]] = [

    # =====================================================================
    # §2 Executive Summary (carried over from teach_exec_summary.py)
    # =====================================================================

    # P1 S1
    ("the paper", "integrates", "computational models"),
    ("the paper", "integrates", "experimental data"),
    ("the paper", "aims_to_optimize", "rna aptamer design"),
    ("rna aptamer design", "emphasizes", "mechanical forces"),
    ("mechanical forces", "role_in", "rna stability"),
    ("mechanical forces", "role_in", "rna functionality"),

    # P1 S2
    ("the study", "uses", "crowdsourcing"),
    ("crowdsourcing", "done_through", "eterna game labs"),
    ("eterna game labs", "solves", "rna folding problems"),
    ("the study", "introduces", "marker theory"),
    ("the study", "introduces", "switch acceptance theory"),
    ("the study", "introduces", "theory of mechanics of rna folding model"),
    ("marker theory", "is_a", "theory"),
    ("switch acceptance theory", "is_a", "theory"),
    ("theory of mechanics of rna folding model", "is_a", "theory"),

    # P1 S3
    ("marker theory", "predicts", "rna performance"),
    ("switch acceptance theory", "predicts", "rna performance"),
    ("theory of mechanics of rna folding model", "predicts", "rna performance"),
    ("marker theory", "optimizes", "rna aptamer design"),
    ("switch acceptance theory", "optimizes", "rna aptamer design"),
    ("theory of mechanics of rna folding model", "optimizes", "rna aptamer design"),

    # P1 S4
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

    # P1 S5
    ("the paper", "describes", "molecular snare mechanism"),
    ("the paper", "describes", "axial forces"),
    ("the paper", "describes", "fold signal region"),
    ("molecular snare mechanism", "involves", "molecular snare"),
    ("axial forces", "important_to", "rna aptamer design"),

    # P2 5'3' Static Stem Mechanics of Materials Hypothesis
    ("5'3' static stem mechanics of materials hypothesis", "is_a", "hypothesis"),
    ("5'3' static stem mechanics of materials hypothesis", "explains", "mechanical forces"),
    ("mechanical forces", "act_within", "5'3' static stem"),
    ("5'3' static stem", "part_of", "rna aptamer"),
    ("5'3' static stem mechanics of materials hypothesis", "describes_process", "cumulative negative axial forces"),
    ("cumulative negative axial forces", "generated_by", "nucleotide pair"),
    ("nucleotide pair", "generates", "negative axial force"),
    ("cumulative negative axial forces", "contribute_to", "rna stability"),
    ("cumulative negative axial forces", "contribute_to", "rna functionality"),
    ("negative axial force", "is_a", "axial force"),
    ("researchers", "study", "mechanical behavior"),
    ("mechanical behavior", "property_of", "rna aptamer"),
    ("5'3' static stem length", "optimized_for", "rna stability"),
    ("5'3' static stem length", "optimized_for", "rna functionality"),
    ("5'3' static stem mechanics of materials hypothesis", "recommends", "optimize 5'3' static stem length"),
    ("mechanical stability", "crucial_to", "rna aptamer functionality"),
    ("rna aptamer", "undergoes", "state transitions"),
    ("state transitions", "requires", "mechanical stability"),

    # P3 5'3' Static Stem Thermodynamics Hypothesis
    ("5'3' static stem thermodynamics hypothesis", "is_a", "hypothesis"),
    ("5'3' static stem thermodynamics hypothesis", "links", "thermodynamics"),
    ("5'3' static stem thermodynamics hypothesis", "links", "mechanical behavior"),
    ("thermodynamics", "linked_to", "mechanical behavior"),
    ("5'3' static stem thermodynamics hypothesis", "examines", "moment forces"),
    ("5'3' static stem thermodynamics hypothesis", "examines", "energy"),
    ("5'3' static stem thermodynamics hypothesis", "examines", "entropy"),
    ("moment forces", "related_to", "energy"),
    ("moment forces", "related_to", "entropy"),
    ("rna bending", "produces", "moment forces"),
    ("moment forces", "add_energy_to", "bonds"),
    ("moment forces", "fights", "formed bonds"),
    ("formed bonds", "is_a", "bonds"),
    ("rna stability", "requires", "balanced forces"),
    ("balanced forces", "is_a", "force balance"),
    ("entropy", "role_in", "rna behavior"),
    ("enthalpy", "role_in", "rna behavior"),
    ("5'3' static stem thermodynamics hypothesis", "emphasizes", "entropy"),
    ("5'3' static stem thermodynamics hypothesis", "emphasizes", "enthalpy"),

    # P4 SSNG comparative analysis
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
    ("higher static stem nucleotide ratio", "correlates_with", "better performance"),
    ("higher static stem nucleotide ratio", "is_a", "static stem nucleotide ratio"),
    ("static stem nucleotide ratio", "related_to", "rna stability"),
    ("ssng comparative analysis", "optimizes", "rna aptamer design"),
    ("static stem nucleotide ratio", "significant_in", "rna aptamer design"),

    # P5 Molecular Snare Mechanics Hypothesis
    ("molecular snare mechanics hypothesis", "is_a", "hypothesis"),
    ("molecular snare mechanics hypothesis", "describes", "molecular snare"),
    ("rna aptamer", "detects", "target molecule"),
    ("rna aptamer", "binds", "target molecule"),
    ("molecular snare", "part_of", "rna aptamer"),
    ("molecular snare", "function", "detect and bind target molecule"),
    ("static stem", "interacts_with", "binding sites"),
    ("static stem", "part_of", "rna aptamer"),
    ("binding sites", "part_of", "rna aptamer"),
    ("static stem binding sites interaction", "produces", "high fold change"),
    ("static stem binding sites interaction", "maintains", "rna stability"),
    ("high fold change", "is_a", "fold change"),
    ("fold change", "is_a", "metric"),
    ("molecular snare mechanics hypothesis", "emphasizes", "tension"),
    ("molecular snare mechanics hypothesis", "emphasizes", "axial forces"),
    ("molecular snare mechanics hypothesis", "emphasizes", "structural shifts"),
    ("tension", "role_in", "binding process"),
    ("axial forces", "role_in", "binding process"),
    ("structural shifts", "role_in", "binding process"),
    ("molecular snare mechanics hypothesis", "explains", "structural changes"),
    ("molecular snare mechanics hypothesis", "explains", "mechanical changes"),
    ("structural changes", "occur_during", "binding process"),
    ("mechanical changes", "occur_during", "binding process"),
    ("mechanical stability", "required_for", "rna aptamer design"),

    # P6 Molecular Snare Thermodynamics Hypothesis
    ("molecular snare thermodynamics hypothesis", "is_a", "hypothesis"),
    ("molecular snare thermodynamics hypothesis", "unifies", "rna mechanics"),
    ("molecular snare thermodynamics hypothesis", "unifies", "thermodynamics"),
    ("molecular snare thermodynamics hypothesis", "explains", "energy changes"),
    ("energy changes", "occur_during", "binding process"),
    ("thermodynamics", "explains", "rna aptamer stability"),
    ("thermodynamics", "explains", "rna aptamer functionality"),
    ("molecular snare thermodynamics hypothesis", "emphasizes", "energy conservation"),
    ("molecular snare thermodynamics hypothesis", "emphasizes", "structural resemblance"),
    ("energy conservation", "optimizes", "rna aptamer performance"),
    ("structural resemblance", "optimizes", "rna aptamer performance"),
    ("molecular snare thermodynamics hypothesis", "explains", "target molecule detection"),
    ("molecular snare thermodynamics hypothesis", "explains", "target molecule binding"),
    ("optimal energy balance", "required_for", "rna aptamer design"),

    # P7 Fold Signal Region
    ("fold signal region", "is_a", "component"),
    ("fold signal region", "part_of", "rna aptamer"),
    ("fold signal region", "is", "dynamic"),
    ("fold signal region", "function", "detect target molecule"),
    ("fold signal region", "function", "bind target molecule"),
    ("fold signal region", "contains", "static elements"),
    ("fold signal region", "contains", "dynamic elements"),
    ("static stems", "reinforce", "fold signal region"),
    ("loops", "reinforce", "fold signal region"),
    ("energy conservation", "contributes_to", "fold signal region function"),
    ("fold signal region", "has", "mechanical movement constraints"),
    ("static loops", "act_as", "fulcrums"),
    ("fulcrums", "overcome", "mechanical movement constraints"),
    ("static loops", "is_a", "loops"),
    ("fold signal region", "requires_balance_of", "stability"),
    ("fold signal region", "requires_balance_of", "flexibility"),
    ("fold signal region study", "optimizes", "rna aptamer performance"),

    # P8 Holistic synthesis
    ("the paper", "provides", "holistic understanding of rna aptamer"),
    ("rna aptamer", "has_process", "folding"),
    ("rna aptamer", "has_process", "function"),
    ("the paper", "accelerates", "rna aptamer design"),
    ("rna aptamer design", "leverages", "mechanics"),
    ("rna aptamer design", "leverages", "thermodynamics"),
    ("mechanics and thermodynamics", "achieves", "optimal performance"),
    ("serena rna analysis tool", "is_a", "tool"),
    ("eterna game labs", "is_a", "tool"),
    ("the paper", "advances", "rna engineering"),
    ("energy conservation", "important_in", "rna aptamer design"),
    ("structural resemblance", "important_in", "rna aptamer design"),
    ("mechanical stability", "important_in", "rna aptamer design"),
    ("the paper", "guides", "rna research"),
    ("the paper", "guides", "rna development"),
    ("rna-based tools", "benefit_from", "integrated approach"),

    # Title thesis
    ("the paper", "identifies", "design rules"),
    ("design rules", "govern", "short-length rna aptamer engineering"),
    ("design rules", "observed_in", "massive open laboratory dataset"),
    ("massive open laboratory dataset", "is_a", "dataset"),
    ("the knob", "fine_tunes", "rna aptamer"),
    ("the knob", "is_a", "design metaphor"),
    ("rna aptamer", "can_render_inert", "disease"),
    ("fine-tuned rna aptamer", "therapeutic_target", "disease"),

    # =====================================================================
    # §3 Introduction
    # =====================================================================

    # S1: SELEX problems
    ("selex", "is_a", "method"),
    ("selex", "full_name", "systematic evolution of ligands by exponential enrichment"),
    ("selex", "is_current_method_for", "rna aptamer design"),
    ("selex", "is", "labor-intensive"),
    ("selex", "is", "time-consuming"),
    ("selex", "relies_on", "repetitive mutations"),
    ("selex", "relies_on", "experimental chance"),
    ("selex", "analogous_to", "using a hammer when a scalpel would be appropriate"),
    ("selex", "lacks", "precision"),
    ("selex", "lacks", "understanding of nature's design rules"),
    ("selex", "produces", "suboptimal designs"),
    ("selex", "is_a_method_of", "trial-and-error"),
    ("selex", "fails_to", "direct performance to meet medical needs"),

    # S2: proposed solution
    ("the paper", "proposes", "systematic approach"),
    ("systematic approach", "leverages", "computational models"),
    ("systematic approach", "leverages", "experimental data"),
    ("systematic approach", "leverages", "rna mechanics"),
    ("systematic approach", "leverages", "thermodynamics"),
    ("systematic approach", "moves_beyond", "selex"),
    ("computational models", "simulate", "aptamer target molecule interactions"),
    ("computational models", "provide", "insights into design parameters"),
    ("experimental techniques", "validate", "computational models"),
    ("the knob", "analogous_to", "fine-tuning aptamer performance"),
    ("the knob", "more_controlled_than", "selex"),
    ("the knob", "more_predictable_than", "selex"),

    # S3: advanced metrics
    ("the paper", "advocates", "advanced metrics"),
    ("the paper", "advocates", "structural analysis"),
    ("advanced metrics", "identify", "optimal ratios and configurations"),
    ("rna aptamer", "can_be", "biosensor"),
    ("biosensor", "detects", "environmental toxins"),
    ("rna aptamer", "can_be", "therapeutic agent"),
    ("therapeutic agent", "targets", "disease-related molecules"),
    ("knob-turning approach", "simplifies", "rna aptamer design process"),

    # =====================================================================
    # §4 Test Methodologies and Tools
    # =====================================================================

    # Introduction paragraph
    ("the paper", "introduces", "suite of methodologies"),
    ("suite of methodologies", "optimizes", "rna aptamer design"),
    ("suite of methodologies", "addresses", "rna folding complexity"),
    ("suite of methodologies", "addresses", "rna binding complexity"),
    ("suite of methodologies", "addresses", "rna switching complexity"),

    # Theory of Mechanics paragraph
    ("theory of mechanics of rna", "applies", "classical mechanics"),
    ("theory of mechanics of rna", "treats", "pairing probabilities as vectors"),
    ("pairing probabilities", "analogous_to", "forces"),
    ("theory of mechanics of rna", "predicts", "rna behavior"),
    ("theory of mechanics of rna", "predicts", "state transitions"),
    ("state transitions", "between", "unbound and bound states"),
    ("theory of mechanics of rna", "emphasizes", "axial forces"),
    ("theory of mechanics of rna", "emphasizes", "shearing forces"),
    ("theory of mechanics of rna", "emphasizes", "tension"),
    ("theory of mechanics of rna", "emphasizes", "bending moments"),

    # Serena paragraph (more detail than §2 P1 S4)
    ("serena rna analysis tool", "incorporates", "marker theory"),
    ("serena rna analysis tool", "incorporates", "switch acceptance theory"),
    ("serena rna analysis tool", "assesses", "rna aptamer stability"),
    ("serena rna analysis tool", "assesses", "rna aptamer switching capabilities"),
    ("ev", "abbreviation_of", "ensemble variation"),
    ("lmv", "abbreviation_of", "local minima variation"),
    ("ws", "abbreviation_of", "weighted structures"),

    # Eterna paragraph
    ("eterna game labs", "uses", "interactive puzzles"),
    ("eterna game labs", "uses", "interactive challenges"),
    ("eterna game labs", "engages", "global community of players"),
    ("eterna platform", "uses", "game mechanics"),
    ("eterna platform", "simulates", "rna interactions"),
    ("eterna platform", "predicts", "optimal design parameters"),
    ("eterna players", "contribute_to", "discovery of new rna structures"),
    ("eterna game labs", "exemplifies", "crowdsourcing in scientific research"),

    # Marker Theory paragraph
    ("marker theory", "is_foundational", "true"),
    ("marker theory", "identifies", "structural markers in rna secondary structures"),
    ("structural markers", "indicate", "performance ranges"),
    ("structural markers", "predict", "folding actions"),
    ("structural markers", "predict", "switching actions"),
    ("marker theory", "evaluates", "rna aptamer designs"),
    ("marker theory", "ensures", "smooth state transitions"),
    ("marker theory", "ensures", "minimal energy expenditure"),

    # Switch Acceptance Theory paragraph
    ("switch acceptance theory", "builds_on", "marker theory"),
    ("switch acceptance theory", "assumes", "structural shape is entire ensemble potentiality"),
    ("mfe", "abbreviation_of", "minimum free energy"),
    ("switch acceptance theory", "not_limited_to", "mfe structure"),
    ("switch acceptance theory", "analyzes", "alternate structures in ensemble"),
    ("switch acceptance theory", "analyzes", "pairing probabilities"),
    ("switch acceptance theory", "searches_for", "echoes of target alternate states"),
    ("switch acceptance theory", "predicts", "ease of folding into alternate state"),
    ("switch acceptance theory", "essential_for", "rna aptamer design optimization"),

    # =====================================================================
    # §5 Theory of Mechanics of RNA Folding Model
    # =====================================================================

    # §5 overview
    ("theory of mechanics of rna folding model", "posits", "rna folding follows classical mechanics"),
    ("theory of mechanics of rna folding model", "treats", "pairing probabilities as vectors"),
    ("theory of mechanics of rna folding model", "applies", "laws of motion"),
    ("theory of mechanics of rna folding model", "considers", "axial forces"),
    ("theory of mechanics of rna folding model", "considers", "shearing forces"),
    ("theory of mechanics of rna folding model", "considers", "tension forces"),
    ("theory of mechanics of rna folding model", "considers", "bending moments"),

    # §5.1 Axial Forces
    ("axial forces", "act_along", "axis of rna strand"),
    ("axial forces", "can_be", "positive axial force"),
    ("axial forces", "can_be", "negative axial force"),
    ("positive axial force", "causes", "stretching"),
    ("negative axial force", "causes", "compression"),
    ("hydrogen bonds", "generate", "internal negative axial forces"),
    ("internal negative axial forces", "pull_together", "nucleotides"),
    ("internal negative axial forces", "maintain", "rna stability"),
    ("external positive axial force", "pulls_apart", "nucleotides"),
    ("external positive axial force", "causes", "rna unfolding"),
    ("balance of axial forces", "crucial_for", "aptamer stability"),
    ("balance of axial forces", "crucial_for", "aptamer functionality"),
    ("cumulative negative axial forces", "join", "5' and 3' ends"),
    ("5'3' static stem", "creates", "flexible loop"),
    ("flexible loop", "withstands", "high stresses"),
    ("tension forces", "transmitted_via", "rna phosphate backbone"),
    ("tension forces", "contribute_to", "rna stability"),
    ("tension forces", "contribute_to", "rna functionality"),

    # §5.2 Shearing Forces
    ("shearing forces", "occur_when", "relative displacement between nucleotide pairs"),
    ("shearing forces", "cause", "shifts in base pair alignment"),
    ("shearing forces", "can_break", "bonds at hairpin ends"),
    ("shearing forces", "can_form", "new configurations"),
    ("au hairpin", "is_a", "hairpin"),
    ("shearing forces", "can_break", "au hairpin end bonds"),
    ("shearing forces", "act_parallel_to", "axis of rna strand"),
    ("shearing forces", "can_be", "positive shearing force"),
    ("shearing forces", "can_be", "negative shearing force"),
    ("positive shearing force", "moves_upward", "nucleotides"),
    ("negative shearing force", "moves_downward", "nucleotides"),
    ("shearing forces", "affect", "rna stability"),
    ("shearing forces", "can_cause", "rna unfolding"),

    # §5.3 Tension Forces
    ("tension forces", "transmitted_along", "length of rna strand"),
    ("tension forces", "analogous_to", "forces in rope or chain"),
    ("tension forces", "contribute_to", "mechanical stability"),
    ("tension forces", "play_role_in", "maintaining stability"),
    ("tension forces", "important_in", "experimental labs"),

    # §5.4 Bending Moments
    ("bending moments", "arise_when", "forces cause rna to bend or curve"),
    ("bending moments", "relevant_in", "loop formation"),
    ("bending moments", "relevant_in", "hairpin formation"),
    ("bending moments", "balance", "straightening forces vs bending forces"),
    ("bending moments", "can_be", "positive bending moment"),
    ("bending moments", "can_be", "negative bending moment"),
    ("positive bending moment", "bends_up", "rna ends"),
    ("negative bending moment", "bends_down", "rna ends"),
    ("bending moments", "crucial_for_understanding", "rna flexibility"),

    # §5.5 Newton's First Law (Inertia) applied to RNA
    ("newton's first law", "also_known_as", "law of inertia"),
    ("newton's first law", "states", "object stays at rest or in uniform motion unless acted upon"),
    ("newton's first law", "applies_to", "rna molecules"),
    ("inertia in rna", "is", "tendency to maintain current state"),
    ("rna molecule", "will_not_change_state", "without external force"),
    ("equilibrium state of rna", "is_a", "most stable configuration"),
    ("equilibrium state of rna", "called", "mfe state"),
    ("mfe state", "is_a", "equilibrium state"),
    ("rna molecule", "stays_in", "equilibrium state"),
    ("external force", "disrupts", "equilibrium state"),
    ("external force", "can_be", "temperature change"),
    ("external force", "can_be", "binding molecule"),
    ("external force", "can_be", "presence of ions"),
    ("temperature increase", "adds", "thermal energy"),
    ("thermal energy", "breaks", "hydrogen bonds"),
    ("thermal energy", "causes", "rna unfolding"),
    ("temperature decrease", "causes", "rna to lose kinetic energy"),
    ("rna", "transitions_to", "more stable configuration when cooled"),
    ("more stable configurations", "have", "higher inertia"),
    ("more stable configurations", "require", "larger external forces to change"),
    ("target molecule binding", "acts_as", "external force on aptamer"),
    ("rna aptamer", "can_return_to", "stable state after ligand removal"),
    ("newton's first law", "provides_framework_for", "rna stability understanding"),

    # §5.6 Newton's Second Law applied to RNA
    ("newton's second law", "states", "force equals mass times acceleration"),
    ("newton's second law", "formula", "f equals ma"),
    ("newton's second law", "applies_to", "rna molecules"),
    ("force in rna", "means", "pairing probabilities between nucleotides"),
    ("mass in rna", "means", "mass of rna nucleotides"),
    ("acceleration in rna", "means", "change in velocity of nucleotides"),
    ("pairing probabilities", "are_analogous_to", "forces"),
    ("high pairing probability", "indicates", "strong attractive force"),
    ("high pairing probability", "causes", "nucleotides to move closer"),
    ("low pairing probability", "indicates", "weaker force"),
    ("low pairing probability", "results_in", "less movement"),
    ("velocity in rna", "is", "rate of nucleotide position change"),
    ("velocity in rna", "is_a", "vector quantity"),
    ("velocity", "has", "magnitude and direction"),
    ("acceleration in rna", "is", "rate of velocity change over time"),
    ("applying newton's second law", "step", "identify forces"),
    ("applying newton's second law", "step", "calculate mass"),
    ("applying newton's second law", "step", "determine acceleration"),
    ("heavier nucleotides", "have", "more inertia"),
    ("heavier nucleotides", "require", "more force for same acceleration"),
    ("a u pair", "has", "high pairing probability"),
    ("external force like temperature change", "causes", "unfolding"),
    ("newton's second law", "allows_prediction_of", "rna folding"),
    ("newton's second law", "allows_design_of", "rna aptamers"),
    ("newton's second law", "useful_for", "rna-based sensors"),

    # §5.7 Newton's Third Law applied to RNA
    ("newton's third law", "states", "every action has equal and opposite reaction"),
    ("newton's third law", "applies_to", "rna molecules"),
    ("newton's third law", "observed_in", "nucleotide interactions"),
    ("hydrogen bond formation", "involves", "forces between nucleotides"),
    ("hydrogen bond formation", "exemplifies", "newton's third law"),
    ("forces in hydrogen bonds", "are", "balanced"),
    ("hydrogen bond breaking", "releases", "forces"),
    ("hydrogen bond breaking", "causes", "structural changes"),
    ("rna folding", "involves", "hydrogen bond formation"),
    ("rna unfolding", "involves", "hydrogen bond breaking"),
    ("ligand binding", "exerts_force_on", "rna aptamer"),
    ("rna aptamer", "exerts_reaction_force_on", "ligand"),
    ("ligand binding", "causes", "aptamer structural change"),
    ("releasing ligand", "requires", "overcoming binding forces"),
    ("newton's third law", "useful_for_predicting", "rna behavior"),
    ("newton's third law", "useful_for_designing", "rna aptamers"),

    # §5.8 Application
    ("theory of mechanics of rna folding model", "applies_to", "rna aptamer design"),
    ("theory of mechanics of rna folding model", "applies_to", "rna aptamer optimization"),
    ("theory of mechanics of rna folding model", "provides_framework_for", "predicting rna behavior"),
    ("theory of mechanics of rna folding model", "advances", "rna engineering"),

    # =====================================================================
    # §6 Description of the Serena RNA Analysis Tool
    # =====================================================================

    # Overview
    ("serena rna analysis tool", "is_a", "research instrument"),
    ("serena rna analysis tool", "integrates", "computational models"),
    ("serena rna analysis tool", "integrates", "experimental data"),
    ("serena rna analysis tool", "evaluates", "rna aptamer performance"),
    ("serena rna analysis tool", "focuses_on", "rna stability"),
    ("serena rna analysis tool", "focuses_on", "binding affinity"),
    ("serena rna analysis tool", "focuses_on", "switching capabilities"),

    # Ensemble Variation (EV)
    ("ensemble variation", "measures", "instantaneous variation within rna ensemble"),
    ("ensemble variation", "inspired_by", "nupack ensemble defect"),
    ("ensemble variation", "addresses_limitations_of", "nupack ensemble defect"),
    ("ensemble variation", "provides_representation_of", "variation at 1 to 2 kcal from mfe"),
    ("ensemble variation", "considers", "number of nucleotides"),
    ("ensemble variation", "considers", "number of structures in ensemble"),
    ("lower ev value", "indicates", "more stable rna fold"),
    ("higher ev value", "indicates", "greater instability"),

    # Local Minima Variation (LMV)
    ("local minima variation", "measures", "variation within subset of rna ensemble"),
    ("local minima variation", "measured_relative_to", "reference secondary structure"),
    ("local minima variation", "identifies", "local minima within ensemble"),
    ("local minima variation", "expressed_in_units_of", "ensemble variation"),
    ("lmv_c", "is_a", "local minima variation flavor"),
    ("lmv_c", "measures_against", "weighted structures"),
    ("lmv_r", "is_a", "local minima variation flavor"),
    ("lmv_r", "measures_against", "first structure in subset"),
    ("lmv_m", "is_a", "local minima variation flavor"),
    ("lmv_m", "measures_against", "unbound mfe structure"),
    ("high lmv value", "indicates", "greater variation from reference structure"),
    ("low lmv value", "indicates", "less variation"),

    # Weighted Structures (WS)
    ("weighted structures", "represent", "average and most common nucleotide pairing"),
    ("weighted structures", "notation", "dot-bracket"),
    ("weighted structures", "examines", "each nucleotide position for alternative secondary structure"),
    ("weighted structures", "finds", "most common binding"),
    ("weighted structures", "replaces", "mfe as ensemble representation"),
    ("weighted structure not converting cleanly", "indicates", "rna intrinsic instability"),

    # Comparison Structures (CS)
    ("comparison structures", "is_a", "novel metric"),
    ("comparison structures", "provides", "single structural representation of two ensembles"),
    ("comparison structures", "represents", "whether weighted structure resembles second state or first state mfe"),
    ("comparison structures", "useful_for", "switching capabilities evaluation"),
    ("comparison structures", "has_sub_metrics", "buratio"),
    ("comparison structures", "has_sub_metrics", "both_raise"),
    ("comparison structures", "has_sub_metrics", "braise"),
    ("comparison structures", "has_sub_metrics", "udrop"),
    ("comparison structures", "has_sub_metrics", "bothtototal"),
    ("comparison structures", "has_sub_metrics", "boundtototal"),
    ("comparison structures", "has_sub_metrics", "unboundtototal"),
    ("comparison structures", "has_sub_metrics", "bound_both"),

    # CS sub-metrics
    ("buratio", "abbreviation_of", "bound to unbound ratio"),
    ("buratio", "measures", "ratio of bound to unbound nucleotides"),
    ("buratio", "provides_insights_into", "binding affinity"),
    ("both_raise", "measures", "increase in both-bound-and-unbound nucleotides between groups"),
    ("braise", "abbreviation_of", "bound raise"),
    ("braise", "measures", "increase in bound nucleotides between groups"),
    ("braise", "provides_insights_into", "binding efficiency"),
    ("udrop", "abbreviation_of", "unbound drop"),
    ("udrop", "measures", "decrease in unbound nucleotides between groups"),
    ("udrop", "provides_insights_into", "transition dynamics"),
    ("bothtototal", "abbreviation_of", "both to total ratio"),
    ("bothtototal", "measures", "ratio of both-bound-and-unbound to total nucleotides"),
    ("boundtototal", "abbreviation_of", "bound to total ratio"),
    ("boundtototal", "measures", "ratio of bound nucleotides to total"),
    ("boundtototal", "provides_insights_into", "binding efficiency"),
    ("unboundtototal", "abbreviation_of", "unbound to total ratio"),
    ("unboundtototal", "measures", "ratio of unbound nucleotides to total"),
    ("bound_both", "abbreviation_of", "bound to both ratio"),
    ("bound_both", "measures", "ratio of bound nucleotides to both-bound-and-unbound"),
    ("bound_both", "provides_insights_into", "switching capabilities"),

    # §6.1 Serena's plots
    ("serena plots", "generated_from", "first 7 kcal from mfe"),
    ("serena plots", "created_at_increments_of", "1 kcal"),
    ("first 1 kcal plots", "used_for", "plot interpretation"),
    ("sum of 7 kcal alternate structures", "used_for", "structure count analysis"),
    ("serena plots", "visualize", "ev lmv ws cs metrics"),
    ("serena plots", "identify", "trends and patterns in rna behavior"),

    # §6.2 Eterna wetlab metrics
    ("eterna switch subscore", "quantifies", "fold-increase in kd,obs between off and on states"),
    ("eterna switch subscore", "measures", "range of switching at optimal ms2 concentration"),
    ("higher switch subscore", "indicates", "more effective switch"),
    ("eterna baseline subscore", "quantifies", "closeness of on state affinity to native ms2 hairpin"),
    ("higher baseline subscore", "indicates", "aptamer mimics native ms2 hairpin"),
    ("eterna folding subscore", "rewards", "proper folding of ms2 hairpin"),
    ("eterna folding subscore", "decreases_if", "max signal below threshold"),
    ("higher folding subscore", "signifies", "reliable secondary structure formation"),
    ("eterna score", "is_composite_of", "switch subscore"),
    ("eterna score", "is_composite_of", "baseline subscore"),
    ("eterna score", "is_composite_of", "folding subscore"),
    ("eterna score", "captures", "three qualities of good switch"),
    ("higher eterna score", "indicates", "well-designed rna aptamer"),
    ("kdoff", "measures", "aptamer affinity to off state"),
    ("higher kdoff", "indicates", "greater fold change"),
    ("kdon", "measures", "aptamer affinity to on state"),
    ("lower kdon", "indicates", "greater fold change"),
    ("fold change", "calculated_as", "ratio of kd,obs in off state to kd,obs in on state"),
    ("higher fold change", "indicates", "more effective switch"),

    # =====================================================================
    # §7 Introduction to Key Observations
    # =====================================================================

    ("observation a", "is_a", "observation"),
    ("observation b", "is_a", "observation"),
    ("observation c", "is_a", "observation"),
    ("the paper", "introduces", "observation a"),
    ("the paper", "introduces", "observation b"),
    ("the paper", "introduces", "observation c"),
    ("observation c", "topic_of", "follow-up paper"),
    ("observation c", "tracks_with", "lmv_rel metric"),
    ("observation c", "uses_unit", "ensemble variation"),

    # §7.1 Observation A
    ("observation a", "describes", "rna aptamers as complex systems of interlocking parts"),
    ("rna aptamer", "is_a", "complex system of interlocking parts"),
    ("rna aptamer", "contains", "5'3' static stem"),
    ("rna aptamer", "contains", "molecular snare"),
    ("rna aptamer", "contains", "signal fold region"),
    ("5'3' static stem", "function", "structural stability"),
    ("5'3' static stem", "prevents", "failure under high fold change forces"),
    ("molecular snare", "captures", "target molecules"),
    ("molecular snare", "triggers", "signal fold region"),
    ("signal fold region", "facilitates", "conformational changes for binding"),
    ("5'3' static stem", "is_a", "stabilizing backbone"),
    ("signal fold region", "expresses", "stacks"),
    ("signal fold region", "expresses", "hairpins"),
    ("signal fold region", "expresses", "loops"),
    ("rna aptamer", "has", "isolated motifs"),
    ("isolated motifs", "is_a", "local minima"),
    ("isolated motifs", "can_have", "optimal secondary structure"),
    ("isolated motifs", "can_have", "suboptimal secondary structure"),
    ("secondary structure configurations", "influence", "fold change"),
    ("secondary structure configurations", "influence", "kdoff"),
    ("secondary structure configurations", "influence", "kdon"),

    # §7.2 Observation B
    ("observation b", "describes", "role of structural components in rna aptamer functionality"),
    ("observation b", "focuses_on", "5'3' static stem"),
    ("observation b", "focuses_on", "molecular snare"),
    ("observation b", "focuses_on", "signal fold region"),
    ("5'3' static stem", "contains", "first set of consecutive nucleotide pairs"),
    ("5'3' static stem", "encompasses", "5' end of rna"),
    ("5'3' static stem", "encompasses", "3' end of rna"),
    ("5'3' static stem", "allows", "flexibility in other regions"),
    ("5'3' static stem", "provides_support_for", "binding process"),
    ("5'3' static stem", "facilitates", "conformational changes for binding"),
    ("molecular snare", "contains", "specific binding sites"),
    ("molecular snare", "captures_with", "high specificity"),
    ("molecular snare", "captures_with", "high affinity"),
    ("molecular snare", "triggers", "conformational change"),
    ("molecular snare", "activates", "signal fold region"),
    ("molecular snare", "contributes_to", "rna aptamer stability"),
    ("signal fold region", "undergoes", "significant conformational changes"),
    ("signal fold region", "trigger", "target binding"),
    ("signal fold region", "is_dynamic", "true"),
    ("signal fold region", "contributes_to", "rna aptamer stability"),

    # =====================================================================
    # §8 Detailed 5'3' Static Stem Observation — MOP, 150x, flexible loop
    # =====================================================================

    # Introduction
    ("5'3' static stem observation", "emphasizes", "structural stability during state transitions"),
    ("5'3' static stem", "is_a", "first set of consecutive nucleotide pairs"),
    ("5'3' static stem", "prevents", "aptamer collapse under high fold change forces"),

    # MOP concept (KEY)
    ("mechanical overload point", "abbreviation", "mop"),
    ("mop", "abbreviation_of", "mechanical overload point"),
    ("mop", "represents", "maximum internal stress limit rna can withstand"),
    ("mop", "exceeding_causes", "complete structural stability loss"),
    ("mop", "exceeding_causes", "rna being ripped apart during state transition"),
    ("mop", "analogous_to", "melt point of rna"),
    ("melt point of rna", "is_where", "structure denatures from heat"),
    ("5'3' static stem", "increases", "upper limit for mop"),
    ("higher mop", "allows", "higher fold changes"),
    ("higher mop", "allows", "greater amplitudes of internal forces"),

    # Length relationship to MOP
    ("5'3' static stem length", "directly_related_to", "mop"),
    ("longer 5'3' static stem", "provides", "higher mop"),
    ("longer 5'3' static stem", "provides", "greater structural stability"),
    ("shorter 5'3' static stem", "provides", "lower mop"),
    ("shorter 5'3' static stem", "more_susceptible_to", "mechanical failure during state transitions"),

    # Snare + stem synergy
    ("incorporating 5'3' static stem into molecular snare", "further_increases", "mop"),
    ("molecular snare with 5'3' static stem", "enables", "fold changes greater than 100"),
    ("bound molecule", "provides", "extra support to rna aptamer"),
    ("bound molecule", "helps", "rna withstand higher internal forces during state transitions"),

    # §8.1 Mechanics of Materials Hypothesis — the 5-step process
    ("5'3' static stem mechanics of materials hypothesis", "describes_step_by_step", "5 step process"),

    # Step 1
    ("mechanics of materials step 1", "name", "cumulative negative axial force"),
    ("nucleotide pair in 5'3' static stem", "provides", "cumulative negative axial force"),
    ("cumulative negative axial force", "joins", "5' and 3' ends"),
    ("cumulative negative axial force", "compresses", "rna structure"),
    ("cumulative negative axial force", "creates", "stable stack of nucleotide pairs"),
    ("cumulative negative axial force", "maintains", "rna structural integrity during state transitions"),

    # Step 2
    ("mechanics of materials step 2", "name", "formation of flexible loop"),
    ("sufficient axial forces joining 5' and 3' ends", "creates", "flexible loop"),
    ("flexible loop", "joined_by", "phosphate backbone"),
    ("flexible loop", "analogous_to", "cable seal"),
    ("flexible loop", "has", "tension parallel to loop length"),
    ("loop tension", "distributes", "forces evenly across rna structure"),
    ("loop tension", "prevents", "localized stress points"),
    ("localized stress points", "could_lead_to", "structural failure"),

    # Step 3
    ("mechanics of materials step 3", "name", "positive axial forces from bending moments"),
    ("flexible loop tension", "causes", "positive axial forces at 5'3' static stem bonds"),
    ("bending moments", "form_when", "rna tries to straighten out"),
    ("bending moments", "add_energy_to", "bonds"),
    ("bending moments", "fight", "formed bonds"),
    ("greater moment forces", "produce", "higher free energy"),
    ("greater moment forces", "bring_system_closer_to", "entropy"),
    ("balance of negative and positive axial forces", "crucial_for", "rna stability"),

    # Step 4 and 5
    ("mechanics of materials step 4", "name", "increasing negative axial forces with pair count"),
    ("increasing pair count in 5'3' static stem", "increases", "negative axial forces"),
    ("negative axial forces with sufficient pair count", "eventually_exceed", "positive axial forces from tension and bending"),
    ("5'3' static stem of sufficient length", "enables", "flexible loop to withstand high stresses"),
    ("flexible loop with sufficient length", "withstands_fold_changes_up_to", "150"),
    ("fold changes reaching 150", "achievable_by", "sufficient 5'3' static stem length"),
    ("mechanics of materials step 5", "name", "achieving mechanical stability"),
    ("mechanical stability", "crucial_for", "aptamer functionality during state transitions"),

    # §8.2 Thermodynamics Hypothesis (detailed)
    ("5'3' static stem thermodynamics hypothesis", "demonstrates", "theory of mechanics necessitates thermodynamics"),
    ("moment forces from rna straightening", "analogous_to", "external forces disrupting hydrogen bonds"),
    ("less moment forces bending", "produces", "lower free energy"),
    ("mfe structure", "is", "middle ground of moment forces"),
    ("mfe structure", "has", "neither too high nor too low moment forces"),
    ("entropy role", "appears_when", "rna straightens in absence of heat and bonds"),
    ("increasing free energy", "moves_system_toward", "higher entropy"),
    ("applying heat", "forms", "hydrogen bonds"),
    ("forming hydrogen bonds", "increases", "enthalpy"),
    ("higher enthalpy", "produces", "more stable structure"),
    ("interplay between entropy and enthalpy", "crucial_for", "rna thermodynamic behavior"),

    # §8.4/8.5 SSNG comparative analysis on metrics
    ("higher static stem nucleotide ratio", "produces", "better eterna total score"),
    ("higher static stem nucleotide ratio", "produces", "better eterna folding score"),
    ("higher static stem nucleotide ratio", "produces", "greater fold change"),
    ("higher static stem nucleotide ratio", "produces", "higher kdoff values"),
    ("higher static stem nucleotide ratio", "produces", "lower kdon values"),
    ("smaller ensemble groups", "associated_with", "higher eterna scores"),
    ("smaller ensemble groups", "associated_with", "higher kdoff values"),
    ("ensemble groups fewer than 8000 structures", "associated_with", "better performance"),
    ("ensemble groups fewer than 4000 structures", "associated_with", "highest kdoff values"),
    ("ensemble groups fewer than 4000 structures", "associated_with", "lowest kdon values"),

    # Per-sublab findings
    ("ssng1", "optimal_ratio_range", "greater than 10 percent"),
    ("ssng1", "highest_eterna_total_score", "with higher static stem nucleotide ratios"),
    ("ssng2", "optimal_ratio_range", "20 to 26 percent"),
    ("ssng2", "has_dual_mode_kdon", "nominal and super-performing"),
    ("ssng3", "optimal_ratio_range", "12 to 20 percent"),
    ("ssng3", "max_score_ratio", "approximately 16 percent"),

    # =====================================================================
    # §9 Detailed Description of the Molecular Snare Observations
    # =====================================================================

    # Overview
    ("molecular snare", "detects", "specific molecules"),
    ("molecular snare", "binds", "specific molecules"),
    ("molecular snare", "forms", "loop around detected molecule in bound state"),
    ("molecular snare", "is_subsystem_of", "rna aptamer"),
    ("molecular snare", "composed_of", "independent static stem"),
    ("molecular snare", "composed_of", "triggering molecule binding sites"),
    ("independent static stem", "can_be", "5'3' static stem"),
    ("independent static stem", "anchors", "binding sites"),
    ("triggering molecule binding sites", "initially_connected_to", "static stem"),
    ("triggering molecule binding sites", "stretched_out_in", "unbound state"),
    ("triggering molecule binding sites", "temporarily_bind_to", "hybrid stems"),
    ("tab", "is_a", "binding site configuration"),
    ("tab", "forms_when", "binding sites bind hybrid stems in unbound state"),
    ("target molecule encounter", "triggers", "binding sites pull inward"),
    ("binding sites pull inward", "creates", "tension in rna strand"),
    ("binding sites pull inward", "initiates", "secondary structure shift"),
    ("binding sites", "engage_with", "target molecule"),
    ("stable loop around target", "secures", "target molecule in place"),

    # Magnet Stem concept
    ("magnet stem", "used_in", "opentb winning design"),
    ("magnet stem", "used_for", "tuberculosis sensor development"),
    ("magnet stem", "is_essentially", "molecular snare"),
    ("magnet stem", "enhances", "aptamer performance"),
    ("magnet stem", "increases", "fold change potential"),
    ("opentb round", "is_a", "eterna round"),
    ("tuberculosis sensor", "developed_from", "magnet stem insight"),

    # 5'3' static stem + snare
    ("5'3' static stem as independent static stem", "achieves", "fold change greater than 50"),
    ("5'3' static stem configuration", "required_for", "highest fold changes"),
    ("fmn binding sites", "increase_strength_of", "5'3' static stem"),
    ("fmn binding sites", "provide_support_for", "molecular snare"),
    ("unbound state", "characterized_by", "mechanical equilibrium"),
    ("unbound state", "has", "static stem and hybrid stems maintaining structure"),
    ("target encounter", "causes", "inward pull of binding sites"),

    # Key principles
    ("molecular snare", "is_a", "molecular switch"),
    ("molecular switch behavior", "essential_for", "biosensing"),
    ("molecular switch behavior", "essential_for", "therapeutic applications"),
    ("energy conservation", "essential_for", "aptamer design"),
    ("energy efficient aptamers", "achieve", "higher fold changes"),
    ("energy efficient aptamers", "achieve", "greater stability"),

    # §9.1 Molecular Snare Mechanics Hypothesis (7-step temporal process)
    ("molecular snare mechanics hypothesis", "describes_steps", "7"),

    # Step 1
    ("molecular snare step 1", "name", "initial unbound state"),
    ("initial unbound state", "has", "mechanical equilibrium"),
    ("initial unbound state", "has", "5'3' static stem providing stability"),
    ("initial unbound state", "has", "hybrid stems maintaining structure"),
    ("initial unbound state", "has", "loops maintaining structure"),
    ("initial unbound state", "has", "relaxed molecular snare configuration"),
    ("initial unbound state", "has", "tabs stretched out and bound to hybrid stems"),

    # Step 2
    ("molecular snare step 2", "name", "encounter with target molecule"),
    ("target encounter step", "triggers", "molecular snare"),
    ("target encounter step", "causes", "tabs pulled inward"),
    ("target encounter step", "creates", "tension within rna strand"),
    ("target encounter step", "initiates", "secondary structure shift"),
    ("static stem during encounter", "remains", "stable"),

    # Step 3
    ("molecular snare step 3", "name", "initial structural shift"),
    ("tension from inward pull", "breaks", "hybrid stem temporary bonds"),
    ("initial structural shift", "begins", "loop formation around target"),
    ("static stem during shift", "provides", "controlled structural changes"),

    # Step 4
    ("molecular snare step 4", "name", "formation of loop"),
    ("loop formation", "reconfigures", "hybrid stems"),
    ("loop formation", "creates", "new bonds stabilizing loop"),
    ("loop formation", "relies_on", "static stem intact"),
    ("positive axial forces in loop", "exceed", "strength of negative axial bonds of stacks"),

    # Step 5
    ("molecular snare step 5", "name", "stabilization of bound state"),
    ("stabilization", "creates", "new bonds from reconfigured hybrid stems"),
    ("stabilization", "securely_traps", "target molecule"),
    ("static stem during stabilization", "anchors", "rna structure"),

    # Step 6
    ("molecular snare step 6", "name", "full engagement with target"),
    ("full engagement", "achieves", "stable bound state"),
    ("full engagement", "activates", "molecular snare completely"),

    # Step 7
    ("molecular snare step 7", "name", "mechanical equilibrium in bound state"),
    ("bound state mechanical equilibrium", "maintains", "stability"),
    ("bound state mechanical equilibrium", "completes", "molecular snare mechanism"),

    # §9.1 Energy conservation + structural resemblance
    ("unbound ensemble resembling bound ensemble", "eases", "structural shift"),
    ("unbound ensemble resembling bound ensemble", "increases", "energy conservation"),
    ("structural resemblance", "reduces", "transition energy required"),
    ("structural resemblance", "makes_more_efficient", "binding process"),

    # §9.1 Molecular Snare Thermodynamics Hypothesis
    ("molecular snare thermodynamics hypothesis", "grounded_in", "first law of thermodynamics"),
    ("first law of thermodynamics", "states", "change in internal energy equals heat plus work"),
    ("first law of thermodynamics", "formula", "delta u equals q plus w"),
    ("molecular snare encounter", "has_assumed_q_zero", "heat transfer is zero"),
    ("work done on system", "results_in", "positive delta u"),
    ("mechanical work in snare", "equals", "forces from hydrogen bonds in snare and target"),
    ("structural shifts in binding", "accompanied_by", "positive internal energy shift"),

    # §9.1 Combating heat loss
    ("molecular snare", "combats", "heat loss"),
    ("heat loss", "affects", "aptamer stability"),
    ("heat loss", "affects", "aptamer performance"),
    ("minimizing transition energy", "combats", "heat loss"),
    ("designing rna resembling bound state in unbound state", "reduces", "transition energy"),

    # §9.1 Initiation
    ("molecular snare action", "initiates", "chemically measured changes in rna aptamer switch"),
    ("molecular snare action", "triggers", "secondary structure shift"),

    # §9.2 Molecular_snare_nuc_to_total metric
    ("molecular_snare_nuc_to_total", "is_a", "metric"),
    ("molecular_snare_nuc_to_total", "is_sub_metric_of", "comparison structures"),
    ("molecular_snare_nuc_to_total", "quantifies", "ratio of static nucleotides in snare to total"),
    ("higher molecular_snare_nuc_to_total ratio", "indicates", "more static molecular snare"),
    ("more static molecular snare", "contributes_to", "more stable folding"),
    ("molecular_snare_nuc_to_total", "has", "artificial hard limit"),
    ("molecular_snare_nuc_to_total", "enables", "comparative analysis of rna designs"),

    # Per-sublab snare findings
    ("ssng1 molecular snare", "requires_for_score_100", "at least 18 percent static nucleotide ratio"),
    ("ssng1 molecular snare", "drops_sharply_below", "18 percent ratio"),
    ("ssng3 molecular snare", "optimal_ratio", "20 percent"),
    ("ssng2 molecular snare", "achieves_score_100_even_below", "24 percent ratio"),
    ("ssng2 flexibility", "attributed_to", "5'3' static stem integration"),

    # =====================================================================
    # §10 Fold Signal Region (detailed composition and mechanics)
    # =====================================================================

    ("fold signal region composition", "includes", "static stems"),
    ("fold signal region composition", "includes", "dynamic stems"),
    ("fold signal region composition", "includes", "hybrid stems"),
    ("fold signal region composition", "includes", "static loops"),
    ("fold signal region composition", "includes", "dynamic loops"),
    ("larger fold signal region", "provides", "more space for structural changes"),
    ("too large fold signal region", "becomes", "mechanically unstable"),
    ("smaller fold signal region", "may_not_provide", "enough space for conformational changes"),
    ("smaller fold signal region", "produces", "suboptimal performance"),

    # Stem and loop types
    ("static stems", "do_not_change_pairs", "between state changes"),
    ("static stems", "provide", "stable backbone"),
    ("dynamic stems", "change_pairs", "between states"),
    ("dynamic stems", "form", "different stacks between states"),
    ("dynamic stems", "allow", "flexibility"),
    ("hybrid stems", "retain", "same dot-parenthesis structure between states"),
    ("hybrid stems", "have", "different nucleotide pairs between states"),
    ("hybrid stems", "balance", "stability and flexibility"),
    ("static loops", "have", "same nucleotides between states"),
    ("static loops", "act_as", "stable points"),
    ("dynamic loops", "form", "different loops in different states"),
    ("dynamic loops", "contribute_to", "flexibility"),
    ("hybrid stacks", "create", "temporary stable structures in unbound state"),
    ("hybrid stacks", "shift_to", "new stable structures in bound state"),

    # Mechanical movement constraints
    ("mechanical movement constraints", "occur_at", "smaller fold signal region sizes"),
    ("mechanical movement constraints", "result_in", "binding issues"),
    ("backbone limited bending ability", "restricts", "movement"),
    ("backbone limited bending", "analogous_to", "stack of quarters with slight gaps"),
    ("sufficient fold signal region size", "enables", "efficient mechanical movement"),
    ("optimal fold signal region size", "balances", "flexibility and stability"),

    # Static loops as fulcrums
    ("static loops as fulcrums", "provide", "pivot points for structural changes"),
    ("static loops as fulcrums", "act_as", "hinges"),
    ("static loops as fulcrums", "reduce", "energy required for structural changes"),
    ("fulcrum limits", "defined_by", "rna backbone bending ability"),
    ("static loops", "must_be", "sufficient size to be efficient"),

    # High forces
    ("fold signal region during state transition", "experiences", "high internal forces"),
    ("state transition forces", "include", "tension"),
    ("state transition forces", "include", "compression"),
    ("state transition forces", "include", "bending moments"),
    ("fold signal region", "requires", "reinforcement"),
    ("static stems", "provide", "reinforcement to fold signal region"),
    ("loops", "provide", "reinforcement to fold signal region"),
    ("static stems as rigid backbones", "support", "fold signal region"),

    # Conservation of energy
    ("efficient energy use during state transition", "essential_for", "rapid binding"),
    ("efficient energy use during state transition", "essential_for", "effective binding"),
    ("static loops pivot points", "reduce", "energy required for structural changes"),
    ("dynamic and hybrid stems", "contribute_to", "energy conservation"),
    ("dynamic and hybrid stems", "allow", "flexibility with low energy"),
    ("balance between static and dynamic elements", "crucial_for", "energy conservation"),

    # =====================================================================
    # §11 Fold Signal Region Mechanics — FMN binding walkthrough
    # =====================================================================

    # 11.1 No FMN Present
    ("fmn binding state 1", "name", "no fmn present"),
    ("rna in no fmn state", "is_at", "rest"),
    ("rna in no fmn state", "is_in", "low-energy state"),
    ("rna in no fmn state", "is_in", "mechanical equilibrium"),
    ("intramolecular forces in no fmn state", "include", "hydrogen bonds"),
    ("intramolecular forces in no fmn state", "include", "base stacking interactions"),
    ("no fmn state", "has_formed", "5'3' static stem"),
    ("no fmn state", "has_formed", "molecular snare static stems"),
    ("no fmn state", "has_formed", "hybrid stems with dynamic switching pairs"),

    # 11.2 FMN Presented: Bind Step 1
    ("fmn binding state 2", "name", "bind step 1 fmn presented"),
    ("fmn molecule", "begins_binding_to", "aptamer"),
    ("fmn binding", "triggers", "molecular snare"),
    ("tab of nucleotides", "pulled_toward", "center of binding sites"),
    ("nucleotides pulled in", "causes", "compression of system"),
    ("compression of system", "creates", "tension in rna strand"),
    ("compression and tension", "initiate", "structural shift"),
    ("structural shift", "forces", "hybrid stems to break bonds and switch to new nucleotides"),
    ("bottom half of unbound structure", "shifts_away_from", "center mass"),
    ("new structure in fmn binding", "seeks", "new mechanical equilibrium"),
    ("lower resistance to new equilibrium", "produces", "higher potential fold change"),

    # 11.3 Bind Step 2
    ("fmn binding state 3", "name", "bind step 2"),
    ("bind step 2", "produces", "toggle from state a to state b"),
    ("bind step 2", "transforms", "horizontal flat structure into vertical with ms2 hairpin"),
    ("bind step 2", "reveals", "snare-flexclamp-flexsignal design pattern"),
    ("hybrid stems with lost bonds", "attracted_to", "opposing hybrid stem lanes"),
    ("opposing hybrid stem attraction", "further_compresses", "signal fold region"),
    ("signal fold area in bind step 2", "elongates_vertically", "in parallel with snare orientation"),
    ("tension in phosphate backbone", "increases_during", "bind step 2"),
    ("tension in phosphate backbone", "generates", "forces attempting to rip signal fold apart"),
    ("5'3' static stem", "protects_against", "harmful forces during bind step 2"),
    ("5'3' static stem protection", "increases", "potential fold change"),

    # 11.4 Bind Step 3
    ("fmn binding state 4", "name", "bind step 3"),
    ("bind step 3", "is", "toggle from a to b nearly complete"),
    ("critical mass for state b", "achieved_by", "new hybrid stem lane and fmn bonds"),
    ("short stack connecting snare and signal fold", "is", "weak"),
    ("short stack", "has", "gu pair at end"),
    ("gu pair", "acts_as", "trigger for returning to state a"),
    ("fmn leaving", "destabilizes", "short stack"),
    ("rna in bind step 3", "seals_up_like", "zipper starting at 11:42 pair"),
    ("zipper sealing", "progresses_sequentially", "from one end to other"),

    # 11.5 FMN Bound
    ("fmn binding state 5", "name", "fmn bound"),
    ("rna aptamer fully bound", "reaches", "state b"),
    ("fmn bound state", "has", "fmn ligand fully bound"),
    ("fmn bound state", "has", "ms2 signal reporter fully exposed"),
    ("fmn bound state", "has", "hybrid stem lanes fully bound to new lanes"),

    # =====================================================================
    # §12 Snare-FlexClamp-FlexSignal Design Pattern
    # =====================================================================

    ("snare-flexclamp-flexsignal", "is_a", "design pattern"),
    ("snare-flexclamp-flexsignal", "has_variant", "with tail"),
    ("snare-flexclamp-flexsignal", "has_variant", "without tail"),
    ("flexclamp flexor", "count_in_design", "2"),
    ("flexclamp flexors", "located_on", "each side of secondary structure"),
    ("flexclamp flexors", "part_of", "loops"),
    ("flexclamp flexors", "provide", "pivot points"),
    ("flexsignal flexor", "count_in_design", "1"),
    ("flexsignal flexor", "located_between", "two flexclamp flexors"),
    ("flexsignal flexor", "part_of", "loop"),
    ("flexsignal flexor", "facilitates", "signal region formation"),

    # 12.5 FMN Presented + begins binding
    ("flexclamp pivot 1", "begins_opening_when", "internal structure moves right"),
    ("flexclamp pivot 2", "begins_opening_when", "internal structure moves left"),
    ("flexsignal pivot 1", "begins_closing_when", "internal structure moves down"),
    ("flexsignal pivot 1 closing", "forms", "signal region"),

    # 12.7 FMN binding progresses
    ("hybrid stems in binding", "attracted_to", "new base pair configurations"),
    ("flexclamp flexor 1 during binding", "opens_more", "pulled right"),
    ("flexclamp flexor 2 during binding", "opens_more", "pulled left"),

    # 12.9 FMN binding nears completion
    ("hybrid stems near completion", "almost_fully_bound_to", "new base pairs"),
    ("flexclamp 1 and 2", "approach", "each other"),
    ("flexclamp 1 and 2", "form", "multi-loop"),
    ("flexsignal 1", "closes_when", "ms2 signal forms"),

    # 12.11 FMN fully bound
    ("flexclamp 1 and 2 fully bound", "form", "multi-loop structure"),
    ("flexsignal 1 fully bound", "forms", "ms2 reporter signal"),

    # 12.13 Fully formed
    ("flexclamp fully formed", "consists_of", "stack connecting flexor loop to snare"),
    ("flexsignal fully formed", "consists_of", "stack connecting signal to flexclamp"),
    ("ssng1", "has", "clamp and signal overlap"),
    ("ssng2", "has", "non-overlapping clamp and signal"),
    ("ssng1 overlap", "caused_by", "5'3' static stem forcing reduced fold area"),
    ("ssng1 overlap", "results_in", "lower fold change"),

    # Design Pattern Implementation Hypothesis
    ("snare-flexclamp-flexsignal implementation", "involves", "analyzing known good fmn-activated ms2 reporter"),
    ("snare-flexclamp-flexsignal implementation", "modifies", "design to remove ms2 reporter"),
    ("snare-flexclamp-flexsignal implementation", "retains", "fmn binding ability"),
    ("ms2 signal region", "can_be_converted_to", "simple hairpin"),
    ("modified aptamer", "undergoes", "conformational changes on fmn binding"),
    ("modified aptamer", "eliminates", "ms2 signaling component"),
    ("modified aptamer", "may_increase", "fold change by order of magnitude"),

    # =====================================================================
    # §13 The Knob, In Conclusion
    # =====================================================================

    ("the paper", "provides", "comprehensive analysis of rna aptamer design"),
    ("the paper", "emphasizes", "limitations of selex"),
    ("the paper", "proposes", "systematic approach"),
    ("supplementing selex", "with", "computationally guided design"),
    ("computationally guided design", "enhances", "rna aptamer development"),
    ("integrated approach", "reduces_reliance_on", "trial-and-error methods"),
    ("integrated approach", "accelerates", "rna aptamer design process"),
    ("integrated approach", "improves", "rna aptamer performance"),

    # Understanding the key components
    ("understanding 5'3' static stem", "crucial_for", "rna therapeutics design"),
    ("understanding molecular snare", "crucial_for", "rna therapeutics design"),
    ("understanding signal fold region", "crucial_for", "rna therapeutics design"),
    ("5'3' static stem", "provides", "stability to withstand mechanical stresses during state transitions"),
    ("5'3' static stem", "is_a", "stabilizing backbone"),
    ("molecular snare", "captures_and_holds", "target molecules"),
    ("molecular snare", "initiates", "conformational changes in aptamer"),
    ("signal fold region", "undergoes", "significant structural changes on target binding"),
    ("signal fold region", "facilitates", "transition from unbound to bound state"),

    # Therapeutic vision
    ("predictive rna-based tools", "require", "accounting for rna mechanical movements"),
    ("predictive rna-based tools", "require", "understanding rna mechanical limitations"),
    ("fine-tuned aptamer", "can_detect", "diseases in human body"),
    ("fine-tuned aptamer", "can_disable", "diseases in human body"),
    ("aptamer with affinity for bound state", "could_be_designed_to", "never let go in body"),
    ("aptamer that never lets go", "could", "render disease inert for period of time"),
    ("rna aptamer affinity", "depends_on", "configuration"),
    ("knowledge of rna mechanics", "enables", "rapid rna aptamer development"),
    ("knowledge of rna mechanics", "enables", "targeted rna aptamer development"),

    # =====================================================================
    # §14 The Knob Turn — section heading only, no body; carry thesis
    # §15 Observation C — placeholder for follow-up paper
    # =====================================================================

    ("the knob turn", "is_a", "section"),
    ("observation c", "will_be_covered_in", "follow-up paper"),

    # =====================================================================
    # Full-Disclosure / authorship
    # =====================================================================

    ("the paper", "written_with_assistance_of", "chat-gpt"),
    ("chat-gpt", "wrote_reports_on_sections_of", "200 slide presentation by jennifer pearl"),
    ("200 slide presentation", "created_by", "jennifer pearl"),
    ("200 slide presentation", "created_without", "llm or ai assistance"),
    ("the research", "performed_by", "jennifer pearl"),
    ("the paper draft", "written_by", "chat-gpt4"),
    ("chat-gpt4 draft", "based_exclusively_on", "jennifer pearl's 200 slide presentation"),
    ("chat-gpt4 draft", "based_exclusively_on", "jennifer pearl's book on theory of mechanics of rna"),
    ("jennifer pearl's book", "topic", "theory of mechanics of rna"),
]


def main() -> None:
    db_path = Path("aptamer_full.db")
    if db_path.exists():
        raise FileExistsError(
            f"{db_path} exists — delete or rename before re-teaching"
        )

    brain = Brain(str(db_path))
    print(f"Fresh brain: {db_path}")
    print(f"{len(TRIPLES)} triples to teach (full paper, §2–§15)\n")

    t0 = time.time()
    for i, (s, r, o) in enumerate(TRIPLES, 1):
        result = brain.teach_triple(s, r, o, source_label=SOURCE)
        ok = "OK " if result else "FAIL"
        if i % 25 == 0 or not result:
            print(f"[{i:4d}/{len(TRIPLES)}] {ok} | {s[:40]!r} --[{r}]--> {o[:40]!r}")

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
