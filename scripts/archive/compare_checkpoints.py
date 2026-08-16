"""Compare two hamroby_extractor_v1 checkpoints on the failure battery.

Runs the diagnostic harness against two checkpoints (default: the
baseline at `hamroby_extractor_v1.pt` and a candidate at
`hamroby_extractor_v1_aug.pt`) and prints per-sentence (baseline →
candidate) deltas. Used after retraining to see which failures the
augmentations fixed and whether anything regressed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from sara_brain.cortex.transformer.hamroby_extractor_v1.feature_extractor import (
    load_domain_nlp,
    parse_sentence,
)
from sara_brain.cortex.transformer.hamroby_extractor_v1.extraction_head import (
    ExtractionHead,
)
from sara_brain.cortex.transformer.hamroby_extractor_v1.model import (
    ExtractorConfig,
    GrammarEncoder,
)
from sara_brain.cortex.transformer.hamroby_extractor_v1.decoder import decode


BATTERY: list[tuple[str, str]] = [
    # Each row is (sentence, category). Categories group sentences by
    # the structural pattern under test, so per-pattern pass rates can
    # be read off the report instead of guessing from one example.

    # === intj_pp: weird-shape subject token + PP head-modifier + cop ===
    ("K_d for the binding is 1.2nM.", "intj_pp"),
    ("ATP at the active site is 5mM.", "intj_pp"),
    ("EC50 for the inhibitor is 30nM.", "intj_pp"),
    ("delta_G of the reaction is -2.3 kcal.", "intj_pp"),
    ("k_off during the wash is 0.04 per second.", "intj_pp"),
    ("p_max in the trial is 0.92.", "intj_pp"),

    # === intj_bare: weird-shape subject token, NO PP modifier ===
    ("K_d remains constant.", "intj_bare"),
    ("p<0.05 indicates significance.", "intj_bare"),
    ("ATP binds the receptor.", "intj_bare"),
    ("alpha-helices fold rapidly.", "intj_bare"),

    # === compound_oblique: compound subject NP + transitive verb + dobj + oblique PP ===
    ("Marker theory predicts kdoff with p<0.05.", "compound_oblique"),
    ("Cluster analysis groups proteins by similarity.", "compound_oblique"),
    ("Computational models predict folding under stress.", "compound_oblique"),
    ("The control system stabilizes pressure during loading.", "compound_oblique"),
    ("Statistical tests evaluate significance via permutation.", "compound_oblique"),

    # === weird_token_no_pp: sentence-initial weird token, NOT in copular position ===
    ("The 5'3' static stem provides stability.", "weird_token_no_pp"),
    ("The K_d-bound complex resists denaturation.", "weird_token_no_pp"),
    ("The pH-7 buffer protects the enzyme.", "weird_token_no_pp"),

    # === gerund_subject: gerund-headed NP as subject ===
    ("Noticing a limitation shows the way.", "gerund_subject"),
    ("Solving the equation requires care.", "gerund_subject"),
    ("Finding the binding pocket helps the design.", "gerund_subject"),
    ("Refusing the offer surprised the team.", "gerund_subject"),

    # === conj_subject: conjoined-NP subject ===
    ("John and Mary went home.", "conj_subject"),
    ("Alice and Bob signed the contract.", "conj_subject"),
    ("The receptor and the ligand bind tightly.", "conj_subject"),
    ("DNA and RNA share base pairing.", "conj_subject"),

    # === conj_object: conjoined-NP direct object ===
    # Existing 4 cover the original named cases.
    ("The system processes books and papers.", "conj_object"),
    ("The reaction produces water and salt.", "conj_object"),
    ("She bought apples and oranges.", "conj_object"),
    ("Researchers compared yeast and mammalian cells.", "conj_object"),
    # Pronoun-subject probes: isolate whether PRON subject + bare conj
    # dobj is the systematic failure (vs aug4's single observation on
    # "She bought apples and oranges").
    ("He read books and magazines.", "conj_object"),
    ("They eat fruits and vegetables.", "conj_object"),
    ("We study cells and tissues.", "conj_object"),
    ("I carry pens and pencils.", "conj_object"),
    # Pronoun-subject + DET+NOUN conj dobj (variant — does article
    # rescue the dobj for pronoun subjects?):
    ("She bought the apples and the oranges.", "conj_object"),
    ("He read a book and a magazine.", "conj_object"),
    # Noun-subject + bare conj dobj (variant — confirms whether the
    # article on the subject ("The system") was load-bearing):
    ("Plants need sunlight and water.", "conj_object"),
    ("Bacteria release toxins and enzymes.", "conj_object"),

    # === pron_svo: pronoun-subject + bare-noun dobj (no conjunction) ===
    # Probes whether PRON-subject + bare-dobj alone fails, isolating
    # the conjunction as the trigger.
    ("She read books.", "pron_svo"),
    ("He bought apples.", "pron_svo"),
    ("They eat fruits.", "pron_svo"),

    # === propn_aux: multi-word proper noun containing a capitalized auxiliary-like token ===
    ("Bruce Lee created Jeet Kune Do.", "propn_aux"),
    ("Ada Lovelace wrote the first algorithm.", "propn_aux"),
    ("Marie Curie discovered radium.", "propn_aux"),
    ("The Hubble Space Telescope observed distant galaxies.", "propn_aux"),

    # === svo_basic: plain transitive SVO controls ===
    ("She built a house.", "svo_basic"),
    ("Lee Smith leads the team.", "svo_basic"),
    ("The frog catches flies.", "svo_basic"),
    ("The student wrote a paper.", "svo_basic"),
    ("The teacher graded the assignments.", "svo_basic"),

    # === particle_verb: verb + particle-or-prep object pattern ===
    ("The protein folds into hairpins.", "particle_verb"),
    ("The team looks for evidence.", "particle_verb"),

    # === copular_simple: bare copular (X is Y) controls ===
    ("The molecular snare is critical.", "copular_simple"),
    ("The reagent is stable.", "copular_simple"),
    ("Caffeine is a stimulant.", "copular_simple"),

    # === intransitive: intransitive verb (no object) ===
    ("Strong proteins persist.", "intransitive"),
    ("The crystal grows slowly.", "intransitive"),
    ("Ice melts at zero.", "intransitive"),
]


def load_head(path: Path) -> ExtractionHead:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExtractorConfig(**raw["encoder_cfg"])
    head = ExtractionHead(GrammarEncoder(cfg))
    head.load_state_dict(raw["head_state"])
    head.eval()
    return head


def run(head: ExtractionHead, parsed) -> list:
    pos_ids = torch.tensor([[fid[0] for fid in parsed.feature_ids]], dtype=torch.long)
    dep_ids = torch.tensor([[fid[1] for fid in parsed.feature_ids]], dtype=torch.long)
    off_ids = torch.tensor([[fid[2] for fid in parsed.feature_ids]], dtype=torch.long)
    fw_ids = torch.tensor([[fid[3] for fid in parsed.feature_ids]], dtype=torch.long)
    tags = head.predict_tags(pos_ids, dep_ids, off_ids, fw_ids)[0].tolist()
    return decode(parsed, tags)


def fmt(triples) -> str:
    if not triples:
        return "NO TRIPLE"
    return " | ".join(
        f"({t.subject!r}, {t.relation!r}, {t.object!r})" for t in triples
    )


def main() -> int:
    base_path = Path(os.environ.get(
        "BASELINE",
        str(REPO / "src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt"),
    ))
    cand_path = Path(os.environ.get(
        "CANDIDATE",
        str(REPO / "src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug.pt"),
    ))
    if not base_path.exists():
        print(f"missing baseline: {base_path}", file=sys.stderr)
        return 1
    if not cand_path.exists():
        print(f"missing candidate: {cand_path}", file=sys.stderr)
        return 1

    print(f"baseline:  {base_path.name}", file=sys.stderr)
    print(f"candidate: {cand_path.name}", file=sys.stderr)
    base = load_head(base_path)
    cand = load_head(cand_path)
    nlp = load_domain_nlp()

    print(f"# Checkpoint comparison\n")
    print(f"- baseline: `{base_path.relative_to(REPO)}`")
    print(f"- candidate: `{cand_path.relative_to(REPO)}`\n")

    categories: list[str] = []
    for _, cat in BATTERY:
        if cat not in categories:
            categories.append(cat)

    summary: list[tuple[str, int, int, int, int]] = []  # (cat, n, changed, fixed, regressed)

    for cat in categories:
        sentences = [s for s, k in BATTERY if k == cat]
        print(f"## {cat}\n")
        n = changed = fixed = regressed = 0
        for sent in sentences:
            parsed = parse_sentence(sent, nlp)
            base_t = run(base, parsed)
            cand_t = run(cand, parsed)
            base_str = fmt(base_t)
            cand_str = fmt(cand_t)
            same = base_str == cand_str
            n += 1
            if not same:
                changed += 1
                # Heuristic gain/loss: NO TRIPLE → triple = fixed; triple → NO TRIPLE = regressed.
                # Anything else changed but ambiguous (manual read needed).
                if base_str == "NO TRIPLE" and cand_str != "NO TRIPLE":
                    fixed += 1
                elif base_str != "NO TRIPLE" and cand_str == "NO TRIPLE":
                    regressed += 1
            arrow = "=" if same else "→"
            print(f"### `{sent}`\n")
            print(f"- baseline:  {base_str}")
            print(f"- candidate: {cand_str} {'(unchanged)' if same else arrow + ' CHANGED'}\n")
        summary.append((cat, n, changed, fixed, regressed))

    print("## Summary by category\n")
    print("| category | n | unchanged | changed | NO→triple | triple→NO |")
    print("|---|---:|---:|---:|---:|---:|")
    for cat, n, changed, fixed, regressed in summary:
        unchanged = n - changed
        print(f"| {cat} | {n} | {unchanged} | {changed} | {fixed} | {regressed} |")
    total_n = sum(r[1] for r in summary)
    total_changed = sum(r[2] for r in summary)
    total_fixed = sum(r[3] for r in summary)
    total_regressed = sum(r[4] for r in summary)
    print(
        f"| **total** | **{total_n}** | **{total_n - total_changed}** | "
        f"**{total_changed}** | **{total_fixed}** | **{total_regressed}** |\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
