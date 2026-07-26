"""Generate raw substrate text for Phase 1 language model training.

Creates synthetic brains, runs wavefront queries, saves the output
text. No labels needed — just substrate format for next-token prediction.

Usage:
    python scripts/generate_substrate_lm_data.py \
        --num-substrates 5000 --queries-per-substrate 20 \
        --out training_data/substrate_lm_100k.txt
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sara_brain.core.brain import Brain
from sara_reader.stateless_reader import (
    _filter_seeds_by_substrate,
    _format_wavefront_substrate,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-substrates", type=int, default=5000)
    ap.add_argument("--queries-per-substrate", type=int, default=20)
    ap.add_argument("--concepts", type=int, default=40)
    ap.add_argument("--triples", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=9999)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")
    chunks = 0
    t0 = time.time()

    for i in range(args.num_substrates):
        db_path = f"/tmp/lm_synth_{i}.db"
        sub_seed = args.seed + i

        # Generate substrate
        result = subprocess.run(
            [sys.executable, "papers/instrument_validation/generate_synthetic_substrate.py",
             "--out", db_path, "--concepts", str(args.concepts),
             "--triples", str(args.triples), "--seed", str(sub_seed),
             "--compound-fraction", "0.5"],
            capture_output=True, text=True)
        if result.returncode != 0:
            continue

        brain = Brain(db_path)

        # Run random wavefront queries
        all_neurons = [n for n in brain.neuron_repo.list_all() if n.neuron_type.value == "concept"]
        if not all_neurons:
            brain.close()
            continue

        for _ in range(args.queries_per_substrate):
            # Pick 1-3 random concept labels as seeds
            n_seeds = rng.randint(1, 3)
            seed_neurons = rng.sample(all_neurons, min(n_seeds, len(all_neurons)))
            seeds = [n.label for n in seed_neurons]

            try:
                brain.recognizer.max_depth = 2
                with brain.short_term(event_type="lm_gen") as st:
                    brain.propagate_into(seeds, st, exact_only=True)
                    convergence_map = dict(st.convergence_map)
                    intersections = st.intersections(min_sources=2)
                text = _format_wavefront_substrate(brain, seeds, convergence_map, intersections)
                if text.strip():
                    out_f.write(text + "\n\n")
                    chunks += 1
            except Exception:
                continue

        brain.close()
        # Cleanup
        for f in Path("/tmp").glob(f"lm_synth_{i}.db*"):
            f.unlink(missing_ok=True)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{args.num_substrates}] chunks={chunks} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    out_f.close()
    print(f"\nDone. {chunks} chunks written to {args.out} ({time.time()-t0:.0f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
