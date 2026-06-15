#!/usr/bin/env python3
"""Full biology ingestion using the v2-clean extractor.

This processes all 47 biology chapters using the verified v2-clean
extractor and stores the result in a persistent location.

Total time: ~8 hours on RTX 3070.
"""

import os
import sys
import glob
import subprocess
import time

DB_PATH = "data/biology_full_v2_clean.db"
FACTS_DIR = "benchmarks/biology2e_facts"

if not os.path.exists("data"):
    os.makedirs("data")

if os.path.exists(DB_PATH):
    print(f"Removing old DB at {DB_PATH}...")
    os.remove(DB_PATH)

files = sorted(glob.glob(f"{FACTS_DIR}/ch*_facts.txt"))
if not files:
    print(f"No files found in {FACTS_DIR}!")
    sys.exit(1)

print(f"Found {len(files)} chapters to ingest using v2-clean extractor.")
t0 = time.time()

for f in files:
    ch_name = os.path.basename(f).replace("_facts.txt", "")
    print(f"=== Ingesting {ch_name} ===")

    cmd = [
        ".venv/bin/python", "src/sara_reader/cli_teach_book.py",
        "--brain", DB_PATH,
        "--extractor", "sara",
        "--multipass",
        "--quiet",
        f
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {f}: {e}")
        # Keep going on error to avoid losing progress
        continue

elapsed = time.time() - t0
print(f"\nDone ingesting all {len(files)} chapters in {elapsed:.1f}s ({elapsed/3600:.1f}h).")

# Print stats
subprocess.run([".venv/bin/python", "sara_q.py", "stats", "--db", DB_PATH])
