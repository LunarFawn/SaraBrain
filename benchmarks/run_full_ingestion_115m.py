#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import time

DB_PATH = "data/biology_full_115m.db"
FACTS_DIR = "benchmarks/biology2e_facts"

if os.path.exists(DB_PATH):
    print(f"Removing old DB at {DB_PATH}...")
    os.remove(DB_PATH)

files = sorted(glob.glob(f"{FACTS_DIR}/ch*_facts.txt"))
if not files:
    print(f"No files found in {FACTS_DIR}!")
    sys.exit(1)

print(f"Found {len(files)} chapters to ingest.")
t0 = time.time()

for f in files:
    ch_name = os.path.basename(f).replace("_facts.txt", "")
    print(f"=== Ingesting {ch_name} ===")
    
    cmd = [
        ".venv/bin/python", "src/sara_reader/cli_teach_book.py",
        "--brain", DB_PATH,
        "--extractor", "sara",
        "--quiet", # reduce spam
        f
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {f}: {e}")
        sys.exit(1)

elapsed = time.time() - t0
print(f"\nDone ingesting all {len(files)} chapters in {elapsed:.1f}s ({elapsed/60:.1f}m).")

# Print stats
subprocess.run([".venv/bin/python", "sara_q.py", "stats", "--db", DB_PATH])
