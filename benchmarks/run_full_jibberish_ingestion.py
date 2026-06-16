#!/usr/bin/env python3
"""Ingest Jibberished Biology into a new Giant Brain.
"""

import os
import sys
import glob
import subprocess
import time

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-chapter", type=int, default=1, help="Chapter number to start/resume from")
    args = ap.parse_args()

    DB_PATH = "data/jibberish_biology_v2_stable.db"
    FACTS_DIR = "data/biology_jibberish"

    if not os.path.exists("data"):
        os.makedirs("data")

    # Only remove if starting from chapter 1
    if args.start_chapter == 1 and os.path.exists(DB_PATH):
        print(f"Removing old DB at {DB_PATH}...")
        os.remove(DB_PATH)

    all_files = sorted(glob.glob(f"{FACTS_DIR}/ch*_facts.txt"))
    # Filter files based on start chapter
    files = []
    for f in all_files:
        try:
            # Extract chapter number from filename like 'ch01_facts.txt'
            name = os.path.basename(f)
            ch_num_str = name.split('_')[0].replace('ch', '')
            ch_num = int(ch_num_str)
            if ch_num >= args.start_chapter:
                files.append(f)
        except Exception as e:
            continue

    if not files:
        print(f"No files found to process (start_chapter={args.start_chapter})")
        return

    print(f"Ingesting {len(files)} chapters starting from ch{args.start_chapter}...")
    t0 = time.time()

    for f in files:
        ch_name = os.path.basename(f).replace("_facts.txt", "")
        print(f"=== Ingesting {ch_name} ===")

        cmd = [
            ".venv/bin/python", "src/sara_reader/cli_teach_book.py",
            "--brain", DB_PATH,
            "--extractor", "sara",
            "--multipass",
            "--no-dictionary",
            "--quiet",
            f
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {f}: {e}")
            continue

    elapsed = time.time() - t0
    print(f"\nDone ingesting all {len(files)} chapters in {elapsed:.1f}s ({elapsed/3600:.1f}h).")

    # Print stats
    subprocess.run([".venv/bin/python", "sara_q.py", "stats", "--db", DB_PATH])

if __name__ == "__main__":
    main()
