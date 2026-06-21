# description_frequency.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Scans every career_history entry across the full candidates.jsonl,
#   hashes each description with MD5, counts global frequency, and writes
#   a JSON file sorted by frequency descending (most cloned at top).
#
# WHY MD5
#   We need a fixed-length key per description for the hash table.
#   MD5 is fast and collision probability is negligible for ~1M strings
#   of this length. We store the original text alongside the hash so
#   the output is human-readable.
#
# OUTPUT
#   description_frequency.json  — sorted array, highest count first
#   Each entry:
#     {
#       "count":        369,          ← how many times across full dataset
#       "description":  "built ...",  ← original text
#       "md5":          "a3f...",      ← hash key used for dedup
#       "candidate_ids": ["CAND_...", ...]  ← which candidates carry this desc
#     }
#
# USAGE
#   python description_frequency.py
#   (run from the folder containing candidates.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import time
import hashlib
from collections import defaultdict

INPUT_FILE  = "candidates.jsonl"
OUTPUT_FILE = "description_frequency.json"


def md5(text: str) -> str:
    """MD5 hash of a stripped description string."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def run():
    if not os.path.exists(INPUT_FILE):
        print(f"❌  File not found: {INPUT_FILE}")
        return

    start = time.time()

    # Hash table:
    #   key   → md5 of description
    #   value → {"description": str, "count": int, "candidate_ids": set}
    freq: dict[str, dict] = {}

    total_candidates = 0
    total_descriptions = 0
    skipped_lines = 0

    print(f"📂  Reading {INPUT_FILE} ...")

    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"    ⚠  Skipped line {i}: {e}")
                skipped_lines += 1
                continue

            total_candidates += 1
            cid = c.get("candidate_id", f"UNKNOWN_{i}")

            for job in c.get("career_history", []):
                desc = (job.get("description") or "").strip()
                if not desc:
                    continue

                total_descriptions += 1
                h = md5(desc)

                if h not in freq:
                    freq[h] = {
                        "md5":           h,
                        "description":   desc,
                        "count":         0,
                        "candidate_ids": []
                    }

                freq[h]["count"] += 1
                # store candidate_id only once per candidate
                # (avoid duplicating if same candidate has same desc twice)
                if not freq[h]["candidate_ids"] or freq[h]["candidate_ids"][-1] != cid:
                    freq[h]["candidate_ids"].append(cid)

            if i % 10_000 == 0:
                print(f"    ... {i:,} candidates scanned")

    print(f"    ✅ Scan complete — {total_candidates:,} candidates, "
          f"{total_descriptions:,} descriptions, "
          f"{len(freq):,} unique descriptions\n")

    # Sort by count descending
    sorted_entries = sorted(freq.values(), key=lambda x: x["count"], reverse=True)

    # Write output
    print(f"💾  Writing {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        json.dump(sorted_entries, fh, ensure_ascii=False, indent=2)

    elapsed = time.time() - start

    # Print top 20 for quick inspection
    print(f"\n{'RANK':<5} {'COUNT':>6}  DESCRIPTION (first 90 chars)")
    print("-" * 100)
    for rank, entry in enumerate(sorted_entries[:20], 1):
        preview = entry["description"][:90].replace("\n", " ")
        print(f"{rank:<5} {entry['count']:>6}  {preview}")

    print()
    print("=" * 60)
    print("  DESCRIPTION FREQUENCY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Total candidates scanned  : {total_candidates:,}")
    print(f"  Total descriptions parsed : {total_descriptions:,}")
    print(f"  Unique descriptions       : {len(freq):,}")
    print(f"  Most cloned description   : {sorted_entries[0]['count']} times")
    print(f"  Descriptions seen once    : {sum(1 for e in sorted_entries if e['count'] == 1):,}")
    print(f"  Output file               : {os.path.abspath(OUTPUT_FILE)}")
    print(f"  Wall time                 : {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    run()