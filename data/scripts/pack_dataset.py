#!/usr/bin/env python3
"""
scripts/pack_dataset.py
-----------------------
Convert data_Clean/Dataset/{country}/{experience}/{domain}/{availability}/candidates.json
(each file is a JSON array) into:

  data/candidates.zip       — flat ZIP pruning.py can consume
  data/candidates.jsonl     — flat JSONL for rank.py --candidates mode

ZIP internal structure:
  {country}_{experience}_{domain}_{availability}/candidates.json

Folder naming uses underscores so pruning.py can parse each path component.

Usage:
  python scripts/pack_dataset.py
  python scripts/pack_dataset.py --dataset data_Clean/Dataset --out data/candidates.zip
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path


# ── Helpers ──────────────────────────────────────────────────────────────────

def iter_leaf_folders(dataset_root: Path):
    """
    Yield (folder_name, merged_candidates_list) for every
    {availability}/{country}/{experience}/{domain}/ group.

    Actual tree written by pipeline_1.py:
        {availability}/{country}/{experience}/{domain}/{role}.json

    All role files within a domain folder are merged into one list.
    ZIP folder name: {country}_{experience}_{domain}_{availability}
    """
    for avail_dir in sorted(dataset_root.iterdir()):
        if not avail_dir.is_dir():
            continue
        for country_dir in sorted(avail_dir.iterdir()):
            if not country_dir.is_dir():
                continue
            for exp_dir in sorted(country_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                for domain_dir in sorted(exp_dir.iterdir()):
                    if not domain_dir.is_dir():
                        continue
                    # Merge all role JSON files in this domain folder
                    merged = []
                    for role_file in sorted(domain_dir.glob("*.json")):
                        merged.extend(load_candidates(role_file))
                    if not merged:
                        continue
                    folder_name = "_".join([
                        country_dir.name,
                        exp_dir.name,
                        domain_dir.name,
                        avail_dir.name,
                    ])
                    yield folder_name, merged


def load_candidates(path: Path) -> list:
    """
    Load candidates.json → list.
    Handles:
      - JSON array  (normal case)
      - JSONL       (one object per line, fallback)
    Returns empty list on error or empty file.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        # Unlikely, but if it's a single object wrap it
        if isinstance(data, dict):
            return [data]
        return []
    except json.JSONDecodeError:
        # Fallback: try JSONL
        results = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pack data_Clean/Dataset/ tree → data/candidates.zip + candidates.jsonl"
    )
    parser.add_argument(
        "--dataset",
        default="data_Clean/Dataset",
        help="Root of the cleaned dataset tree (default: data_Clean/Dataset)",
    )
    parser.add_argument(
        "--out",
        default="data/candidates.zip",
        help="Output ZIP path (default: data/candidates.zip)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    zip_out = Path(args.out)

    if not dataset_root.exists():
        print(f"[ERROR] Dataset root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    zip_out.parent.mkdir(parents=True, exist_ok=True)

    jsonl_out = zip_out.parent / "candidates.jsonl"

    total_folders = 0
    total_candidates = 0
    skipped_empty = 0

    print(f"[pack_dataset] Scanning: {dataset_root}")
    print(f"[pack_dataset] ZIP  → {zip_out}")
    print(f"[pack_dataset] JSONL→ {jsonl_out}")
    print()

    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf, \
         open(jsonl_out, "w", encoding="utf-8") as jl:

        for folder_name, candidates in iter_leaf_folders(dataset_root):
            if not candidates:
                skipped_empty += 1
                print(f"  [SKIP] {folder_name}  (empty)")
                continue

            # Write the JSON array as-is into the ZIP
            # pruning.py's discover_folders handles both array and JSONL
            json_bytes = json.dumps(candidates, ensure_ascii=False, indent=None).encode("utf-8")
            arc_name = f"{folder_name}/candidates.json"
            zf.writestr(arc_name, json_bytes)

            # Append each candidate to the flat JSONL
            for item in candidates:
                jl.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_folders += 1
            total_candidates += len(candidates)
            print(f"  [OK]   {folder_name}  ({len(candidates):,} candidates)")

    zip_size_mb = zip_out.stat().st_size / (1024 * 1024)

    print()
    print("=" * 56)
    print(f"  Total folders written : {total_folders:,}")
    print(f"  Folders skipped (empty): {skipped_empty:,}")
    print(f"  Total candidates      : {total_candidates:,}")
    print(f"  Output ZIP            : {zip_out}  ({zip_size_mb:.1f} MB)")
    print(f"  Output JSONL          : {jsonl_out}")
    print("=" * 56)


if __name__ == "__main__":
    main()
