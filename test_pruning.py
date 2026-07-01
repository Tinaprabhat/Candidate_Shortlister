#!/usr/bin/env python3
"""
test_pruning.py — Run ONLY preprocessing + pruning, skip L1–L5 scoring.

Steps:
  1. candidates.jsonl  →  data/preprocessing/Dataset/  (6-level tree)
  2. discover_folders()  finds all bucket keys
  3. prune_folders()     applies Phase-1a/b/c/d/e + Phase-2 token-overlap scoring
  4. Prints full tree with KEEP/DROP decisions + per-step timing breakdown
  5. Writes a JSON log to logs/pruning_test_<timestamp>.json

Usage:
    python test_pruning.py
    python test_pruning.py --candidates ./data/candidates.jsonl --jd ./data/jd.json
"""

import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("test_pruning")

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def _ms(seconds: float) -> str:
    """Format seconds as milliseconds or seconds depending on magnitude."""
    ms = seconds * 1000
    if ms >= 1000:
        return f"{seconds:.3f}s"
    return f"{ms:.1f}ms"


def main():
    ap = argparse.ArgumentParser(description="Test pruning only (no scoring)")
    ap.add_argument("--candidates", type=Path, default=DATA_DIR / "candidates.jsonl")
    ap.add_argument("--jd",         type=Path, default=DATA_DIR / "jd.json")
    args = ap.parse_args()

    if not args.candidates.exists():
        ap.error(f"candidates.jsonl not found: {args.candidates}")
    if not args.jd.exists():
        ap.error(f"jd.json not found: {args.jd}")

    timings = {}   # step → elapsed seconds
    t0 = time.time()

    # ── Load JD ──────────────────────────────────────────────────────────────
    t = time.time()
    from pipeline import utils
    jd = utils.load_jd_json(args.jd)
    timings["jd_load"] = time.time() - t
    logger.info(
        f"JD: title='{jd.get('job_title','')}' | "
        f"location='{jd.get('location','')}' | "
        f"exp={jd.get('experience_min',0)}-{jd.get('experience_max',99)} yrs | "
        f"{len(jd.get('explicit_required',[]))} required skills"
    )

    # ── Step 1: Preprocessing (candidates.jsonl → Dataset/ tree) ─────────────
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "pipeline_1",
        PROJECT_ROOT / "data" / "preprocessing" / "pipeline_1.py",
    )
    pipeline_1 = _ilu.module_from_spec(spec)
    spec.loader.exec_module(pipeline_1)

    dataset_dir = PROJECT_ROOT / "data" / "preprocessing" / "Dataset"
    logger.info(f"[Step 1] Preprocessing {args.candidates} → {dataset_dir} ...")
    t = time.time()
    pipeline_1.run_preprocessing(args.candidates, dataset_dir)
    timings["preprocessing"] = time.time() - t
    logger.info(f"[Step 1] Done in {_ms(timings['preprocessing'])}")

    # ── Step 2: Discover folders ──────────────────────────────────────────────
    from pipeline import pruning
    logger.info("[Step 2] Discovering folders ...")
    t = time.time()
    folder_paths = pruning.discover_folders(dataset_dir)
    timings["discover_folders"] = time.time() - t
    discovered = list(folder_paths.keys())
    logger.info(f"[Step 2] Discovered {len(discovered)} bucket(s) in {_ms(timings['discover_folders'])}")

    if not discovered:
        logger.error("No folders discovered — check that preprocessing produced a non-empty Dataset/")
        return

    # ── Step 3: Prune ─────────────────────────────────────────────────────────
    logger.info("[Step 3] Pruning ...")
    t = time.time()
    kept = pruning.prune_folders(discovered, jd)
    timings["prune_folders"] = time.time() - t
    phase_timing = pruning.last_prune_timing.copy()   # per-phase breakdown

    kept_set = {name for name, _ in kept}
    dropped  = [n for n in discovered if n not in kept_set]
    decisions = {n: "DROP" for n in dropped}
    decisions.update({n: f"KEEP (score={s:.3f})" for n, s in kept})

    # ── Step 4: Count candidates per kept folder ──────────────────────────────
    logger.info("[Step 4] Counting candidates in kept folders ...")
    t = time.time()
    total_candidates = 0
    folder_counts = {}
    for name, _ in kept:
        data_path = folder_paths[name]
        try:
            if data_path.suffix.lower() == ".jsonl":
                lines = [l for l in data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
                count = len(lines)
            else:
                data = json.loads(data_path.read_text(encoding="utf-8"))
                count = len(data) if isinstance(data, list) else 1
        except Exception as e:
            count = -1
            logger.warning(f"Could not count {data_path}: {e}")
        folder_counts[name] = count
        total_candidates += max(count, 0)
    timings["count_candidates"] = time.time() - t

    timings["total"] = time.time() - t0

    # ── Print summary ─────────────────────────────────────────────────────────
    W = 72
    print("\n" + "=" * W)
    print(f"  PRUNING SUMMARY  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * W)
    print(f"  Folders discovered : {len(discovered)}")
    print(f"  Kept               : {len(kept)}")
    print(f"  Dropped            : {len(dropped)}")
    print()

    print("  KEPT folders:")
    for name, score in kept:
        cnt = folder_counts.get(name, -1)
        print(f"    [KEEP  score={score:.3f}]  {name}  ({cnt} candidates)")
    if not kept:
        print("    (none)")
    print()

    print("  DROPPED folders:")
    for name in sorted(dropped):
        print(f"    [DROP]  {name}")
    if not dropped:
        print("    (none)")
    print()

    print(f"  Total candidates entering pipeline: {total_candidates}")
    print()

    # ── Timing breakdown ──────────────────────────────────────────────────────
    print("─" * W)
    print("  TIMING BREAKDOWN")
    print("─" * W)

    steps = [
        ("JD load",               timings["jd_load"]),
        ("Preprocessing (pass 1-3)", timings["preprocessing"]),
        ("discover_folders()",    timings["discover_folders"]),
        ("prune_folders() total", timings["prune_folders"]),
        ("Count candidates",      timings["count_candidates"]),
    ]
    for label, secs in steps:
        bar_width = max(1, int(secs / timings["total"] * 30))
        bar = "█" * bar_width
        print(f"  {label:<28}  {_ms(secs):>10}   {bar}")

    print()
    print("  prune_folders() internal breakdown:")
    sub_steps = [
        ("JD classify",                  phase_timing.get("jd_classify_ms", 0) / 1000),
        ("Phase 1a  coding-type check",  phase_timing.get("phase1a_ms", 0) / 1000),
        ("Phase 1b  inactive check",     phase_timing.get("phase1b_ms", 0) / 1000),
        ("Phase 1c  location check",     phase_timing.get("phase1c_ms", 0) / 1000),
        ("Phase 1d  experience check",   phase_timing.get("phase1d_ms", 0) / 1000),
        ("Phase 1e  role check",         phase_timing.get("phase1e_ms", 0) / 1000),
        ("Phase 2a  build JD signal",    phase_timing.get("phase2_signal_ms", 0) / 1000),
        ("Phase 2b  score_folder×N",     phase_timing.get("phase2_score_ms", 0) / 1000),
        ("Phase 2c  sort",               phase_timing.get("phase2_sort_ms", 0) / 1000),
    ]
    prune_total = timings["prune_folders"] or 1
    for label, secs in sub_steps:
        pct = secs / prune_total * 100
        drops_key = f"phase{label[6:8].strip().lower()}_drops" if "Phase" in label else ""
        drop_str = f"  ({phase_timing.get(drops_key, 0)} dropped)" if drops_key and "1" in label else ""
        print(f"    {label:<30}  {_ms(secs):>10}  {pct:5.1f}%{drop_str}")

    print()
    print(f"  TOTAL                              {_ms(timings['total']):>10}")
    print("=" * W + "\n")

    # ── Write log ─────────────────────────────────────────────────────────────
    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"pruning_test_{run_id}.json"
    log_data = {
        "run_id":             run_id,
        "jd_title":           jd.get("job_title", ""),
        "jd_location":        jd.get("location", ""),
        "jd_exp_min":         jd.get("experience_min", 0),
        "jd_exp_max":         jd.get("experience_max", 99),
        "jd_required_skills": jd.get("explicit_required", []),
        "timing_seconds": {k: round(v, 4) for k, v in timings.items()},
        "timing_ms_breakdown": phase_timing,
        "discovered_count":   len(discovered),
        "kept_count":         len(kept),
        "dropped_count":      len(dropped),
        "total_candidates":   total_candidates,
        "kept_folders":       [
            {"name": n, "score": round(s, 4), "candidates": folder_counts.get(n, -1)}
            for n, s in kept
        ],
        "dropped_folders":    sorted(dropped),
        "decisions":          decisions,
    }
    log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Log written → {log_path}")


if __name__ == "__main__":
    main()
