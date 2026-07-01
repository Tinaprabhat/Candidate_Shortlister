#!/usr/bin/env python3
"""
rank.py — RedRob ranking entry point (OFFLINE, no API calls).

Pipeline:
  L1 hard-reject → L1b profile-integrity → L1c skill-match
  → L2 table-extract → L3 fuzzy-score → gate (top 50% + random 25% = 75%)
  → L4 semantic-score → top 200 → donts penalty → top 100
  → L5 FlashRank on top 50 (total_score = prev + flashrank) → final top 100

Single-command:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

For a ZIP with folder hierarchy:
    python rank.py --zip ./data/candidates.zip --out ./submission.csv

PREREQUISITE: data/jd.json must already exist. Generate it once with:
    python -m pipeline.jd_parser --jd ./data/job_description.pdf --out ./data/jd.json
"""

import csv
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import List

from pipeline import constants as C
from pipeline import utils, layers, pruning
from pipeline.tracer import RunTracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rank")

_OUTPUT_DIR = Path(__file__).parent / "output"


# ── Early cascade (per-folder) ────────────────────────────────────────────────

def run_early_cascade(
    candidates: List[dict], jd: dict, models: dict, tracer: RunTracer
) -> List[dict]:
    """
    L1 hard-reject → L1b profile-integrity → L1c skill-match.
    Returns scored survivors; the global 75% gate is applied in main() afterwards.
    """
    before = len(candidates)

    t = time.time()
    survivors = layers.l1_hard_reject(candidates, models["fraud_kb"])
    tracer.record_cascade_step("L1_hard_reject", before, len(survivors),
                               notes={"elapsed_s": round(time.time() - t, 3)})
    if not survivors:
        return []

    before = len(survivors)
    t = time.time()
    survivors = layers.l1b_profile_integrity(survivors)
    tracer.record_cascade_step("L1b_profile_integrity", before, len(survivors),
                               notes={"elapsed_s": round(time.time() - t, 3)})
    if not survivors:
        return []

    before = len(survivors)
    t = time.time()
    survivors = layers.l1c_skill_match(survivors, jd)
    tracer.record_cascade_step("L1c_skill_match", before, len(survivors),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "avg_score": round(
                                       sum(c.get("l1c_score", 0) for c in survivors)
                                       / max(len(survivors), 1), 3
                                   ),
                               })
    return survivors


# ── Late cascade (global, after early cascade) ───────────────────────────────

def run_late_cascade(
    candidates: List[dict], jd: dict, tracer: RunTracer
) -> List[dict]:
    """
    L2 table-extract → L3 fuzzy-score → gate (75%)
    → L4 semantic-score → top 200
    → donts penalty → top 100
    → L5 FlashRank top 50 (total_score = donts_score + flashrank) → top 100
    """
    # L2 — build 28-column table_row per candidate
    t = time.time()
    candidates = layers.l2_table_extract(candidates, jd)
    tracer.record_cascade_step("L2_table_extract", len(candidates), len(candidates),
                               notes={"elapsed_s": round(time.time() - t, 3)})

    # L3 — Sugeno fuzzy score (no knockouts — all candidates scored)
    t = time.time()
    candidates = layers.l3_fuzzy_score(candidates, jd)
    tracer.record_cascade_step("L3_fuzzy_score", len(candidates), len(candidates),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "avg_score": round(
                                       sum(c.get("l3_score", 0) for c in candidates)
                                       / max(len(candidates), 1), 3
                                   ),
                               })

    # Gate — top 50% + random 25% by l3_score = 75%
    before_gate = len(candidates)
    t = time.time()
    candidates = layers.l3_gate(candidates)
    tracer.record_cascade_step("L3_gate", before_gate, len(candidates),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "top_pct":    C.GATE_TOP_FRACTION,
                                   "random_pct": C.GATE_RANDOM_FRACTION,
                               })
    logger.info(f"L3 gate: {before_gate} → {len(candidates)} passed (75%)")

    # L4 — semantic work relevance (returns all sorted by l4_combined_score)
    # Cap input to top-N by L3 score; evidence shows all final candidates have L3 >> rescue pool
    if len(candidates) > C.L4_INPUT_CAP:
        candidates = candidates[:C.L4_INPUT_CAP]
        logger.info(f"L4 input capped at {C.L4_INPUT_CAP} (sorted by L3 score)")
    t = time.time()
    candidates = layers.l4_semantic_work(candidates, jd)
    tracer.record_cascade_step("L4_semantic_work", len(candidates), len(candidates),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "avg_relevance": round(
                                       sum(c.get("l4_work_relevance", 0) for c in candidates)
                                       / max(len(candidates), 1), 3
                                   ),
                               })

    # Top 200 by l4_combined_score (list already sorted)
    top200 = candidates[:200]
    logger.info(f"After L4: {len(candidates)} scored → top {len(top200)} forwarded to donts layer")

    # Donts penalty (high) → top 100
    t = time.time()
    top100 = layers.donts_penalty_layer(top200, jd)
    n_penalised = sum(1 for c in top100 if c.get("l5_donts_mult", 1.0) < 1.0)
    tracer.record_cascade_step("Donts_penalty", len(top200), len(top100),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "donts_penalised": n_penalised,
                               })

    # L5 — FlashRank on top 50; total_score = donts_score + flashrank
    t = time.time()
    top100 = layers.l5_flashrank_rerank(top100, jd)
    tracer.record_cascade_step("L5_flashrank", len(top100), len(top100),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "avg_total": round(
                                       sum(c.get("l5_total_score", 0) for c in top100)
                                       / max(len(top100), 1), 3
                                   ),
                               })

    return top100


# ── Output helpers ────────────────────────────────────────────────────────────

def _build_rows(top: List[dict], jd: dict) -> List[dict]:
    rows = []
    prev = float("inf")
    for rank_pos, c in enumerate(top, start=1):
        score = min(float(c.get("l5_total_score", 0.0)), prev)
        prev  = score
        rows.append({
            "candidate_id": str(c.get("candidate_id", c.get("id", f"UNKNOWN_{rank_pos}"))),
            "rank":         rank_pos,
            "score":        round(score, 6),
            "reasoning":    c.get("l3_reasoning", ""),
        })
    return rows


def _write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)


def _write_ranked_json(top: List[dict], rows: List[dict], jd: dict, run_id: str) -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_map = {r["candidate_id"]: r for r in rows}
    records = []
    for c in top:
        cid = str(c.get("candidate_id", c.get("id", "")))
        row = score_map.get(cid, {})
        records.append({
            "rank":         row.get("rank"),
            "candidate_id": cid,
            "final_score":  row.get("score"),
            "reasoning":    c.get("l3_reasoning", ""),
            "score_breakdown": {
                "l1c_skill_match":       round(c.get("l1c_score", 0.0), 4),
                "l1c_matched_explicit":  c.get("l1c_matched_explicit", []),
                "l1c_matched_required":  c.get("l1c_matched_required", []),
                "l1c_missing_required":  c.get("l1c_missing_required", []),
                "l1b_integrity_penalty": round(c.get("l1b_penalty", 1.0), 4),
                "l1b_flags":            c.get("l1b_flags", []),
                "l3_fuzzy_score":       round(c.get("l3_score", 0.0), 4),
                "l3_class":             c.get("l3_class", ""),
                "l4_work_relevance":    round(c.get("l4_work_relevance", 0.0), 4),
                "l4_combined_score":    round(c.get("l4_combined_score", 0.0), 4),
                "l5_donts_mult":        round(c.get("l5_donts_mult", 1.0), 4),
                "l5_donts_score":       round(c.get("l5_donts_score", 0.0), 4),
                "l5_flashrank_score":   round(c.get("l5_flashrank_score", 0.0), 4),
                "l5_total_score":       round(c.get("l5_total_score", 0.0), 4),
            },
            "experience_years": utils.get_total_experience_years(c),
            "skills_snippet":   utils.get_skills_text(c)[:200],
        })
    out = {
        "run_id":        run_id,
        "jd_title":      jd.get("job_title", ""),
        "jd_location":   jd.get("location", ""),
        "jd_industry":   jd.get("industry", ""),
        "total_ranked":  len(records),
        "candidates":    records,
    }
    path = _OUTPUT_DIR / f"ranked_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Ranked JSON → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    tracer = RunTracer()

    ap = argparse.ArgumentParser(description="RedRob candidate ranking (offline)")
    ap.add_argument("--candidates", type=Path, help="Path to flat candidates.jsonl")
    ap.add_argument("--zip",        type=Path, help="Path to candidates ZIP (folder hierarchy)")
    ap.add_argument("--jd",         type=Path, default=C.JD_JSON_PATH)
    ap.add_argument("--out",        type=Path, default=Path("./submission.csv"))
    ap.add_argument("--stagger",    type=int,  default=C.FOLDER_DISPATCH_STAGGER_SEC)
    args = ap.parse_args()

    if not args.candidates and not args.zip:
        ap.error("Provide either --candidates <jsonl> or --zip <zip>")

    # Load JD
    t_jd = time.time()
    jd = utils.load_jd_json(args.jd)
    tracer.record_timing("jd_load", round(time.time() - t_jd, 3))
    logger.info(
        f"Loaded jd.json: title='{jd.get('job_title','')}' | "
        f"location='{jd.get('location','')}' | industry='{jd.get('industry','')}' | "
        f"exp={jd.get('experience_min',0)}-{jd.get('experience_max',99)} yrs | "
        f"{len(jd.get('explicit_required',[]))} required skills | "
        f"{len(jd.get('donts',[]))} donts"
    )
    tracer.set_jd(jd)

    # Decompress models on first run only — no-op once models/decompressed/ is populated
    t_decomp = time.time()
    utils.ensure_models_decompressed()
    tracer.record_timing("models_decompress", round(time.time() - t_decomp, 3))

    # Load models
    logger.info("Loading models...")
    t_models = time.time()
    models = {
        "fraud_kb":  utils.load_fraud_kb(),
        "flashrank": _try_load_flashrank(),
        "embedder":    utils.load_sentence_transformer()
    }
    tracer.record_timing("models_load", round(time.time() - t_models, 3))

    # Gather candidates — early cascade (L1 → L1b → L1c) per folder
    all_early_scored: List[dict] = []
    lock = threading.Lock()

    def process_fn(folder_name: str, cands: List[dict]):
        scored = run_early_cascade(cands, jd, models, tracer)
        with lock:
            all_early_scored.extend(scored)

    if args.zip:
        t_zip = time.time()
        root = pruning.extract_zip(args.zip)
        folder_paths = pruning.discover_folders(root)
        tracer.record_timing("zip_extract_discover", round(time.time() - t_zip, 3))
        discovered = list(folder_paths.keys())

        import types
        original_prune = pruning.prune_folders

        def _traced_prune(names, jd_):
            kept_ = original_prune(names, jd_)
            kept_set = {n for n, _ in kept_}
            decisions = {n: ("KEEP" if n in kept_set else "DROP:pruned") for n in names}
            tracer.record_pruning(names, decisions, kept_)
            return kept_

        t_prune = time.time()
        kept = _traced_prune(discovered, jd)
        tracer.record_timing("prune_folders", round(time.time() - t_prune, 3))
        t_dispatch = time.time()
        pruning.dispatch_folders_staggered(
            kept, folder_paths, process_fn, stagger_sec=args.stagger, tracer=tracer
        )
        tracer.record_timing("early_cascade_dispatch", round(time.time() - t_dispatch, 3))
    else:
        # Step 1: Always run preprocessing — clean + fabrication flags + build Dataset/ tree
        import importlib.util as _ilu
        _pipeline_spec = _ilu.spec_from_file_location(
            "pipeline_1",
            Path(__file__).parent / "data" / "preprocessing" / "pipeline_1.py",
        )
        _pipeline = _ilu.module_from_spec(_pipeline_spec)
        _pipeline_spec.loader.exec_module(_pipeline)

        dataset_dir = Path(__file__).parent / "data" / "preprocessing" / "Dataset"
        t_pre = time.time()
        logger.info(f"[preprocess] Starting pipeline_1 preprocessing of {args.candidates} ...")
        _pipeline.run_preprocessing(args.candidates, dataset_dir)
        tracer.record_timing("preprocessing", round(time.time() - t_pre, 3))
        logger.info(f"[preprocess] Done in {time.time() - t_pre:.1f}s → {dataset_dir}")

        # Step 2: Discover folders from the freshly-built tree
        t_disc = time.time()
        folder_paths = pruning.discover_folders(dataset_dir)
        tracer.record_timing("folder_discover", round(time.time() - t_disc, 3))
        discovered = list(folder_paths.keys())
        logger.info(f"Discovered {len(discovered)} folder buckets from preprocessed tree")

        # Step 3: Prune + dispatch (same path as ZIP)
        import types as _types
        original_prune = pruning.prune_folders

        def _traced_prune(names, jd_):
            kept_ = original_prune(names, jd_)
            kept_set = {n for n, _ in kept_}
            decisions = {n: ("KEEP" if n in kept_set else "DROP:pruned") for n in names}
            tracer.record_pruning(names, decisions, kept_)
            return kept_

        t_prune = time.time()
        kept = _traced_prune(discovered, jd)
        tracer.record_timing("prune_folders", round(time.time() - t_prune, 3))

        t_dispatch = time.time()
        pruning.dispatch_folders_staggered(
            kept, folder_paths, process_fn, stagger_sec=args.stagger, tracer=tracer
        )
        tracer.record_timing("early_cascade_dispatch", round(time.time() - t_dispatch, 3))

    if not all_early_scored:
        logger.error("No candidates survived early cascade (L1/L1b/L1c).")
        tracer.record_error("cascade", "No candidates survived early cascade")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # Hard pool cap before expensive L2/L3 layers (sort by l1c_score as proxy)
    t_cap = time.time()
    if len(all_early_scored) > C.HARD_POOL_CAP:
        all_early_scored.sort(key=lambda c: c.get("l1c_score", 0), reverse=True)
        all_early_scored = all_early_scored[:C.HARD_POOL_CAP]
        logger.info(f"Capped pool to {C.HARD_POOL_CAP}")
    tracer.record_timing("pool_cap_sort", round(time.time() - t_cap, 3))

    # Late cascade — L2 → L3 → gate → L4 → top200 → donts → top100 → L5
    t_late = time.time()
    top100 = run_late_cascade(all_early_scored, jd, tracer)
    tracer.record_timing("late_cascade_total", round(time.time() - t_late, 3))

    if not top100:
        logger.error("Late cascade produced no candidates.")
        tracer.record_error("cascade", "Late cascade returned empty")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # Build output
    rows = _build_rows(top100, jd)
    t_write = time.time()
    json_path = _write_ranked_json(top100, rows, jd, tracer.run_id)
    _write_csv(args.out, rows)
    tracer.record_timing("output_write", round(time.time() - t_write, 3))

    utils.cleanup()
    elapsed = time.time() - t0
    trace_path = tracer.finish(
        args.out,
        output_json=json_path,
        rows_written=len(rows),
        top_scores=[r["score"] for r in rows[:10]],
        elapsed=elapsed,
    )
    _print_timing_summary(tracer, elapsed)
    logger.info(f"Wrote {len(rows)} ranked candidates → {args.out}")
    logger.info(f"Ranked JSON → {json_path}")
    logger.info(f"Run trace   → {trace_path}")
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")


def _print_timing_summary(tracer: RunTracer, total_elapsed: float) -> None:
    """Print a human-readable timing breakdown to stdout."""
    trace = tracer.as_dict()

    timings: dict = {t["name"]: t["elapsed_s"] for t in trace.get("setup_timings", [])}

    cascade_steps: dict = {}
    for step in trace.get("cascade", {}).get("steps", []):
        name = step["layer"]
        elapsed = step.get("notes", {}).get("elapsed_s", 0.0)
        cascade_steps[name] = cascade_steps.get(name, 0.0) + elapsed

    def _fmt(s: float) -> str:
        if s >= 60:
            return f"{s/60:.2f}min"
        if s >= 1:
            return f"{s:.3f}s"
        return f"{s*1000:.1f}ms"

    def _bar(s: float, total: float, width: int = 28) -> str:
        frac = min(s / total, 1.0) if total > 0 else 0
        filled = max(1, round(frac * width))
        return "█" * filled

    W = 76
    print("\n" + "=" * W)
    print("  RANK.PY TIMING SUMMARY")
    print("=" * W)

    sections = [
        ("── Setup", [
            ("JD load",               timings.get("jd_load", 0)),
            ("Models load",            timings.get("models_load", 0)),
        ]),
        ("── Preprocessing & pruning", [
            ("Preprocessing (3-pass)", timings.get("preprocessing", 0)),
            ("Folder discover",        timings.get("folder_discover", 0)),
            ("prune_folders()",        timings.get("prune_folders", 0)),
        ]),
        ("── Early cascade (L1/L1b/L1c) per folder", [
            ("Dispatch + early cascade", timings.get("early_cascade_dispatch", 0)),
            ("  L1  hard reject",         cascade_steps.get("L1_hard_reject", 0)),
            ("  L1b profile integrity",   cascade_steps.get("L1b_profile_integrity", 0)),
            ("  L1c skill match",         cascade_steps.get("L1c_skill_match", 0)),
        ]),
        ("── Late cascade (global)", [
            ("Late cascade total",     timings.get("late_cascade_total", 0)),
            ("  L2  table extract",    cascade_steps.get("L2_table_extract", 0)),
            ("  L3  fuzzy score",      cascade_steps.get("L3_fuzzy_score", 0)),
            ("  L3  gate",             cascade_steps.get("L3_gate", 0)),
            ("  L4  semantic work",    cascade_steps.get("L4_semantic_work", 0)),
            ("  Donts penalty",        cascade_steps.get("Donts_penalty", 0)),
            ("  L5  flashrank",        cascade_steps.get("L5_flashrank", 0)),
        ]),
        ("── Output", [
            ("Pool cap sort",          timings.get("pool_cap_sort", 0)),
            ("Output write",           timings.get("output_write", 0)),
        ]),
    ]

    for section_title, rows in sections:
        print(f"\n  {section_title}")
        for label, secs in rows:
            indent = "  " if label.startswith("  ") else ""
            bar = _bar(secs, total_elapsed)
            print(f"  {indent}{label:<34}  {_fmt(secs):>10}   {bar}")

    print()
    print(f"  {'TOTAL':<34}  {_fmt(total_elapsed):>10}")

    cascade_totals = trace.get("cascade", {})
    print()
    print(f"  Candidates in  : {cascade_totals.get('total_input', '?')}")
    print(f"  Candidates out : {cascade_totals.get('total_output', '?')}")
    print(f"  Final ranked   : {trace.get('output', {}).get('candidates_ranked', '?')}")

    top_scores = trace.get("output", {}).get("top_scores", [])
    if top_scores:
        scores_str = "  ".join(f"{s:.4f}" for s in top_scores[:5])
        print(f"  Top-5 scores   : {scores_str}")

    print("=" * W + "\n")


def _try_load_flashrank():
    try:
        import flashrank  # noqa
        return utils.load_flashrank()
    except Exception as exc:
        logger.warning(f"FlashRank not available ({exc}); L5 will degrade gracefully")
        return None


if __name__ == "__main__":
    main()
