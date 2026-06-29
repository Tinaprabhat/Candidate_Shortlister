#!/usr/bin/env python3
"""
rank.py — RedRob ranking entry point (OFFLINE, no API calls).

Pipeline:
  L1 hard-reject → L1b profile-integrity → L1c skill-match → L1d bonus-match
  → L2 table-extract → L3 fuzzy-score → gate (top 50% + random 25% = 75%)
  → L4 semantic-score + donts penalty (baked in) → top 100
  → L5 FlashRank on top 50 (total_score = 0.4*l4_score + 0.3*l3 + 0.3*fr) → final top 100

Single-command:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

For a ZIP with folder hierarchy:
    python rank.py --zip ./data/candidates.zip --out ./submission.csv

PREREQUISITE: data/jd.json must already exist. Generate it once with:
    python -m src.jd_parser --jd ./data/job_description.pdf --out ./data/jd.json
"""

import csv
import json
import time
import logging
import argparse
import threading
from pathlib import Path
from typing import List

from src import constants as C
from src import utils, layers, pruning
from src.tracer import RunTracer

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
    if not survivors:
        return []

    t = time.time()
    survivors = layers.l1d_explicit_bonus(survivors, jd)
    tracer.record_cascade_step("L1d_explicit_bonus", len(survivors), len(survivors),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "avg_bonus_match": round(
                                       sum(c.get("l1d_bonus_match_ratio", 1.0) for c in survivors)
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

    # Top 100 by l4_combined_score (sorted; donts penalty already baked into l4_score by L4)
    top100 = candidates[:100]
    n_penalised = sum(1 for c in top100 if c.get("l4_donts_penalty", 0.0) > 0)
    logger.info(f"After L4+donts: {len(candidates)} scored → top {len(top100)} ({n_penalised} donts-penalised)")
    tracer.record_cascade_step("L4_donts_baked", len(candidates), len(top100),
                               notes={"donts_penalised": n_penalised})

    # L5 — FlashRank on top 50; total_score = 0.4*l4_score + 0.3*l3 + 0.3*flashrank
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

def _build_rows(top: List[dict]) -> List[dict]:
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
        profile = c.get("profile") or {}
        name = (
            c.get("name")
            or profile.get("name")
            or profile.get("full_name")
            or cid
        )
        records.append({
            # ── Identity & rank ───────────────────────────────────────────
            "rank":         row.get("rank"),
            "candidate_id": cid,
            "name":         name,
            "final_score":  row.get("score"),

            # ── Full candidate profile ────────────────────────────────────
            "profile":        profile,
            "education":      c.get("education") or [],
            "career_history": c.get("career_history") or c.get("work_experience") or [],
            "skills":         c.get("skills") or [],
            "certifications": c.get("certifications") or [],
            "languages":      c.get("languages") or [],
            "projects":       c.get("projects") or [],
            "publications":   c.get("publications") or c.get("research_papers") or [],
            "redrob_signals": c.get("redrob_signals") or {},

            # ── Skill match detail ────────────────────────────────────────
            "skill_match": {
                "matched_explicit_required": c.get("l1c_matched_explicit", []),
                "matched_all_required":      c.get("l1c_matched_required", []),
                "missing_required":          c.get("l1c_missing_required", []),
                "matched_explicit_bonus":    c.get("l1d_matched_bonus", []),
                "unmatched_explicit_bonus":  c.get("l1d_unmatched_bonus", []),
                "matched_all_bonus":         c.get("l1c_matched_bonus", []),
                "jd_req_score":              round(c.get("jd_req_score", 0.0), 4),
                "jd_req_results":            c.get("jd_req_results", {}),
            },

            # ── Per-layer scores ──────────────────────────────────────────
            "layer_scores": {
                "l1_score":              round(float(c.get("l1_score") or 0.5), 4),
                "l1_flags":              c.get("l1_flags") or [],
                "l1_status":             c.get("l1_status", "pass"),
                "l1b_penalty":           round(float(c.get("l1b_penalty") or 1.0), 4),
                "l1b_flags":             c.get("l1b_flags") or [],
                "l1b_status":            c.get("l1b_status", "pass"),
                "l1c_score":             round(float(c.get("l1c_score") or 0.0), 4),
                "l1d_bonus_match_ratio": round(float(c.get("l1d_bonus_match_ratio") or 1.0), 4),
                "l1d_bonus_penalty":     round(float(c.get("l1d_bonus_penalty") or 0.0), 4),
                "l3_score":              round(float(c.get("l3_score") or 0.0), 4),
                "l3_class":              c.get("l3_class", ""),
                "l4_work_relevance":     round(float(c.get("l4_work_relevance") or 0.0), 4),
                "l4_combined_score":     round(float(c.get("l4_combined_score") or 0.0), 4),
                "l5_donts_mult":         round(float(c.get("l5_donts_mult") or 1.0), 4),
                "l5_donts_score":        round(float(c.get("l5_donts_score") or 0.0), 4),
                "l5_flashrank_score":    round(float(c.get("l5_flashrank_score") or 0.0), 4),
                "l5_total_score":        round(float(c.get("l5_total_score") or 0.0), 4),
            },

            # ── L2 table — all computed columns ──────────────────────────
            "l2_table": c.get("table_row") or {},

            # ── Reasoning string from L3 ──────────────────────────────────
            "reasoning": c.get("l3_reasoning", ""),
        })

    out = {
        "run_id":       run_id,
        "jd_title":     jd.get("job_title", ""),
        "jd_location":  jd.get("location", ""),
        "jd_industry":  jd.get("industry", ""),
        "total_ranked": len(records),
        "candidates":   records,
    }
    path = _OUTPUT_DIR / f"ranked_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
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

    def process_fn(_, cands: List[dict]):
        scored = run_early_cascade(cands, jd, models, tracer)
        with lock:
            all_early_scored.extend(scored)

    if args.zip:
        t_zip = time.time()
        root = pruning.extract_zip(args.zip)
        folder_paths = pruning.discover_folders(root)
        tracer.record_timing("zip_extract_discover", round(time.time() - t_zip, 3))
        discovered = list(folder_paths.keys())

        original_prune = pruning.prune_folders

        def _traced_prune(names, jd_):
            kept_ = original_prune(names, jd_)
            kept_set = {n for n, _ in kept_}
            decisions = {n: ("KEEP" if n in kept_set else "DROP:pruned") for n in names}
            tracer.record_pruning(names, decisions, kept_)
            return kept_

        kept = _traced_prune(discovered, jd)
        pruning.dispatch_folders_staggered(
            kept, folder_paths, process_fn, stagger_sec=args.stagger, tracer=tracer
        )
    else:
        t_read = time.time()
        cands = utils.read_jsonl(args.candidates)
        tracer.record_timing("candidates_io", round(time.time() - t_read, 3))
        logger.info(f"Loaded {len(cands)} candidates from {args.candidates}")
        tracer.record_folder_load("flat", len(cands), len(cands))
        process_fn("flat", cands)

    if not all_early_scored:
        logger.error("No candidates survived early cascade (L1/L1b/L1c).")
        tracer.record_error("cascade", "No candidates survived early cascade")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # Hard pool cap before expensive L2/L3 layers (sort by l1c_score as proxy)
    if len(all_early_scored) > C.HARD_POOL_CAP:
        all_early_scored.sort(key=lambda c: c.get("l1c_score", 0), reverse=True)
        all_early_scored = all_early_scored[:C.HARD_POOL_CAP]
        logger.info(f"Capped pool to {C.HARD_POOL_CAP}")

    # Late cascade — L2 → L3 → gate → L4 → top200 → donts → top100 → L5
    top100 = run_late_cascade(all_early_scored, jd, tracer)

    if not top100:
        logger.error("Late cascade produced no candidates.")
        tracer.record_error("cascade", "Late cascade returned empty")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # Build output
    rows = _build_rows(top100)
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
    logger.info(f"Wrote {len(rows)} ranked candidates → {args.out}")
    logger.info(f"Ranked JSON → {json_path}")
    logger.info(f"Run trace   → {trace_path}")
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")


def _try_load_flashrank():
    try:
        __import__("flashrank")
        return utils.load_flashrank()
    except Exception as exc:
        logger.warning(f"FlashRank not available ({exc}); L5 will degrade gracefully")
        return None


if __name__ == "__main__":
    main()
