#!/usr/bin/env python3
"""
rank.py — RedRob ranking entry point (OFFLINE, no API calls).

Reads candidates + pre-computed jd.json, runs the 6-layer cascade + FIS,
writes top-100 submission CSV, and saves a full run trace to logs/.

Single-command (Stage 3 reproducible):
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

For a ZIP with folder hierarchy (Streamlit path uses this internally):
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
from src import utils, layers, fis, pruning
from src.tracer import RunTracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rank")


def run_cascade_on_pool(
    candidates: List[dict], jd: dict, models: dict, tracer: RunTracer
) -> List[dict]:
    """Run L1–L6 on a pool of candidates. Returns scored survivors (post-gate)."""
    # L1 — hard reject
    before = len(candidates)
    t = time.time()
    survivors = layers.l1_hard_reject(candidates, models["fraud_kb"])
    elapsed_l1 = time.time() - t
    logger.info(f"L1 elapsed: {elapsed_l1:.3f}s")
    tracer.record_cascade_step("L1_hard_reject", before, len(survivors),
                               notes={"elapsed_s": round(elapsed_l1, 3)})
    if not survivors:
        return []

    # L2 — bi-encoder + gate
    before = len(survivors)
    t = time.time()
    survivors = layers.l2_bi_encoder(survivors, jd, models["st"])
    survivors = layers.l2_gate(survivors)
    elapsed_l2 = time.time() - t
    logger.info(f"L2 elapsed: {elapsed_l2:.3f}s")
    tracer.record_cascade_step("L2_bi_encoder_gate", before, len(survivors),
                               notes={"elapsed_s": round(elapsed_l2, 3)})
    if not survivors:
        return []

    # L3 — seniority soft penalty
    t = time.time()
    survivors = layers.l3_seniority(survivors, jd)
    elapsed_l3 = time.time() - t
    logger.info(f"L3 elapsed: {elapsed_l3:.3f}s")
    n_penalized_l3 = sum(1 for c in survivors if c.get("l3_penalty", 1.0) < 1.0)
    tracer.record_cascade_step("L3_seniority", len(survivors), len(survivors),
                               notes={"penalized": n_penalized_l3,
                                      "penalty_factor": C.SENIORITY_PENALTY,
                                      "elapsed_s": round(elapsed_l3, 3)})

    # L4 — semantic work-to-JD relevance
    t = time.time()
    survivors = layers.l4_semantic_work(survivors, jd, models["st"])
    elapsed_l4 = time.time() - t
    logger.info(f"L4 elapsed: {elapsed_l4:.3f}s")
    n_with_desc = sum(1 for c in survivors if c.get("l4_score", 0.0) > 0)
    tracer.record_cascade_step("L4_semantic_work", len(survivors), len(survivors),
                               notes={"with_descriptions": n_with_desc,
                                      "no_descriptions": len(survivors) - n_with_desc,
                                      "elapsed_s": round(elapsed_l4, 3)})

    # L6 — behavioral signals
    t = time.time()
    layers.l6_behavioral(survivors, jd)
    elapsed_l6 = time.time() - t
    logger.info(f"L6 elapsed: {elapsed_l6:.3f}s")
    tracer.record_cascade_step("L6_behavioral", len(survivors), len(survivors),
                               notes={"scored": len(survivors),
                                      "elapsed_s": round(elapsed_l6, 3)})

    return survivors


def main():
    t0 = time.time()
    tracer = RunTracer()

    ap = argparse.ArgumentParser(description="RedRob candidate ranking (offline)")
    ap.add_argument("--candidates", type=Path, help="Path to flat candidates.jsonl")
    ap.add_argument("--zip", type=Path, help="Path to candidates ZIP (folder hierarchy)")
    ap.add_argument("--jd", type=Path, default=C.JD_JSON_PATH, help="Path to pre-computed jd.json")
    ap.add_argument("--out", type=Path, default=Path("./submission.csv"))
    ap.add_argument(
        "--stagger", type=int, default=C.FOLDER_DISPATCH_STAGGER_SEC,
        help="Seconds between folder dispatch (set 0 to disable for testing)",
    )
    args = ap.parse_args()

    if not args.candidates and not args.zip:
        ap.error("Provide either --candidates <jsonl> or --zip <zip>")

    # ── Load jd.json (pre-computed, offline) ──────────────────────────────────
    jd = utils.load_jd_json(args.jd)
    logger.info(
        f"Loaded jd.json: title='{jd.get('job_title','')}', "
        f"location='{jd.get('location','')}', "
        f"exp={jd.get('experience_min',0)}-{jd.get('experience_max',99)}yrs, "
        f"{len(jd.get('explicit_required', []))} required skills"
    )
    tracer.set_jd(jd)

    # ── Load models (all from disk, offline) ──────────────────────────────────
    logger.info("Loading models...")
    models = {
        "st": utils.load_sentence_transformer(),
        "fraud_kb": utils.load_fraud_kb(),
        "flashrank": utils.load_flashrank() if _flashrank_available() else None,
    }

    # ── Gather candidate pools ────────────────────────────────────────────────
    all_scored: List[dict] = []
    lock = threading.Lock()
    folder_decisions: dict = {}   # populated by pruning, used for tracer

    def process_fn(folder_name: str, cands: List[dict]):
        scored = run_cascade_on_pool(cands, jd, models, tracer)
        with lock:
            all_scored.extend(scored)

    if args.zip:
        root = pruning.extract_zip(args.zip)
        folder_paths = pruning.discover_folders(root)
        discovered = list(folder_paths.keys())

        # Monkey-patch prune_folders to capture decisions for the tracer
        import types

        original_prune = pruning.prune_folders

        def _traced_prune(names, jd_):
            kept_ = original_prune(names, jd_)
            # Reconstruct decisions from kept set for tracer
            kept_set = {n for n, _ in kept_}
            for n in names:
                if n not in kept_set:
                    folder_decisions[n] = "DROP:pruned"
                else:
                    folder_decisions[n] = "KEEP"
            return kept_

        kept = _traced_prune(discovered, jd)
        tracer.record_pruning(discovered, folder_decisions, kept)

        pruning.dispatch_folders_staggered(
            kept, folder_paths, process_fn, stagger_sec=args.stagger, tracer=tracer
        )
    else:
        cands = utils.read_jsonl(args.candidates)
        logger.info(f"Loaded {len(cands)} candidates from {args.candidates}")
        tracer.record_folder_load("flat", len(cands), len(cands))
        process_fn("flat", cands)

    if not all_scored:
        logger.error("No candidates survived the cascade. Writing empty submission.")
        tracer.record_error("cascade", "No candidates survived all layers")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # ── Hard cap before final ranking ─────────────────────────────────────────
    if len(all_scored) > C.HARD_POOL_CAP:
        all_scored.sort(key=lambda c: c.get("l2_score", 0), reverse=True)
        all_scored = all_scored[:C.HARD_POOL_CAP]
        logger.info(f"Capped pool to {C.HARD_POOL_CAP}")

    # ── L7 — FIS + rank + FlashRank polish ────────────────────────────────────
    t = time.time()
    all_scored = fis.run_fis(all_scored)
    logger.info(f"L7 FIS elapsed: {time.time()-t:.3f}s")

    t = time.time()
    ranked = fis.rank_candidates(all_scored, models["fraud_kb"])
    logger.info(f"L7 rank elapsed: {time.time()-t:.3f}s")

    flashrank_ran = models["flashrank"] is not None
    t = time.time()
    ranked = fis.flashrank_polish(ranked, jd, models["flashrank"], models["fraud_kb"])
    logger.info(f"L7 FlashRank polish elapsed: {time.time()-t:.3f}s")

    top = ranked[:C.OUTPUT_TOP_N]
    fis_scores = [c.get("fis_score", 0.0) for c in top]
    tracer.record_l7(
        fis_input=len(all_scored),
        fis_output=len(ranked),
        flashrank_ran=flashrank_ran,
        flashrank_top_n=min(C.FLASHRANK_TOP_N, len(ranked)) if flashrank_ran else 0,
        score_range=(min(fis_scores, default=0.0), max(fis_scores, default=0.0)),
    )

    # ── Build output rows ─────────────────────────────────────────────────────
    rows = []
    prev_score = 1.0
    for rank_pos, c in enumerate(top, start=1):
        score = c.get("fis_score", 0.0)
        score = min(score, prev_score)  # enforce monotonic non-increasing
        prev_score = score
        rows.append({
            "candidate_id": str(c.get("candidate_id", c.get("id", f"UNKNOWN_{rank_pos}"))),
            "rank": rank_pos,
            "score": round(score, 6),
            "reasoning": fis.generate_reasoning(c, jd),
        })

    _write_csv(args.out, rows)

    # ── Save detailed ranked JSON to output/ ──────────────────────────────────
    json_path = _write_ranked_json(top, rows, jd, tracer.run_id)

    utils.cleanup()

    elapsed = time.time() - t0
    top_scores = [r["score"] for r in rows[:10]]
    trace_path = tracer.finish(
        args.out,
        output_json=json_path,
        rows_written=len(rows),
        top_scores=top_scores,
        elapsed=elapsed,
    )

    logger.info(f"Wrote {len(rows)} ranked candidates → {args.out}")
    logger.info(f"Run trace → {trace_path}")
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")


_OUTPUT_DIR = Path(__file__).parent / "output"


def _write_ranked_json(
    top_candidates: List[dict], csv_rows: List[dict], jd: dict, run_id: str
) -> Path:
    """Write a detailed per-run ranked JSON to output/ranked_<run_id>.json."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_map = {r["candidate_id"]: r for r in csv_rows}

    records = []
    for c in top_candidates:
        cid = str(c.get("candidate_id", c.get("id", "")))
        row = score_map.get(cid, {})
        records.append({
            "rank": row.get("rank"),
            "candidate_id": cid,
            "final_score": row.get("score"),
            "reasoning": row.get("reasoning", ""),
            "score_breakdown": {
                "l2_profile_jd_similarity": round(c.get("l2_score", 0.0), 4),
                "l3_seniority_penalty": round(c.get("l3_penalty", 1.0), 4),
                "l3_flag": c.get("l3_flag", ""),
                "l4_work_relevance": round(c.get("l4_score", 0.0), 4),
                "l6_behavioral": round(c.get("l6_score", 0.0), 4),
                "l7_fis_raw": round(c.get("l7_fis_raw", c.get("fis_score", 0.0)), 4),
                "flashrank_score": c.get("flashrank_score"),
                "l7_final_blended": round(c.get("fis_score", 0.0), 4),
                "flashrank_rank": c.get("flashrank_rank"),
            },
            "experience_years": utils.get_total_experience_years(c),
            "skills_snippet": utils.get_skills_text(c)[:200],
        })

    out = {
        "run_id": run_id,
        "jd_title": jd.get("job_title", ""),
        "jd_location": jd.get("location", ""),
        "total_ranked": len(records),
        "candidates": records,
    }
    path = _OUTPUT_DIR / f"ranked_{run_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Ranked JSON → {path}")
    return path


def _write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)


def _flashrank_available() -> bool:
    try:
        import flashrank  # noqa
        return True
    except ImportError:
        logger.warning("flashrank not installed; L7 polish disabled")
        return False


if __name__ == "__main__":
    main()
