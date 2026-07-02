#!/usr/bin/env python3
"""
rank.py — RedRob ranking entry point (OFFLINE, no API calls).

Pipeline:
  L1a hard-reject → L1b profile-integrity → L1c skill-match → L1d inferred-match
  → L2 table-extract → L3 fuzzy-score
    (streamed continuously per candidate, concurrently across candidates —
     no batch wait between stages; see layers.run_streaming_cascade)
  → L4 semantic-score + donts penalty (baked in) [all L3 survivors]
  → top 100 by l4_combined_score

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


# ── Streaming cascade (L1a→L1b→L1c→L1d→L2→L3, continuous & concurrent) ───────

def run_streaming_cascade(
    candidates: List[dict], jd: dict, models: dict, tracer: RunTracer
) -> List[dict]:
    """
    Every candidate streams continuously through L1a→L1b→L1c→L1d→L2→L3 with no
    batch wait in between, and candidates run concurrently against each other
    (a worker pool, not one-at-a-time). The gather below is the ONLY
    compilation point before the 75% FIS gate.
    """
    before = len(candidates)
    t = time.time()
    survivors = layers.run_streaming_cascade(candidates, jd, models["fraud_kb"])
    tracer.record_cascade_step("L1a-L3_streaming_pipeline", before, len(survivors),
                               notes={
                                   "elapsed_s": round(time.time() - t, 3),
                                   "avg_l3_score": round(
                                       sum(c.get("l3_score", 0) for c in survivors)
                                       / max(len(survivors), 1), 3
                                   ),
                                   "workers": C.PIPELINE_MAX_WORKERS,
                               })
    return survivors


# ── Post-gate cascade (global, after the streaming cascade) ─────────────────

def run_post_gate_cascade(
    candidates: List[dict], jd: dict, tracer: RunTracer
) -> List[dict]:
    """
    L4 semantic-score (all L3 survivors) → top 100 [l4_combined_score]
    """
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

    # L4b — explicit required-skill penalty (< 4 matched → graduated −0.075/missing)
    candidates = layers.l4b_explicit_req_penalty(candidates)
    n_l4b = sum(1 for c in candidates if c.get("l4b_explicit_req_penalty", 0.0) > 0)
    tracer.record_cascade_step("L4b_explicit_req_penalty", len(candidates), len(candidates),
                               notes={"penalised": n_l4b})

    # Cap to top 100 by candidate_final_score; tie-break ascending candidate_id
    top100 = sorted(
        candidates,
        key=lambda c: (
            -float(c.get("candidate_final_score", 0.0)),
            str(c.get("candidate_id", c.get("id", ""))),
        ),
    )[:100]
    n_penalised = sum(1 for c in top100 if c.get("l4_donts_penalty", 0.0) > 0)
    logger.info(f"After L4b: {len(candidates)} scored → top {len(top100)} (ranks 1–{len(top100)}, {n_penalised} donts-penalised, {n_l4b} explicit-req-penalised)")
    tracer.record_cascade_step("L4_top100", len(candidates), len(top100),
                               notes={"donts_penalised": n_penalised})

    # L5 DISABLED — top-100 from L4 is the final ranking; no FlashRank reshuffling.
    # # L5 — FlashRank on top 50 (min-max normalised); total_score = (l3 + l4 + flashrank) / 3
    # t = time.time()
    # top100 = layers.l5_flashrank_rerank(top100, jd)
    # tracer.record_cascade_step("L5_flashrank", len(top100), len(top100),
    #                            notes={
    #                                "elapsed_s": round(time.time() - t, 3),
    #                                "avg_total": round(
    #                                    sum(c.get("l5_total_score", 0) for c in top100)
    #                                    / max(len(top100), 1), 3
    #                                ),
    #                            })

    return top100


# ── Output helpers ────────────────────────────────────────────────────────────

def _build_reasoning(c: dict) -> str:
    profile = c.get("profile") or {}
    title   = str(
        profile.get("current_title") or profile.get("title") or profile.get("headline") or "Candidate"
    ).strip()
    exp     = utils.get_total_experience_years(c)
    exp_str = f"{exp:.0f} yrs" if exp else "N/A"
    l1c     = round(float(c.get("l1c_score") or 0.0), 2)
    prof    = round(float(c.get("l1c_explicit_proficiency_score") or 0.0), 2)
    wr      = round(float(c.get("l4_work_relevance") or 0.0), 2)
    l3      = round(float(c.get("l3_score") or 0.0), 2)
    return (
        f"{title} with {exp_str}, {l1c} score for explicit skill match "
        f"with {prof} proficiency score. Has {wr} work relevance score "
        f"and {l3} reasoning layer score. Thus a Good fit for this position."
    )


def _build_rows(top: List[dict]) -> List[dict]:
    """
    Assign ranks 1–N to the already-sorted top list.
    top must arrive sorted (descending final_score, ascending candidate_id on tie)
    — the sort is done once in run_post_gate_cascade.
    """
    rows = []
    for rank_pos, c in enumerate(top, start=1):
        rows.append({
            "Cand_ID":     str(c.get("candidate_id", c.get("id", f"UNKNOWN_{rank_pos}"))),
            "rank":        rank_pos,
            "final_score": round(float(c.get("candidate_final_score", 0.0)), 6),
            "reasoning":   _build_reasoning(c),
        })
    return rows


def _write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Cand_ID", "rank", "final_score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)


def _write_ranked_json(top: List[dict], rows: List[dict], jd: dict, run_id: str) -> Path:
    """
    Write the full ranked output JSON.  Fields follow the 21-point spec:
      1  rank                    2  candidate_id           3  final_score
      4  profile                 5  education              6  career_history
      7  skills                  8  projects               9  publications
      10 skill_assessment_score  11 redrob_signals         12 other_profile_data
      13 flags                   14 explicit_req_matched   15 explicit_bonus_matched
      16 inferred_matched        17 unmatched_skills       18 tool_list
      19 layer_scores            20 l2_table               21 reasoning
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    score_map = {r["Cand_ID"]: r for r in rows}

    # rows is already tie-break sorted; use that order for the JSON too
    cid_to_rank = {r["Cand_ID"]: r["rank"] for r in rows}
    ordered_top = sorted(
        top,
        key=lambda c: cid_to_rank.get(
            str(c.get("candidate_id", c.get("id", ""))), 9999
        ),
    )

    records = []
    for c in ordered_top:
        cid    = str(c.get("candidate_id", c.get("id", "")))
        row    = score_map.get(cid, {})
        rank   = row.get("rank")
        fscore = row.get("final_score")

        profile = c.get("profile") or {}
        signals = c.get("redrob_signals") or {}

        # ── 13: flags — all boolean/flag fields in one place ─────────────
        flags = {
            # L1a fraud flags
            "l1_flags":              c.get("l1_flags") or [],
            "l1_status":             c.get("l1_status", "pass"),
            # L1b availability flags
            "l1b_flags":             c.get("l1b_flags") or [],
            "l1b_status":            c.get("l1b_status", "pass"),
            # L2/L3 profile flags (from table_row)
            "is_phd":                bool((c.get("table_row") or {}).get("is_phd")),
            "consulting_only":       bool((c.get("table_row") or {}).get("consulting_only")),
            "research_published":    bool((c.get("table_row") or {}).get("research_published")),
            "edu_career_gap_flag":   bool((c.get("table_row") or {}).get("edu_career_gap_flag")),
            "low_engagement_flag":   bool((c.get("table_row") or {}).get("low_engagement_flag")),
            "possible_fabrication":  bool((c.get("table_row") or {}).get("possible_fabrication")),
            "no_certifications":     bool((c.get("table_row") or {}).get("no_certifications")),
            "no_offer_history":      bool((c.get("table_row") or {}).get("no_offer_history")),
            "second_undergrad_after_first": bool((c.get("table_row") or {}).get("second_undergrad_after_first")),
            "skill_career_domain_mismatch": bool((c.get("table_row") or {}).get("skill_career_domain_mismatch")),
            # L4 donts flag
            "l4_donts_triggered":    float(c.get("l4_donts_penalty") or 0.0) > 0.0,
            # Platform flags from redrob_signals
            "open_to_work":          signals.get("open_to_work_flag"),
            "verified_email":        signals.get("verified_email"),
            "verified_phone":        signals.get("verified_phone"),
            "linkedin_connected":    signals.get("linkedin_connected"),
            "github_not_linked":     signals.get("github_not_linked"),
            "no_offer_history_sig":  signals.get("no_offer_history"),
        }

        # ── 12: other_profile_data — every candidate field not in main sections ─
        _main_keys = {
            "candidate_id", "id", "name", "profile", "education", "career_history",
            "work_experience", "skills", "projects", "publications", "research_papers",
            "redrob_signals", "certifications", "languages",
            # pipeline-added keys
            "l1_score", "l1_flags", "l1_status", "l1b_penalty", "l1b_flags", "l1b_status",
            "l1c_score", "l1c_matched_required", "l1c_missing_required", "l1c_matched_bonus",
            "l1c_matched_inferred", "l1c_unmatched_skills", "l1c_matched_explicit",
            "l1c_explicit_proficiency_score", "jd_req_score", "jd_req_results",
            "l1d_matched_inferred", "l1d_unmatched_inferred", "l1d_inferred_ratio",
            "l1d_leftover_count", "l1d_tool_list", "l1d_tools_score", "l1d_score",
            "l1d_inferred_matched_score", "l1d_explicit_req_matched_score",
            "l1d_explicit_bonus_matched_score", "l1d_unmatched_skills_score",
            "l1d_inferred_proficiency_score",
            "table_row", "l3_score", "l3_class", "l3_reasoning",
            "l4_work_relevance", "l4_donts_sim", "l4_donts_penalty",
            "l4_combined_score", "l4_score", "candidate_final_score",
            "l5_flashrank_score", "l5_total_score",
            "_folder",
        }
        other_profile_data = {
            k: v for k, v in c.items() if k not in _main_keys
        }

        records.append({
            # 1–3: identity
            "rank":          rank,
            "candidate_id":  cid,
            "final_score":   fscore,

            # 4–9: core profile sections
            "profile":        profile,
            "education":      c.get("education") or [],
            "career_history": c.get("career_history") or c.get("work_experience") or [],
            "skills":         c.get("skills") or [],
            "projects":       c.get("projects") or [],
            "publications":   c.get("publications") or c.get("research_papers") or [],

            # 10: skill assessment score (extracted from signals for quick access)
            "skill_assessment_score": signals.get("skill_assessment_scores") or {},

            # 11: full redrob platform signals
            "redrob_signals": signals,

            # 12: everything else from the candidate record
            "other_profile_data": other_profile_data,

            # 13: all flags
            "flags": flags,

            # 14–18: skill matching lists (from L1c/L1d)
            "explicit_req_matched":   c.get("l1c_matched_required", []),
            "explicit_bonus_matched": c.get("l1c_matched_bonus", []),
            "inferred_matched":       c.get("l1c_matched_inferred", []),
            "unmatched_skills":       c.get("l1c_unmatched_skills", []),
            "tool_list":              c.get("l1d_tool_list", []),

            # 19: per-layer scores
            "layer_scores": {
                "l1_score":                    round(float(c.get("l1_score") or 0.5), 4),
                "l1b_penalty":                 round(float(c.get("l1b_penalty") or 1.0), 4),
                "l1c_score":                   round(float(c.get("l1c_score") or 0.0), 4),
                "l1c_explicit_proficiency":    round(float(c.get("l1c_explicit_proficiency_score") or 0.0), 4),
                "l1d_score":                   round(float(c.get("l1d_score") or 0.0), 4),
                "l1d_inferred_ratio":          round(float(c.get("l1d_inferred_ratio") or 0.0), 4),
                "l1d_tools_score":             int(c.get("l1d_tools_score") or 0),
                "l1d_inferred_proficiency":    round(float(c.get("l1d_inferred_proficiency_score") or 0.0), 4),
                "jd_req_score":                round(float(c.get("jd_req_score") or 0.0), 4),
                "l3_score":                    round(float(c.get("l3_score") or 0.0), 4),
                "l3_class":                    c.get("l3_class", ""),
                "l4_work_relevance":           round(float(c.get("l4_work_relevance") or 0.0), 4),
                "l4_donts_sim":                round(float(c.get("l4_donts_sim") or 0.0), 4),
                "l4_donts_penalty":            round(float(c.get("l4_donts_penalty") or 0.0), 4),
                "l4b_explicit_req_penalty":    round(float(c.get("l4b_explicit_req_penalty") or 0.0), 4),
                "l4_combined_score":           round(float(c.get("l4_combined_score") or 0.0), 4),
                "l5_flashrank_score":          round(float(c.get("l5_flashrank_score") or 0.0), 4),
                "l5_total_score":              round(float(c.get("l5_total_score") or 0.0), 4),
                "candidate_final_score":       round(float(c.get("candidate_final_score") or 0.0), 4),
            },

            # 20: full L2 table row
            "l2_table": c.get("table_row") or {},

            # 21: reasoning summary
            "reasoning": _build_reasoning(c),
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

    # Gather raw candidates from every kept folder (pure I/O — no scoring yet;
    # scoring happens once, globally, in the streaming cascade below so that
    # candidates from different folders can pipeline concurrently together).
    all_raw: List[dict] = []
    lock = threading.Lock()

    def process_fn(_, cands: List[dict]):
        with lock:
            all_raw.extend(cands)

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

    if not all_raw:
        logger.error("No candidates loaded.")
        tracer.record_error("cascade", "No candidates loaded")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # Hard pool cap before the streaming cascade — applied pre-scoring since
    # the cascade no longer pauses between L1 and L2/L3 to compute a proxy sort key.
    if len(all_raw) > C.HARD_POOL_CAP:
        all_raw = all_raw[:C.HARD_POOL_CAP]
        logger.info(f"Capped pool to {C.HARD_POOL_CAP}")

    # Streaming cascade — L1a→L1b→L1c→L1d→L2→L3, continuous per candidate,
    # concurrent across candidates. This is where every candidate gets l3_score.
    all_scored = run_streaming_cascade(all_raw, jd, models, tracer)

    if not all_scored:
        logger.error("No candidates survived the streaming cascade (L1a→L3).")
        tracer.record_error("cascade", "No candidates survived streaming cascade")
        _write_csv(args.out, [])
        tracer.finish(args.out, rows_written=0, elapsed=time.time() - t0)
        return

    # Post-L3 cascade — L4 (all survivors) → top100
    top100 = run_post_gate_cascade(all_scored, jd, tracer)

    if not top100:
        logger.error("Post-gate cascade produced no candidates.")
        tracer.record_error("cascade", "Post-gate cascade returned empty")
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
        top_scores=[r["final_score"] for r in rows[:10]],
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
