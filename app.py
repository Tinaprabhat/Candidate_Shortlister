#!/usr/bin/env python3
"""
app.py — Streamlit UI for RedRob.

Pipeline:
  L1a hard-reject → L1b profile-integrity → L1c skill-match → L1d bonus-match
  → L2 table-extract → L3 fuzzy-score
    (streamed continuously per candidate, concurrently across candidates —
     no batch wait between stages; see layers.run_streaming_cascade)
  → gate (top 50% + random 25% = 75%)            ← the only compilation point
  → L4 semantic-score → top 200 → donts penalty → top 100 [l3_score + l4_score]
  → L5 FlashRank on top 50 (min-max normalised; total = (l3+l4+fr)/3) → final top-100 JSON

Run:  streamlit run app.py
"""

import os
import io
import csv
import json
import time
import tempfile
import threading
import shutil
import concurrent.futures
from pathlib import Path

import streamlit as st

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(".env")

def _read_env_file(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeError:
            continue
    return ""

if _env.exists():
    for _line in _read_env_file(_env).splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())


# ── Groq status (sidebar) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_groq_status():
    api_key    = os.environ.get("GROQ_API_KEY", "").strip()
    model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
    if not api_key:
        return {"state": "missing_key"}
    try:
        from groq import Groq
    except ImportError:
        return {"state": "error", "message": "groq package not installed; install with pip install groq"}
    try:
        client     = Groq(api_key=api_key)
        model_ids  = [m.id for m in client.models.list().data]
        if model_name in model_ids:
            return {"state": "working", "model": model_name}
        # Key is valid even if this specific model name isn't in the list
        return {"state": "working", "model": model_name}
    except Exception as exc:
        return {"state": "error", "message": str(exc)}


from src import constants as C
from src import utils, layers, pruning
from src.jd_parser import parse_jd
from src.tracer import RunTracer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RedRob — AI Candidate Ranker", page_icon="🤖", layout="wide")
st.title("🤖 RedRob — AI-Powered Candidate Ranking")
st.caption("Upload a candidates ZIP and a Job Description file. Get a ranked top-100 shortlist.")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    stagger = st.slider(
        "Folder dispatch stagger (sec)", 0, 60, C.FOLDER_DISPATCH_STAGGER_SEC,
        help="Seconds between dispatching each folder into the cascade",
    )
    has_ollama = shutil.which("ollama") is not None
    st.markdown(f"**Local Ollama CLI:** {'✅ available' if has_ollama else '❌ missing'}")

    gs = get_groq_status()
    if gs["state"] == "working":
        st.markdown(f"**Groq API:** ✅ `{gs['model']}`")
        st.success("Groq is ready for JD parsing.")
    elif gs["state"] == "missing_key":
        st.markdown("**Groq API:** ❌ GROQ_API_KEY not set")
        st.warning("JD parsing will fall back to local Ollama.")
    else:
        st.markdown(f"**Groq API:** ❌ {gs.get('message', gs['state'])}")
        st.warning("JD parsing will fall back to local Ollama.")

    st.markdown("---")
    st.markdown(
        "**Pipeline**\n"
        "1. L1 hard-reject\n"
        "2. L1b profile-integrity (hard reject only)\n"
        "3. L1c skill-match (explicit-skill gate)\n"
        "4. L2 table-extract (31 cols, incl. soft-penalty flags)\n"
        "5. L3 fuzzy-score (conditions a–h)\n"
        "6. Gate: top 50% + random 25% by L3 score\n"
        "7. L4 semantic-work → top 200\n"
        "8. Donts penalty → top 100\n"
        "9. L5 FlashRank top 50 → final top-100"
    )

# ── Uploaders ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    zip_file = st.file_uploader("📦 Candidates ZIP", type=["zip"])
with col2:
    jd_file = st.file_uploader(
        "📄 Job Description (PDF / DOCX / TXT / MD)",
        type=["pdf", "docx", "txt", "md"],
    )

run_btn = st.button("🚀 Run Ranking", type="primary", disabled=not (zip_file and jd_file))

_OUTPUT_DIR = Path(__file__).parent / "output"


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_models():
    def _try_flashrank():
        try:
            import flashrank  # noqa
            return utils.load_flashrank()
        except Exception:
            return None
    return {
        "fraud_kb":  utils.load_fraud_kb(),
        "flashrank": _try_flashrank(),
    }


# ── Streaming cascade (L1a→L1b→L1c→L1d→L2→L3, continuous & concurrent) ───────
def _run_streaming_cascade(cands, jd, models, tracer=None):
    """
    Every candidate streams continuously through L1a→L1b→L1c→L1d→L2→L3 with no
    batch wait in between, and candidates run concurrently against each other
    (a worker pool, not one-at-a-time). The gather here is the ONLY
    compilation point before the 75% FIS gate.
    """
    import logging as _log
    _logger = _log.getLogger("app.cascade")

    before = len(cands)
    t = time.time()
    survivors = layers.run_streaming_cascade(cands, jd, models["fraud_kb"])
    _logger.info(f"L1a→L3 streaming cascade elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L1a-L3_streaming_pipeline", before, len(survivors),
                                   notes={
                                       "elapsed_s": round(time.time()-t, 3),
                                       "avg_l3_score": round(
                                           sum(c.get("l3_score", 0) for c in survivors)
                                           / max(len(survivors), 1), 3
                                       ),
                                       "workers": C.PIPELINE_MAX_WORKERS,
                                   })
    return survivors


# ── Post-gate cascade (global, after the streaming cascade) ─────────────────
def _run_post_gate_cascade(candidates, jd, tracer=None):
    """L4 (all L3 survivors, donts baked in) → top 100."""
    import logging as _log
    _logger = _log.getLogger("app.cascade")

    # L4 semantic work (returns all sorted by l4_combined_score)
    t = time.time()
    candidates = layers.l4_semantic_work(candidates, jd)
    _logger.info(f"L4 elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L4_semantic_work", len(candidates), len(candidates),
                                   notes={"elapsed_s": round(time.time()-t, 3)})

    # Cap to top 100; tie-break ascending candidate_id
    top100 = sorted(
        candidates,
        key=lambda c: (
            -float(c.get("candidate_final_score", 0.0)),
            str(c.get("candidate_id", c.get("id", ""))),
        ),
    )[:100]
    n_pen = sum(1 for c in top100 if c.get("l4_donts_penalty", 0.0) > 0)
    _logger.info(f"Top {len(top100)} selected (ranks 1–{len(top100)}, {n_pen} donts-penalised)")
    if tracer:
        tracer.record_cascade_step("L4_top100", len(candidates), len(top100),
                                   notes={"donts_penalised": n_pen})

    # L5 DISABLED — top-100 from L4 is the final ranking; no FlashRank reshuffling.
    # # L5 FlashRank on top 50, min-max normalised; total_score = (l3 + l4 + flashrank) / 3
    # t = time.time()
    # top100 = layers.l5_flashrank_rerank(top100, jd)
    # _logger.info(f"L5 elapsed: {time.time()-t:.3f}s")
    # if tracer:
    #     tracer.record_cascade_step("L5_flashrank", len(top100), len(top100),
    #                                notes={"elapsed_s": round(time.time()-t, 3)})

    return top100


# ── Output helpers ────────────────────────────────────────────────────────────
def _write_ranked_json(top: list, rows: list, jd: dict, run_id: str) -> Path:
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
    score_map = {r["candidate_id"]: r for r in rows}

    # rows is already tie-break sorted; use that order for the JSON too
    cid_to_rank = {r["candidate_id"]: r["rank"] for r in rows}
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
        fscore = row.get("score")

        profile = c.get("profile") or {}
        signals = c.get("redrob_signals") or {}

        # ── 13: flags — all boolean/flag fields in one place ─────────────
        flags = {
            "l1_flags":              c.get("l1_flags") or [],
            "l1_status":             c.get("l1_status", "pass"),
            "l1b_flags":             c.get("l1b_flags") or [],
            "l1b_status":            c.get("l1b_status", "pass"),
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
            "l4_donts_triggered":    float(c.get("l4_donts_penalty") or 0.0) > 0.0,
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
        other_profile_data = {k: v for k, v in c.items() if k not in _main_keys}

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

            # 10: skill assessment score
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
                "l4_combined_score":           round(float(c.get("l4_combined_score") or 0.0), 4),
                "l5_flashrank_score":          round(float(c.get("l5_flashrank_score") or 0.0), 4),
                "l5_total_score":              round(float(c.get("l5_total_score") or 0.0), 4),
                "candidate_final_score":       round(float(c.get("candidate_final_score") or 0.0), 4),
            },

            # 20: full L2 table row
            "l2_table": c.get("table_row") or {},

            # 21: L3 reasoning string
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
    return path


# ── Run block ─────────────────────────────────────────────────────────────────
if run_btn:
    t0     = time.time()
    tracer = RunTracer()
    tmp    = Path(tempfile.mkdtemp(prefix="redrob_ui_"))

    zip_path = tmp / "candidates.zip"
    jd_path  = tmp / jd_file.name
    zip_path.write_bytes(zip_file.getvalue())
    jd_path.write_bytes(jd_file.getvalue())

    # ── Step 1 + 2: Parse JD & load models in parallel ───────────────────────
    with st.status("Parsing JD & loading models...", expanded=True) as status:
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            t_jd_submit = time.time()
            jd_fut     = ex.submit(parse_jd, jd_path, C.JD_JSON_PATH)
            t_models_submit = time.time()
            models_fut = ex.submit(_load_models)
            jd_exc = models_exc = None
            try:
                jd = jd_fut.result()
                tracer.record_timing("jd_parse", round(time.time() - t_jd_submit, 3))
            except Exception as e:
                jd_exc = e
            try:
                models = models_fut.result()
                tracer.record_timing("models_load", round(time.time() - t_models_submit, 3))
            except Exception as e:
                models_exc = e

        if jd_exc:
            status.update(label="JD parsing failed", state="error")
            st.error(f"JD parsing error: {jd_exc}")
            st.stop()
        if models_exc:
            status.update(label="Model loading failed", state="error")
            st.error(f"Model loading error: {models_exc}")
            st.stop()

        tracer.set_jd(jd)
        st.write(
            f"✅ JD parsed — **{jd.get('job_title','')}** | "
            f"Industry: **{jd.get('industry','n/a')}** | "
            f"Location: **{jd.get('location','n/a')}** | "
            f"Exp: **{jd.get('experience_min',0)}–{jd.get('experience_max',99)} yrs**"
        )
        st.write(
            f"Skills: **{len(jd.get('explicit_required',[]))} explicit** + "
            f"**{len(jd.get('inferred_required',[]))} inferred** | "
            f"Donts: **{len(jd.get('donts',[]))}**"
        )
        with st.expander("JD details"):
            st.json({k: jd.get(k) for k in [
                "job_title", "industry", "location",
                "experience_min", "experience_max",
                "explicit_required", "inferred_required", "donts",
            ]})
        st.write("✅ Models loaded")
        status.update(label="JD parsed & models ready ✅", state="complete")

    # ── Step 3: Load raw candidates per folder (pure I/O — no scoring yet) ──
    all_raw = []
    lock = threading.Lock()

    def process_fn(name, cands):
        with lock:
            all_raw.extend(cands)

    with st.status("Loading candidates...", expanded=True) as status:
        t_zip = time.time()
        root         = pruning.extract_zip(zip_path)
        folder_paths = pruning.discover_folders(root)
        tracer.record_timing("zip_extract_discover", round(time.time() - t_zip, 3))
        discovered   = list(folder_paths.keys())
        st.write(f"Discovered **{len(folder_paths)}** folder(s): {discovered}")

        kept       = pruning.prune_folders(discovered, jd)
        kept_names = [k for k, _ in kept]
        dropped    = [f for f in discovered if f not in kept_names]
        decisions  = {f: ("KEEP" if f in kept_names else "DROP:pruned") for f in discovered}
        tracer.record_pruning(discovered, decisions, kept)

        if dropped:
            st.warning(f"Pruned {len(dropped)} folder(s): {dropped}")
        st.write(f"Kept **{len(kept)}** folder(s) for cascade: {kept_names}")

        pruning.dispatch_folders_staggered(
            kept, folder_paths, process_fn, stagger_sec=stagger, tracer=tracer
        )
        st.write(f"✅ **{len(all_raw)}** candidates loaded")

        if not all_raw:
            status.update(label="No candidates loaded", state="error")
            st.error("No candidates loaded. Check logs/ for details.")
            st.stop()

        if len(all_raw) > C.HARD_POOL_CAP:
            all_raw[:] = all_raw[:C.HARD_POOL_CAP]
            st.info(f"Pool capped to {C.HARD_POOL_CAP}")

        status.update(label="Candidates loaded ✅", state="complete")

    # ── Step 4: Streaming cascade (L1a→L1b→L1c→L1d→L2→L3) ───────────────────
    with st.status("Streaming cascade (L1a→L3, continuous & concurrent)...", expanded=True) as status:
        st.write(f"Streaming {len(all_raw)} candidates through L1a→L1b→L1c→L1d→L2→L3…")
        all_scored = _run_streaming_cascade(all_raw, jd, models, tracer)
        st.write(f"✅ **{len(all_scored)}** candidates survived to the FIS gate")

        if not all_scored:
            status.update(label="No candidates survived the streaming cascade", state="error")
            st.error("No candidates survived L1a→L3. Check logs/ for details.")
            st.stop()

        status.update(label="Streaming cascade complete ✅", state="complete")

    # ── Step 5: Post-gate cascade (gate→L4+donts→L5) ────────────────────────
    with st.status("Post-gate cascade (gate→L4+donts→L5)...", expanded=True) as status:
        top100 = _run_post_gate_cascade(all_scored, jd, tracer)
        st.write(f"✅ Gate (75%) applied → forwarded to L4")
        n_donts_pen = sum(1 for c in top100 if c.get("l4_donts_penalty", 0.0) > 0)
        st.write(
            f"✅ L4 semantic scored + donts penalty baked in → top 100 selected "
            f"({n_donts_pen} donts-penalised)"
        )
        # st.write(f"✅ L5 FlashRank re-ranked top 50, min-max normalised (total = (l3+l4+fr)/3)")  # L5 DISABLED
        status.update(label="Post-gate cascade complete ✅", state="complete")

    if not top100:
        st.error("Post-gate cascade produced no candidates. Check logs/.")
        st.stop()

    # ── Step 5: Build output ──────────────────────────────────────────────────
    rows = []
    prev = float("inf")
    for i, c in enumerate(top100, 1):
        score = min(float(c.get("candidate_final_score", 0.0)), prev)
        prev  = score
        rows.append({
            "candidate_id": str(c.get("candidate_id", c.get("id", f"UNKNOWN_{i}"))),
            "rank":         i,
            "score":        round(score, 6),
            "reasoning":    c.get("l3_reasoning", ""),
        })

    t_write = time.time()
    json_path = _write_ranked_json(top100, rows, jd, tracer.run_id)

    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    w.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")
    tracer.record_timing("output_write", round(time.time() - t_write, 3))

    elapsed    = time.time() - t0
    trace_path = tracer.finish(
        output_json=json_path,
        rows_written=len(rows),
        top_scores=[r["score"] for r in rows[:10]],
        elapsed=elapsed,
    )

    st.success(f"✅ Ranked **{len(rows)}** candidates in **{elapsed:.1f}s**")
    st.caption(
        f"Run trace → `logs/{trace_path.name}` | "
        f"Ranked JSON → `output/ranked_{tracer.run_id}.json`"
    )

    st.download_button(
        "⬇️ Download submission.csv",
        csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
    )

    # ── Results table ─────────────────────────────────────────────────────────
    st.subheader("Top 100 Ranking")
    display_rows = []
    for r in rows:
        cid = r["candidate_id"]
        c   = next((x for x in top100
                    if str(x.get("candidate_id", x.get("id", ""))) == cid), {})
        display_rows.append({
            "Rank":          r["rank"],
            "Candidate ID":  cid,
            "Final Score":   r["score"],
            "L3 Fuzzy":      round(c.get("l3_score", 0.0), 3),
            "L3 Class":      c.get("l3_class", ""),
            "H Penalty":     int(c.get("table_row", {}).get("l3_h_penalty", False)),
            "L4 Relevance":  round(c.get("l4_work_relevance", 0.0), 3),
            "Donts Pen":     round(c.get("l4_donts_penalty", 0.0), 3),
            "L5 FlashRank":  round(c.get("l5_flashrank_score", 0.0), 3),
            "Reasoning":     c.get("l3_reasoning", "")[:120],
        })
    st.dataframe(display_rows, use_container_width=True, height=520)

    with st.expander("Full ranked JSON (first 5)"):
        st.json({
            "run_id": tracer.run_id,
            "candidates": json.loads(
                (json_path).read_text(encoding="utf-8")
            )["candidates"][:5],
        })
