#!/usr/bin/env python3
"""
app.py — Streamlit UI for RedRob.

Flow:
  1. User uploads candidates ZIP + JD file
  2. JD is parsed via local Ollama Mistral → jd.json
  3. rank.py cascade runs → top 100 CSV
  4. Results shown + downloadable

Run:  streamlit run app.py
"""

import os
import io
import json
import time
import tempfile
import threading
import shutil
import concurrent.futures
from pathlib import Path

import streamlit as st

# Load .env
_env = Path(".env")

def read_env_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return ""

if _env.exists():
    for line in read_env_file(_env).splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


@st.cache_data(show_spinner=False)
def get_gemini_status():
    model_name = os.environ.get("GEMINI_MODEL", "models/gemini-flash-latest").strip()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"state": "missing_key", "message": "GEMINI_API_KEY not set."}

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        known_models = {m.name for m in genai.list_models()}
        model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
        if model_name in known_models:
            return {"state": "working", "model": model_name}
        return {"state": "not_found", "message": f"Gemini model {model_name} not available."}
    except Exception as exc:
        return {"state": "error", "message": str(exc)}


from src import constants as C
from src import utils, layers, fis, pruning
from src.jd_parser import parse_jd
from src.tracer import RunTracer

st.set_page_config(page_title="RedRob — AI Candidate Ranker", page_icon="🤖", layout="wide")

st.title("🤖 RedRob — AI-Powered Candidate Ranking")
st.caption("Upload a candidate ZIP and a Job Description PDF. Get a ranked top-100 shortlist.")

# ── Sidebar: config ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    stagger = st.slider("Folder dispatch stagger (sec)", 0, 60, C.FOLDER_DISPATCH_STAGGER_SEC,
                        help="Seconds between dispatching each folder into the cascade")
    has_ollama = shutil.which("ollama") is not None
    st.markdown(f"**Local Ollama CLI:** {'✅ available' if has_ollama else '❌ missing'}")
    if has_ollama:
        st.info("Local Ollama is installed and available for fallback JD parsing.")
    else:
        st.warning("Install Ollama locally and ensure the model is available.")

    gemini_status = get_gemini_status()
    if gemini_status["state"] == "working":
        st.markdown(f"**Gemini API:** ✅ working ({gemini_status['model']})")
        st.success("Gemini is available and ready for JD parsing.")
    elif gemini_status["state"] == "missing_key":
        st.markdown("**Gemini API:** ❌ GEMINI_API_KEY not set")
        st.warning("Gemini is not configured; JD parsing will use local Ollama.")
    elif gemini_status["state"] == "not_found":
        st.markdown(f"**Gemini API:** ❌ {gemini_status['message']}")
        st.warning("Gemini model not available; JD parsing will use local Ollama.")
    else:
        st.markdown(f"**Gemini API:** ❌ {gemini_status['message']}")
        st.warning("Gemini is not available; JD parsing will use local Ollama.")

    st.markdown("---")
    st.markdown("**Pipeline:** L1 reject → L2 bi-encoder (55% gate) → "
                "L3 seniority → L4 semantic → L6 behavioral → "
                "L7 FIS + FlashRank polish")

# ── Uploaders ───────────────────────────────────────────────────────────────
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


def _write_ranked_json(
    top_candidates: list, csv_rows: list, jd: dict, run_id: str
) -> Path:
    """Write detailed ranked candidates to output/ranked_<run_id>.json."""
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
    return path


@st.cache_resource(show_spinner=False)
def _load_models():
    return {
        "st": utils.load_sentence_transformer(),
        "fraud_kb": utils.load_fraud_kb(),
        "flashrank": _try_flashrank(),
    }


def _try_flashrank():
    try:
        import flashrank  # noqa
        return utils.load_flashrank()
    except Exception:
        return None


def _run_cascade(cands, jd, models, tracer=None):
    import logging as _logging
    _log = _logging.getLogger("app.cascade")

    before = len(cands)
    t = time.time()
    survivors = layers.l1_hard_reject(cands, models["fraud_kb"])
    elapsed_l1 = time.time() - t
    _log.info(f"L1 elapsed: {elapsed_l1:.3f}s")
    if tracer:
        tracer.record_cascade_step("L1_hard_reject", before, len(survivors),
                                   notes={"elapsed_s": round(elapsed_l1, 3)})
    if not survivors:
        return []

    before = len(survivors)
    t = time.time()
    survivors = layers.l2_bi_encoder(survivors, jd, models["st"])
    survivors = layers.l2_gate(survivors)
    elapsed_l2 = time.time() - t
    _log.info(f"L2 elapsed: {elapsed_l2:.3f}s")
    if tracer:
        tracer.record_cascade_step("L2_bi_encoder_gate", before, len(survivors),
                                   notes={"elapsed_s": round(elapsed_l2, 3)})
    if not survivors:
        return []

    t = time.time()
    survivors = layers.l3_seniority(survivors, jd)
    elapsed_l3 = time.time() - t
    _log.info(f"L3 elapsed: {elapsed_l3:.3f}s")
    if tracer:
        n_pen = sum(1 for c in survivors if c.get("l3_penalty", 1.0) < 1.0)
        tracer.record_cascade_step("L3_seniority", len(survivors), len(survivors),
                                   notes={"penalized": n_pen,
                                          "elapsed_s": round(elapsed_l3, 3)})

    t = time.time()
    survivors = layers.l4_semantic_work(survivors, jd, models["st"])
    elapsed_l4 = time.time() - t
    _log.info(f"L4 elapsed: {elapsed_l4:.3f}s")
    if tracer:
        n_desc = sum(1 for c in survivors if c.get("l4_score", 0.0) > 0)
        tracer.record_cascade_step("L4_semantic_work", len(survivors), len(survivors),
                                   notes={"with_descriptions": n_desc,
                                          "no_descriptions": len(survivors) - n_desc,
                                          "elapsed_s": round(elapsed_l4, 3)})

    t = time.time()
    layers.l6_behavioral(survivors, jd)
    elapsed_l6 = time.time() - t
    _log.info(f"L6 elapsed: {elapsed_l6:.3f}s")
    if tracer:
        tracer.record_cascade_step("L6_behavioral", len(survivors), len(survivors),
                                   notes={"scored": len(survivors),
                                          "elapsed_s": round(elapsed_l6, 3)})
    return survivors


if run_btn:
    t0 = time.time()
    tracer = RunTracer()
    tmp = Path(tempfile.mkdtemp(prefix="redrob_ui_"))

    # Save uploads to temp dir
    zip_path = tmp / "candidates.zip"
    jd_path = tmp / jd_file.name
    zip_path.write_bytes(zip_file.getvalue())
    jd_path.write_bytes(jd_file.getvalue())

    # ── Step 1+2: Parse JD and load models in parallel ───────────────────────
    with st.status("Parsing JD & loading models in parallel...", expanded=True) as status:
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        jd_json_path = C.JD_JSON_PATH

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            jd_future = executor.submit(parse_jd, jd_path, jd_json_path)
            models_future = executor.submit(_load_models)

            jd_exc = None
            models_exc = None
            try:
                jd = jd_future.result()
            except Exception as e:
                jd_exc = e
            try:
                models = models_future.result()
            except Exception as e:
                models_exc = e

        if jd_exc:
            status.update(label="JD parsing failed", state="error")
            tracer.record_error("jd_parse", str(jd_exc))
            st.error(f"JD parsing error: {jd_exc}")
            st.stop()

        if models_exc:
            status.update(label="Model loading failed", state="error")
            tracer.record_error("model_load", str(models_exc))
            st.error(f"Model loading error: {models_exc}")
            st.stop()

        tracer.set_jd(jd)
        st.write(
            f"✅ JD parsed — {len(jd['explicit_required'])} required, "
            f"{len(jd['inferred_required'])} inferred skills"
        )
        st.write(
            f"📍 Location: **{jd.get('location') or 'not specified'}**  |  "
            f"⏱ Experience: **{jd.get('experience_min',0)}–{jd.get('experience_max',99)} yrs**"
        )
        st.json({
            k: jd[k]
            for k in ["job_title", "location", "experience_min", "experience_max",
                       "explicit_required", "inferred_required"]
        })
        st.write("✅ Models loaded")
        status.update(label="JD parsed & models loaded ✅", state="complete")

    # ── Step 3: Prune + cascade ──────────────────────────────────────────────
    all_scored = []
    lock = threading.Lock()

    def process_fn(name, cands):
        scored = _run_cascade(cands, jd, models, tracer)
        with lock:
            all_scored.extend(scored)

    with st.status("Running cascade...", expanded=True) as status:
        root = pruning.extract_zip(zip_path)
        folder_paths = pruning.discover_folders(root)
        discovered = list(folder_paths.keys())
        st.write(f"**Discovered {len(folder_paths)} folder(s):** {discovered}")

        kept = pruning.prune_folders(discovered, jd)
        kept_names = [k for k, _ in kept]
        dropped_names = [f for f in discovered if f not in kept_names]

        # Record pruning decisions for tracer
        decisions = {}
        for f in discovered:
            if f in kept_names:
                decisions[f] = "KEEP"
            else:
                decisions[f] = "DROP:pruned"
        tracer.record_pruning(discovered, decisions, kept)

        if dropped_names:
            st.warning(f"Pruned {len(dropped_names)} folder(s): {dropped_names}")
        st.write(f"**Kept {len(kept)} folder(s) for cascade:** {kept_names}")

        pruning.dispatch_folders_staggered(
            kept, folder_paths, process_fn, stagger_sec=stagger, tracer=tracer
        )
        st.write(f"✅ {len(all_scored)} candidates scored through cascade")
        status.update(label="Cascade complete ✅", state="complete")

    if not all_scored:
        tracer.record_error("cascade", "No candidates survived all layers")
        tracer.finish(rows_written=0, elapsed=time.time() - t0)
        st.error(
            "No candidates survived the cascade. "
            "Check logs/ for the run trace to debug."
        )
        st.stop()

    # ── Step 4: FIS + rank + polish ──────────────────────────────────────────
    with st.spinner("Final ranking (FIS + FlashRank)..."):
        import logging as _logging
        _log_l7 = _logging.getLogger("app.l7")
        if len(all_scored) > C.HARD_POOL_CAP:
            all_scored.sort(key=lambda c: c.get("l2_score", 0), reverse=True)
            all_scored = all_scored[:C.HARD_POOL_CAP]

        t = time.time()
        all_scored = fis.run_fis(all_scored)
        _log_l7.info(f"L7 FIS elapsed: {time.time()-t:.3f}s")

        t = time.time()
        ranked = fis.rank_candidates(all_scored, models["fraud_kb"])
        _log_l7.info(f"L7 rank elapsed: {time.time()-t:.3f}s")

        flashrank_ran = models["flashrank"] is not None
        t = time.time()
        ranked = fis.flashrank_polish(ranked, jd, models["flashrank"], models["fraud_kb"])
        _log_l7.info(f"L7 FlashRank polish elapsed: {time.time()-t:.3f}s")

        top = ranked[:C.OUTPUT_TOP_N]
        fis_scores = [c.get("fis_score", 0.0) for c in top]
        tracer.record_l7(
            fis_input=len(all_scored),
            fis_output=len(ranked),
            flashrank_ran=flashrank_ran,
            flashrank_top_n=min(C.FLASHRANK_TOP_N, len(ranked)) if flashrank_ran else 0,
            score_range=(min(fis_scores, default=0.0), max(fis_scores, default=0.0)),
        )

    # ── Build CSV ─────────────────────────────────────────────────────────────
    import csv
    rows, prev = [], 1.0
    for i, c in enumerate(top, 1):
        score = min(c.get("fis_score", 0.0), prev); prev = score
        rows.append({
            "candidate_id": str(c.get("candidate_id", c.get("id", f"UNKNOWN_{i}"))),
            "rank": i,
            "score": round(score, 6),
            "reasoning": fis.generate_reasoning(c, jd),
        })

    # ── Save detailed ranked JSON to output/ ──────────────────────────────────
    json_path = _write_ranked_json(top, rows, jd, tracer.run_id)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader(); w.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")

    elapsed = time.time() - t0
    top_scores = [r["score"] for r in rows[:10]]
    trace_path = tracer.finish(
        output_json=json_path,
        rows_written=len(rows), top_scores=top_scores, elapsed=elapsed
    )
    st.success(f"✅ Ranked {len(rows)} candidates in {elapsed:.1f}s")
    st.caption(f"Run trace → `{trace_path.name}` | Ranked JSON → `output/ranked_{tracer.run_id}.json`")

    st.download_button("⬇️ Download submission.csv", csv_bytes,
                       file_name="submission.csv", mime="text/csv", type="primary")

    st.subheader("Top 100 Ranking")
    st.dataframe(rows, use_container_width=True, height=500)
