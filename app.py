#!/usr/bin/env python3
"""
app.py — Streamlit UI for RedRob.

Pipeline:
  L1 hard-reject → L1b profile-integrity → L1c skill-match
  → L2 table-extract → L3 fuzzy-score → gate (top 50% + random 25% = 75%)
  → L4 semantic-score → top 200 → donts penalty → top 100
  → L5 FlashRank on top 50 (total_score = prev + flashrank) → final top-100 JSON

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


# ── Per-folder early cascade ──────────────────────────────────────────────────
def _run_early_cascade(cands, jd, models, tracer=None):
    """L1 → L1b → L1c (no seniority — subsumed by L3 fuzzy)."""
    import logging as _log
    _logger = _log.getLogger("app.cascade")

    before = len(cands)
    t = time.time()
    survivors = layers.l1_hard_reject(cands, models["fraud_kb"])
    _logger.info(f"L1 elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L1_hard_reject", before, len(survivors),
                                   notes={"elapsed_s": round(time.time()-t, 3)})
    if not survivors:
        return []

    before = len(survivors)
    t = time.time()
    survivors = layers.l1b_profile_integrity(survivors)
    _logger.info(f"L1b elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L1b_profile_integrity", before, len(survivors),
                                   notes={"elapsed_s": round(time.time()-t, 3)})
    if not survivors:
        return []

    before = len(survivors)
    t = time.time()
    survivors = layers.l1c_skill_match(survivors, jd)
    _logger.info(f"L1c elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L1c_skill_match", before, len(survivors),
                                   notes={
                                       "elapsed_s": round(time.time()-t, 3),
                                       "avg_score": round(
                                           sum(c.get("l1c_score", 0) for c in survivors)
                                           / max(len(survivors), 1), 3
                                       ),
                                   })
    return survivors


# ── Late cascade (global, after early cascade) ───────────────────────────────
def _run_late_cascade(candidates, jd, tracer=None):
    """L2 → L3 → gate → L4 → top200 → donts → top100 → L5."""
    import logging as _log
    _logger = _log.getLogger("app.cascade")

    # L2 table extract
    t = time.time()
    candidates = layers.l2_table_extract(candidates, jd)
    _logger.info(f"L2 elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L2_table_extract", len(candidates), len(candidates),
                                   notes={"elapsed_s": round(time.time()-t, 3)})

    # L3 fuzzy score (no knockouts — only L1 hard-rejects)
    t = time.time()
    candidates = layers.l3_fuzzy_score(candidates, jd)
    _logger.info(f"L3 elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L3_fuzzy_score", len(candidates), len(candidates),
                                   notes={
                                       "elapsed_s": round(time.time()-t, 3),
                                       "avg_score": round(
                                           sum(c.get("l3_score", 0) for c in candidates)
                                           / max(len(candidates), 1), 3
                                       ),
                                   })

    # Gate — top 50% + random 25% by l3_score = 75%
    before_gate = len(candidates)
    t = time.time()
    candidates = layers.l3_gate(candidates)
    _logger.info(f"L3 gate: {before_gate} → {len(candidates)} elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L3_gate", before_gate, len(candidates),
                                   notes={
                                       "elapsed_s": round(time.time()-t, 3),
                                       "top_pct":    C.GATE_TOP_FRACTION,
                                       "random_pct": C.GATE_RANDOM_FRACTION,
                                   })

    # L4 semantic work (returns all sorted by l4_combined_score)
    t = time.time()
    candidates = layers.l4_semantic_work(candidates, jd)
    _logger.info(f"L4 elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L4_semantic_work", len(candidates), len(candidates),
                                   notes={"elapsed_s": round(time.time()-t, 3)})

    # Top 200 by l4_combined_score (list already sorted)
    top200 = candidates[:200]

    # Donts penalty (high) → top 100
    t = time.time()
    top100 = layers.donts_penalty_layer(top200, jd)
    n_pen = sum(1 for c in top100 if c.get("l5_donts_mult", 1.0) < 1.0)
    _logger.info(f"Donts elapsed: {time.time()-t:.3f}s  penalised={n_pen}")
    if tracer:
        tracer.record_cascade_step("Donts_penalty", len(top200), len(top100),
                                   notes={
                                       "elapsed_s": round(time.time()-t, 3),
                                       "donts_penalised": n_pen,
                                   })

    # L5 FlashRank on top 50; total_score = donts_score + flashrank
    t = time.time()
    top100 = layers.l5_flashrank_rerank(top100, jd)
    _logger.info(f"L5 elapsed: {time.time()-t:.3f}s")
    if tracer:
        tracer.record_cascade_step("L5_flashrank", len(top100), len(top100),
                                   notes={"elapsed_s": round(time.time()-t, 3)})

    return top100


# ── Output helpers ────────────────────────────────────────────────────────────
def _write_ranked_json(top: list, rows: list, jd: dict, run_id: str) -> Path:
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
                "l1b_flags":             c.get("l1b_flags", []),
                "l3_fuzzy_score":        round(c.get("l3_score", 0.0), 4),
                "l3_class":              c.get("l3_class", ""),
                "l4_work_relevance":     round(c.get("l4_work_relevance", 0.0), 4),
                "l4_combined_score":     round(c.get("l4_combined_score", 0.0), 4),
                "l5_donts_mult":         round(c.get("l5_donts_mult", 1.0), 4),
                "l5_donts_score":        round(c.get("l5_donts_score", 0.0), 4),
                "l5_flashrank_score":    round(c.get("l5_flashrank_score", 0.0), 4),
                "l5_total_score":        round(c.get("l5_total_score", 0.0), 4),
            },
            "experience_years": utils.get_total_experience_years(c),
            "skills_snippet":   utils.get_skills_text(c)[:200],
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
        json.dump(out, f, indent=2, ensure_ascii=False)
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
            jd_fut     = ex.submit(parse_jd, jd_path, C.JD_JSON_PATH)
            models_fut = ex.submit(_load_models)
            jd_exc = models_exc = None
            try:
                jd = jd_fut.result()
            except Exception as e:
                jd_exc = e
            try:
                models = models_fut.result()
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

    # ── Step 3: Early cascade (L1→L1b→L1c) per folder ───────────────────────
    all_early_scored = []
    lock = threading.Lock()

    def process_fn(name, cands):
        scored = _run_early_cascade(cands, jd, models, tracer)
        with lock:
            all_early_scored.extend(scored)

    with st.status("Early cascade (L1→L1b→L1c)...", expanded=True) as status:
        root         = pruning.extract_zip(zip_path)
        folder_paths = pruning.discover_folders(root)
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
        st.write(f"✅ **{len(all_early_scored)}** candidates survived L1→L1b→L1c")

        if not all_early_scored:
            status.update(label="No candidates survived early cascade", state="error")
            st.error("No candidates survived L1/L1b/L1c. Check logs/ for details.")
            st.stop()

        if len(all_early_scored) > C.HARD_POOL_CAP:
            all_early_scored.sort(key=lambda c: c.get("l1c_score", 0), reverse=True)
            all_early_scored[:] = all_early_scored[:C.HARD_POOL_CAP]
            st.info(f"Pool capped to {C.HARD_POOL_CAP}")

        status.update(label="Early cascade complete ✅", state="complete")

    # ── Step 4: Late cascade (L2→L3→gate→L4→donts→L5) ───────────────────────
    with st.status("Late cascade (L2→L3→gate→L4→donts→L5)...", expanded=True) as status:
        st.write(f"L2 building table rows for {len(all_early_scored)} candidates…")
        top100 = _run_late_cascade(all_early_scored, jd, tracer)
        st.write(f"✅ L3 fuzzy scored → gate (75%) applied → forwarded to L4")
        st.write(f"✅ L4 semantic scored → top 200 selected")
        st.write(
            f"✅ Donts penalty applied → top 100 | "
            f"{sum(1 for c in top100 if c.get('l5_donts_mult',1.0)<1.0)} penalised"
        )
        st.write(f"✅ L5 FlashRank re-ranked top 50 (total = donts_score + flashrank)")
        status.update(label="Late cascade complete ✅", state="complete")

    if not top100:
        st.error("Late cascade produced no candidates. Check logs/.")
        st.stop()

    # ── Step 5: Build output ──────────────────────────────────────────────────
    rows = []
    prev = float("inf")
    for i, c in enumerate(top100, 1):
        score = min(float(c.get("l5_total_score", 0.0)), prev)
        prev  = score
        rows.append({
            "candidate_id": str(c.get("candidate_id", c.get("id", f"UNKNOWN_{i}"))),
            "rank":         i,
            "score":        round(score, 6),
            "reasoning":    c.get("l3_reasoning", ""),
        })

    json_path = _write_ranked_json(top100, rows, jd, tracer.run_id)

    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    w.writeheader()
    w.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")

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
            "Donts Mult":    round(c.get("l5_donts_mult", 1.0), 3),
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
