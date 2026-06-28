"""
app.py — RedRob Streamlit sandbox for HuggingFace Spaces.

Accepts a candidate JSONL file (≤100 candidates), runs the full offline
ranking pipeline, and lets the user download the ranked submission CSV.
"""

import os
import sys
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
COMPRESSED  = ROOT / "models" / "compressed"
DECOMPRESSED = ROOT / "models" / "decompressed"
JD_JSON     = ROOT / "data" / "jd.json"
RANK_PY     = ROOT / "rank.py"

# ── One-time model decompression ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Decompressing models (first run only)…")
def setup_models():
    if not COMPRESSED.exists():
        return "❌ models/compressed/ not found. Did Git LFS files download?"
    DECOMPRESSED.mkdir(parents=True, exist_ok=True)
    archives = list(COMPRESSED.glob("*.tar.gz"))
    if not archives:
        return "❌ No .tar.gz archives found in models/compressed/"
    for archive in archives:
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(DECOMPRESSED)
    return "ok"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RedRob Ranker",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 RedRob — Candidate Ranker")
st.caption("Intelligent Candidate Discovery & Ranking · India Runs Data & AI Challenge 2026")
st.divider()

# ── Model setup ───────────────────────────────────────────────────────────────
status = setup_models()
if status != "ok":
    st.error(status)
    st.stop()

# ── JD check ──────────────────────────────────────────────────────────────────
if not JD_JSON.exists():
    st.error("❌ `data/jd.json` not found. It must be present in the repository.")
    st.stop()

# ── Info panel ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How this works", expanded=False):
    st.markdown("""
**Pipeline (offline, CPU-only, zero API calls):**

```
Upload JSONL → Pre-process → Folder Pruning
→ L1 Fraud Reject → L1b Profile Integrity → L1c Skill Match (NLP)
→ L2 Feature Extract → L3 Sugeno Fuzzy Score
→ L4 Semantic Relevance (MiniLM) → Top 200
→ L5a Don'ts Penalty → Top 100
→ L5b FlashRank Cross-Encoder (Top 50)
→ FIS Mamdani Final Score → submission.csv
```

**Input:** A `.jsonl` file where each line is a candidate JSON profile.
**Output:** A ranked `submission.csv` with columns: `candidate_id, rank, score, reasoning`.
**Limit:** Works with any pool size. Sandbox demo uses ≤100 candidates.
    """)

# ── File upload ───────────────────────────────────────────────────────────────
st.subheader("1. Upload Candidates")
uploaded = st.file_uploader(
    "Upload a `.jsonl` file (one candidate JSON per line)",
    type=["jsonl", "json"],
    help="Each line must be a valid candidate profile matching the Redrob schema.",
)

if uploaded:
    lines = uploaded.read().decode("utf-8").strip().splitlines()
    n = len(lines)
    st.success(f"✅ {n} candidate{'s' if n != 1 else ''} loaded.")
    if n > 100:
        st.warning(f"⚠️ {n} candidates detected. Sandbox is optimised for ≤100. Proceeding anyway.")

    st.subheader("2. Run the Ranker")
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path   = Path(tmp)
            input_file = tmp_path / "candidates.jsonl"
            output_file = tmp_path / "submission.csv"

            input_file.write_text("\n".join(lines), encoding="utf-8")

            with st.status("Running pipeline…", expanded=True) as status_box:
                st.write("Starting rank.py…")
                t0 = time.time()

                result = subprocess.run(
                    [
                        sys.executable,
                        str(RANK_PY),
                        "--candidates", str(input_file),
                        "--jd",         str(JD_JSON),
                        "--out",        str(output_file),
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(ROOT),
                )

                elapsed = round(time.time() - t0, 1)

                if result.returncode != 0:
                    status_box.update(label="❌ Pipeline failed", state="error")
                    st.error("rank.py returned a non-zero exit code.")
                    st.code(result.stderr[-3000:], language="text")
                    st.stop()

                status_box.update(label=f"✅ Done in {elapsed}s", state="complete")
                st.write(f"Completed in **{elapsed}s**")

            # ── Results ───────────────────────────────────────────────────────
            st.subheader("3. Results")

            if not output_file.exists():
                st.error("submission.csv was not produced. Check logs above.")
                st.stop()

            csv_text = output_file.read_text(encoding="utf-8")
            rows = csv_text.strip().splitlines()
            n_ranked = len(rows) - 1  # subtract header

            st.metric("Candidates ranked", n_ranked)

            # Preview top 10
            import csv, io
            reader = list(csv.DictReader(io.StringIO(csv_text)))
            if reader:
                import pandas as pd
                df = pd.DataFrame(reader[:10])
                st.markdown("**Top 10 candidates:**")
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Download button
            st.download_button(
                label="⬇️ Download submission.csv",
                data=csv_text,
                file_name="submission.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
            )

            # Logs (collapsed)
            with st.expander("📋 Pipeline logs"):
                st.code(result.stdout[-5000:], language="text")

else:
    st.info("Upload a `.jsonl` file above to get started.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "RedRob · Team RedRob · India Runs Data & AI Challenge 2026 · "
    "CPU-only · No API calls · Zero network during ranking"
)
