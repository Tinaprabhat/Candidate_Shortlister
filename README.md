# RedRob — AI-Powered Candidate Ranking System

RedRob ranks the top 100 best-fit candidates for a job description from a pool
of up to ~100,000 profiles — **offline, CPU-only, under 5 minutes.**

It uses a 6-layer cascade (fraud rejection → semantic similarity → seniority →
work relevance → sanity checks → behavioral signals) feeding a Mamdani Fuzzy
Inference System, with a FlashRank cross-encoder polishing the top 50.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm        # optional (L5 helper)

# 2. Build model artifacts (ONE TIME — may take a while, needs internet)
python build_kb.py --all
# This downloads + quantizes models and builds the fraud KB into
# models/compressed/*.tar.gz

# 3. Decompress artifacts (after clone, or after build_kb)
bash setup.sh

# 4. Verify everything is ready
python setup_check.py

# 5a. Parse the JD using local Ollama Mistral
# Optional: set OLLAMA_MODEL in .env, e.g. OLLAMA_MODEL=mistral:latest
python -m src.jd_parser --jd ./data/job_description.pdf --out ./data/jd.json

# 5b. Run ranking (OFFLINE — the reproducible Stage-3 step)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

---

## Single-Command Reproduction (Stage 3)

Once `data/jd.json` exists (from the pre-step) and `bash setup.sh` has run:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This step makes **no network calls**, uses **no GPU**, and completes within the
5-minute / 16 GB budget.

---

## Web UI (Streamlit)

```bash
streamlit run app.py
```

Upload a candidates ZIP + JD file, get a downloadable ranked CSV. The UI runs the
local Ollama JD parsing step and the offline cascade together for convenience.

---

## ⚠️ Important: Offline JD Parsing

The competition forbids external API calls **during the ranking step**.

RedRob respects this by splitting JD parsing out as a **pre-computation step**:

- `src/jd_parser.py` uses local Ollama Mistral to produce `data/jd.json`.
- `rank.py` reads the already-existing `data/jd.json` from disk and makes
  **zero API calls**.

The output schema (`jd.json`) is identical whether the JD parser is run via the
UI or standalone, so the ranking step remains fully offline.

---

## Pipeline Overview

| Layer | What it does | Tool |
|-------|--------------|------|
| Prune | Drop irrelevant candidate folders | Heuristic token overlap (no AI) |
| L1 | Hard reject fraud / impossible profiles | SQLite fraud KB |
| L2 | Wholesome profile↔JD similarity → 55% gate | Bi-encoder (MiniLM, INT8) |
| L3 | Seniority regression soft penalty | Rule-based |
| L4 | Work-description↔JD relevance (independent) | Bi-encoder |
| L5 | Project sanity (perplexity + buzzwords) | KenLM (or heuristic) |
| L6 | Behavioral signals (normalized) | Platform data |
| L7 | Final ranking + top-50 polish | Mamdani FIS + FlashRank |

**Gate:** top 50% by L2 + random 5% = 55% proceed past L2.
**Tie-break:** higher experience → older company → candidate_id ascending.

---

## Repository Structure

```
redrob/
├── rank.py               # Entry point (offline ranking)
├── app.py                # Streamlit UI
├── build_kb.py           # One-time: download/quantize/compress models + build KB
├── setup.sh              # Decompress artifacts after clone
├── setup_check.py        # Pre-flight verification
├── requirements.txt
├── submission_metadata.yaml
├── src/
│   ├── jd_parser.py      # Local Ollama JD parsing → jd.json
│   ├── pruning.py        # Heuristic folder pruning + timed thread dispatch
│   ├── layers.py         # L1–L6 cascade
│   ├── fis.py            # L7 Mamdani FIS + FlashRank polish + reasoning
│   ├── utils.py          # Model loading (offline-first, graceful fallback)
│   └── constants.py      # All weights, thresholds, paths
├── models/
│   ├── compressed/       # Git-tracked .tar.gz (via Git LFS)
│   └── decompressed/     # Created by setup.sh (git-ignored)
├── data/                 # candidates.jsonl, job_description.pdf, jd.json (git-ignored)
└── tests/
    └── test_pipeline.py  # 16 tests: JD parser, honeypots, layers, models
```

---

## Testing

```bash
python tests/test_pipeline.py      # built-in runner
# or
python -m pytest tests/ -v
```

Covers: JD parser wiring, all 4 honeypot types, every layer in isolation,
folder pruning, FIS ranking, reasoning quality, and graceful model loading.

---

## Compute Budget (75K candidates)

| Constraint | Limit | Actual |
|------------|-------|--------|
| Runtime | 5 min | ~3:25 |
| RAM | 16 GB | ~1.7 GB peak |
| Disk | 5 GB | ~1.1 GB |
| GPU | none | CPU only |
| Network (ranking) | off | zero calls |

---

## Notes

- Designed and tested on a Lenovo ThinkBook (16 GB RAM, CPU-only, Python 3.11+).
- All model loads degrade gracefully: missing KenLM → heuristic perplexity;
  missing fraud KB → in-code blacklist; missing FlashRank → FIS order kept.
- `build_kb.py` is the only step that needs internet, and it runs once.
