# RedRob — Offline Candidate Shortlisting Engine

> A 4-layer cascading pipeline that ranks thousands of candidate profiles against a job description — fully offline, no LLM API calls, sub-4GB memory footprint.

[![Live Demo](https://img.shields.io/badge/demo-docker-blue?logo=docker)](https://hub.docker.com/r/abinmukherjee/redrob-sandbox)
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

| Metric | Value |
|---|---|
| **Avg. run time** | ~200s (10k candidates → top 100) |
| **Peak memory** | 4.2 GB |
| **Output** | Top 100 ranked candidates |
| **Models used** | 100% local — zero API calls |
| **Model footprint** | ~186 MB (compressed) |

---

## Table of Contents

1. [Overview](#overview)
2. [Live Demo](#live-demo)
3. [Architecture](#architecture)
4. [Pipeline Layers](#pipeline-layers)
5. [Repository Structure](#repository-structure)
6. [Setup & Run](#setup--run)
7. [Usage](#usage)
8. [Output Format](#output-format)
9. [Performance](#performance)
10. [Tech Stack](#tech-stack)
11. [Known Limitations](#known-limitations)

---

## Overview

RedRob ingests a raw pool of candidate profiles (JSONL or a ZIP of nested folders) and a structured job description, then runs them through a **4-stage cascade** — hard filters → integrity checks → skill matching → semantic relevance — producing a ranked `submission.csv` of the top 100 candidates with full score breakdowns and reasoning.

**Design goals:**
- **Fully offline** — no OpenAI/Anthropic/cloud calls anywhere in the ranking path. Every model runs locally.
- **Cascading cost control** — cheap, high-recall filters run first (L1) to shrink the pool before expensive semantic scoring (L4) runs on survivors only.
- **Concurrent, streamed processing** — candidates are scored per-folder, concurrently, rather than loaded entirely into memory upfront.
- **Fully traceable** — every layer's decisions (kept/dropped/penalized) are logged via `RunTracer` for auditability.

---

## Live Demo

Run the full system (backend + dashboard) locally with a single command:

```bash
docker run -p 8000:8000 abinmukherjee/redrob-sandbox:latest
```

Then open `http://localhost:8000` to explore the React command-center dashboard — ranked candidates, score breakdowns, and system health, all served from the FastAPI backend.

---

## Architecture

```
                         ┌─────────────────────────┐
                         │   candidates.jsonl / .zip │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  Preprocessing (pipeline_1)│
                         │  clean · fabrication flags │
                         │  → folder tree             │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   Folder Discovery         │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │  Heuristic Pruning         │
                         │  (Phase 1a–1e + Phase 2)   │
                         │  drop low-signal folders    │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │ Staggered Folder Dispatch  │
                         │  (concurrent workers)      │
                         └────────────┬─────────────┘
                                      │
        ╔═════════════════════════════▼═════════════════════════════╗
        ║      EARLY CASCADE  (per-folder, streamed per-candidate)   ║
        ║                                                             ║
        ║   L1a → Hard Reject         (disqualifying criteria)        ║
        ║   L1b → Profile Integrity   (fabrication / fraud flags)     ║
        ║   L1c → Skill Match         (explicit + required skills)    ║
        ║   L1d → Inferred Match      (synonym/dictionary expansion)  ║
        ║   L2  → Table Extraction    (structured fields)             ║
        ║   L3  → Weighted Score      (fuzzy score + reasoning)       ║
        ╚═════════════════════════════╤═════════════════════════════╝
                                      │
                         ┌────────────▼─────────────┐
                         │  Hard Pool Cap             │
                         │  (sort by L1c, cap size)   │
                         └────────────┬─────────────┘
                                      │
        ╔═════════════════════════════▼═════════════════════════════╗
        ║        LATE CASCADE  (global, all survivors)                ║
        ║                                                             ║
        ║   L4  → Semantic Work Relevance + "Don'ts" Penalty          ║
        ║          (sentence-transformer embedding similarity)        ║
        ║   L4b → Graduated Penalty                                    ║
        ║          (< 4 matched explicit-required skills)             ║
        ╚═════════════════════════════╤═════════════════════════════╝
                                      │
                         ┌────────────▼─────────────┐
                         │   Top 100 by Final Score   │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                                    ▼
          ┌───────────────────┐              ┌───────────────────────┐
          │  submission.csv     │              │  ranked_{run_id}.json  │
          │  candidate_id,rank,  │              │  full score breakdown  │
          │  score, reasoning    │              │  per candidate         │
          └───────────────────┘              └───────────────────────┘
```

---

## Pipeline Layers

| Layer | Name | Purpose | Cost |
|---|---|---|---|
| **L1a** | Hard Reject | Disqualifies candidates on non-negotiable criteria | Cheapest — rule-based |
| **L1b** | Profile Integrity | Flags fabricated/inconsistent profiles, applies penalty | Cheap — heuristic |
| **L1c** | Skill Match | Matches explicit + required skills from JD | Cheap — string/set match |
| **L1d** | Inferred Match | Expands matches via synonym dictionary (`dictionary.py`) | Cheap — lookup table |
| **L2** | Table Extraction | Pulls structured fields (experience, education, etc.) | Moderate |
| **L3** | Weighted Score | Combines L1–L2 signals into a fuzzy score + reasoning string | Moderate |
| **L4** | Semantic Work Relevance | Embeds candidate work history + JD via `all-MiniLM-L6-v2`, scores cosine similarity; applies "don'ts" penalty | Most expensive — only runs on L3 survivors |
| **L4b** | Explicit Requirement Penalty | Graduated penalty for candidates matching < 4 explicit required skills; re-sorts final ranking | Cheap |

**Why cascade instead of scoring everyone with the expensive model?**
Running a sentence-transformer over every candidate in a 10k+ pool is wasteful when 60–80% can be eliminated by free rule-based checks first. L1a–L1d filter the pool cheaply; L4's semantic embedding only runs on candidates that already passed integrity and skill-match gates.

---

## Repository Structure

```
Candidate_Shortlister/
│
├── rank.py                          # ← Single entry point. Produces submission.csv
├── requirements.txt                 # All Python dependencies
├── setup.sh                         # Decompresses models, installs spaCy
├── submission_metadata.yaml         # Hackathon portal metadata
│
├── pipeline/                        # All AI ranking logic
│   ├── layers.py                    # Cascade layers L1 → L4
│   ├── pruning.py                   # Heuristic folder pruning (Phase 1a–1e + Phase 2)
│   ├── jd_parser.py                 # JD PDF → structured JSON (utility, not used in ranking)
│   ├── constants.py                 # All weights, thresholds, model paths
│   ├── utils.py                     # Shared utilities (KB loader, text normalisation)
│   ├── dictionary.py                # Skill synonym expansion table
│   ├── tracer.py                    # Run telemetry & per-layer timing logger
│   └── scripts/
│       ├── build_kb.py              # Builds fraud_kb.db SQLite knowledge base
│       ├── rebuild_fraud_kb.py      # Rebuilds KB from raw data sources
│       ├── inspect_kb.py            # Interactive KB inspector
│       └── setup_check.py           # Verifies models and environment health
│
├── backend/
│   └── api_server.py                # FastAPI server — 9 endpoints for the dashboard
│
├── frontend/                        # React + Vite command center dashboard
│   ├── src/
│   │   ├── pages/                   # Overview, RankedCandidates, CandidateDetail, SystemHealth
│   │   ├── components/              # UI components (3D, candidates, layout, overview)
│   │   ├── api/                     # Typed API client wired to backend
│   │   ├── hooks/                   # useCandidates, usePipeline, useSystemHealth
│   │   └── store/                   # Zustand global state
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── candidates.jsonl             # Runtime — provided by hackathon (gitignored)
│   ├── jd.json                      # Manually prepared structured JD — committed, no LLM used
│   ├── preprocessing/
│   │   ├── pipeline_1.py            # Cleans + classifies candidates into folder tree
│   │   └── description_frequency.py # Role description frequency analysis
│   └── scripts/
│       └── pack_dataset.py          # Converts folder tree → candidates.zip / .jsonl
│
├── models/
│   ├── compressed/                  # Tracked via Git LFS
│   │   ├── sentence_transformer.tar.gz   # all-MiniLM-L6-v2 (173 MB)
│   │   ├── spacy_model.tar.gz            # en_core_web_sm (13 MB)
│   │   └── fraud_kb.tar.gz               # SQLite knowledge base
│   └── decompressed/                # Generated by setup.sh (gitignored)
│
└── tests/
    ├── fixtures/
    │   └── honeypot_candidates.jsonl    # Synthetic fraud profiles for L1 testing
    ├── honeypot_stresstest.py           # L1 fraud detection stress tests
    ├── model_working_test.py            # Model load & graceful-degrade tests
    └── test_layer.py                    # Full layer unit tests (pytest, no models needed)
```

---

## Setup & Run

### Prerequisites
- Python 3.10+
- ~5 GB free disk space (models + working data)
- No GPU required

### 1. Clone and install

```bash
git clone https://github.com/Tinaprabhat/Candidate_Shortlister.git
cd Candidate_Shortlister
pip install -r requirements.txt
```

### 2. Decompress models + install spaCy

```bash
bash setup.sh
```

This one-time step decompresses `models/compressed/*.tar.gz` into `models/decompressed/` and installs the spaCy language model. Subsequent runs are a no-op.

### 3. Verify environment

```bash
python -m pipeline.scripts.setup_check
```

### 4. Prepare the job description (one-time)

```bash
python -m pipeline.jd_parser --jd ./data/job_description.pdf --out ./data/jd.json
```

> `data/jd.json` must exist before running `rank.py`. It is manually structured — no LLM is used for JD parsing.

---

## Usage

### Flat candidate list

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

### Nested folder ZIP

```bash
python rank.py --zip ./data/candidates.zip --out ./submission.csv
```

### Full flag reference

| Flag | Description | Default |
|---|---|---|
| `--candidates` | Path to flat `candidates.jsonl` | — |
| `--zip` | Path to candidates ZIP (folder hierarchy) | — |
| `--jd` | Path to structured `jd.json` | `data/jd.json` |
| `--out` | Output CSV path | `./submission.csv` |
| `--stagger` | Seconds between staggered folder dispatch | `FOLDER_DISPATCH_STAGGER_SEC` |

Either `--candidates` or `--zip` is required.

---

## Output Format

### `submission.csv`

```csv
candidate_id,rank,score,reasoning
CAND_04821,1,0.947231,"Matched 6/6 required skills; 4.2 yrs relevant experience"
CAND_01193,2,0.931005,"Matched 5/6 required skills; strong semantic work-history overlap"
...
```

### `output/ranked_{run_id}.json`

Full per-candidate score breakdown for audit and debugging:

```json
{
  "run_id": "20260702_143012",
  "jd_title": "Senior Backend Engineer",
  "total_ranked": 100,
  "candidates": [
    {
      "rank": 1,
      "candidate_id": "CAND_04821",
      "final_score": 0.947231,
      "reasoning": "Matched 6/6 required skills...",
      "score_breakdown": {
        "l1c_skill_match": 0.92,
        "l1c_matched_required": ["Python", "Kubernetes", "..."],
        "l1c_missing_required": [],
        "l1b_integrity_penalty": 1.0,
        "l3_fuzzy_score": 0.88,
        "l4_work_relevance": 0.91,
        "l4_donts_penalty": 0.0,
        "l4_combined_score": 0.947231,
        "l4b_explicit_req_penalty": 0.0
      },
      "experience_years": 4.2,
      "skills_snippet": "Python, FastAPI, Kubernetes, PostgreSQL..."
    }
  ]
}
```

A live end-of-run timing summary is also printed to stdout, breaking down elapsed time per layer.

---

## Performance

Benchmarked on a 10,000-candidate pool, CPU-only:

| Stage | Time |
|---|---|
| Preprocessing + folder discovery | ~15s |
| Heuristic pruning | ~2s |
| Early cascade (L1a → L3, streamed, concurrent) | ~140s |
| Late cascade (L4 semantic + L4b) | ~35s |
| Output write | ~3s |
| **Total** | **~200s** |

| Resource | Value |
|---|---|
| Peak memory | 4.2 GB |
| Model footprint (compressed) | ~186 MB |
| GPU required | No |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Ranking pipeline** | Python, concurrent streaming cascade |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| **NLP** | spaCy (`en_core_web_sm`) |
| **Fraud/integrity KB** | SQLite |
| **Backend API** | FastAPI (9 endpoints) |
| **Dashboard** | React + Vite, Zustand state, 3D visualization components |
| **Telemetry** | Custom `RunTracer` — per-layer timing + decision audit trail |
| **Testing** | pytest, honeypot fraud-detection stress tests |

---

## Known Limitations

| Limitation | Impact | Notes |
|---|---|---|
| `jd.json` is manually structured | No automated JD parsing in the ranking path | `jd_parser.py` exists as a utility but is not called by `rank.py` |
| Hard pool cap before L2/L3 | Very large pools (100k+) are capped and sorted by L1c proxy before expensive layers | See `constants.HARD_POOL_CAP` |
| Synonym dictionary is static | Skill matching quality depends on `dictionary.py` coverage | Manually maintained, not learned |
| No GPU acceleration path | Embedding step is CPU-bound | Acceptable at current scale (~200s / 10k candidates) |

---

## Testing

```bash
# Fast unit tests — no models required
pytest tests/test_layer.py

# Model load / graceful-degrade checks
pytest tests/model_working_test.py

# Fraud detection stress test (L1b)
python tests/honeypot_stresstest.py
```

---

## License

MIT License

## Author

**Tina Prabhat**
B.Tech CSE — KIIT University
[GitHub](https://github.com/Tinaprabhat)
