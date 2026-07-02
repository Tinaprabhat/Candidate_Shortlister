---
title: RedRob Ranker
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# RedRob — Intelligent Candidate Discovery & Ranking System

> **Redrob Hackathon Submission** · Team RedRob · India Runs Data & AI Challenge 2026

A fully offline, CPU-only candidate ranking pipeline that processes up to 100K candidate profiles against a structured Job Description and produces a ranked shortlist in **~51 seconds** — with zero external API calls at inference time.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
  - [Pre-Processing](#pre-processing)
  - [Folder Pruning](#folder-pruning)
  - [L1 — Fraud & Hard Reject](#l1--fraud--hard-reject)
  - [L1b — Profile Integrity](#l1b--profile-integrity)
  - [L1c — Skill Match](#l1c--skill-match)
  - [L2 — Feature Table Extraction](#l2--feature-table-extraction)
  - [L3 — Sugeno Fuzzy Scoring](#l3--sugeno-fuzzy-scoring)
  - [L4 — Semantic Work Relevance](#l4--semantic-work-relevance)
  - [L5a — Don'ts Penalty](#l5a--donts-penalty)
  - [FIS — Mamdani Fuzzy Final Score](#fis--mamdani-fuzzy-final-score)
- [Backend API](#backend-api)
- [Frontend Dashboard](#frontend-dashboard)
- [Models & Artifacts](#models--artifacts)
- [Performance Benchmarks](#performance-benchmarks)
- [Team](#team)

---

## Overview

RedRob is a **multi-layer cascade ranker** built for real-world recruiting at scale. The system avoids the common trap of keyword stuffing by combining:

- **Rule-based hard filters** to eliminate fraud and impossible profiles
- **NLP + synonym expansion** for skill matching beyond exact keywords
- **Fuzzy logic** (Sugeno + Mamdani) to score nuanced signal combinations
- **Semantic embeddings** (`all-MiniLM-L6-v2`) for work-history-to-JD relevance

The system is designed to work entirely **offline** — `rank.py` makes zero network calls and runs on CPU-only hardware within the hackathon's 5-minute / 16 GB RAM compute budget.

### Compute Constraints Met

| Constraint | Budget | Achieved |
|---|---|---|
| Runtime | ≤ 300s | ~51s on 100K candidates |
| Memory | ≤ 16 GB RAM | ~350 MB peak |
| Disk | ≤ 5 GB | ~233 MB intermediate |
| GPU | None | CPU only |
| Network (during ranking) | None | Zero external calls |

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
│   ├── fis.py                       # Mamdani FIS final scoring
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
    ├── modular_test.py                  # Per-layer isolation tests
    └── test_layer.py                    # Full layer unit tests (pytest, no models needed)
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/Tinaprabhat/Candidate_Shortlister.git
cd Candidate_Shortlister
pip install -r requirements.txt
bash setup.sh          # decompresses models from models/compressed/
```

### 2. Job Description *(already done — no action needed)*

`data/jd.json` was **prepared manually** by reading the job description and hand-authoring the structured JSON. No LLM, API, or automated parser was used at any step of this project — including JD parsing, candidate scoring, and ranking. The file is committed to the repo; nothing needs to be generated on a fresh clone.

### 3. Add the Candidates File

Place the `candidates.jsonl` file provided for this stage into the `data/` folder:

```
data/candidates.jsonl
```

(This file is intentionally not committed to the repo — it's the dataset provided by the hackathon at evaluation time.)

### 4. Run the Ranker

```bash
# From a flat JSONL file
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# From a pre-built ZIP with folder hierarchy
python rank.py --zip ./data/candidates.zip --out ./submission.csv
```

Produces `submission.csv` with 100 ranked candidates:

```
candidate_id, rank, score, reasoning
CAND_0042871, 1, 0.987, "Senior AI Engineer with 7 years..."
...
```

### 5. Start the Dashboard *(optional)*

```bash
# Terminal 1 — Backend API
uvicorn backend.api_server:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — Frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`

### 6. Sandbox Demo (Docker, single container)

The command center dashboard (frontend + backend) is also packaged as a single public
Docker image — this is the hackathon's sandbox link, satisfying the "sandbox platform"
requirement via the docker escape-hatch in the submission spec:

```bash
docker pull abinmukherjee/redrob-sandbox:latest
docker run -p 8000:8000 abinmukherjee/redrob-sandbox:latest
```

Open `http://localhost:8000`, upload a `candidates.jsonl` sample (≤100 rows, ≤5 MB —
one candidate profile JSON per line), and the container runs the real offline `rank.py`
pipeline against the bundled job description end-to-end, then displays the ranked
shortlist in the dashboard. No network calls during ranking; models are baked into the
image at build time from `models/compressed/`.

To rebuild the image locally instead of pulling:

```bash
docker build -t redrob-sandbox .
docker run -p 8000:8000 redrob-sandbox
```

### 7. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Pipeline Architecture

The ranking cascade runs in two phases: an **early cascade** parallelised per folder bucket, and a **late cascade** applied globally on survivors.

```
candidates.jsonl / .zip
         │
         ▼
 ┌─────────────────────┐
 │   Pre-Processing    │  pipeline_1.py — clean, classify, build folder tree
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │   Folder Pruning    │  Phase 1a–1e (rules) + Phase 2 (token overlap)
 │   (pruning.py)      │  Eliminates irrelevant buckets before any model runs
 └────────┬────────────┘
          │  Parallel threads, one per surviving folder bucket
          ▼
 ┌──────────────────────────────────────────────────────┐
 │             EARLY CASCADE  (per folder)              │
 │                                                      │
 │  L1    Fraud & Hard Reject  ──────────→  knockout    │
 │  L1b   Profile Integrity    ──────────→  soft flags  │
 │  L1c   Skill Match (NLP)    ──────────→  score [0–1] │
 │                                                      │
 │  Gate: top 50% + random 25% = 75% pass forward      │
 └────────────────────┬─────────────────────────────────┘
                      │  Merge survivors from all folder threads
                      ▼
 ┌──────────────────────────────────────────────────────┐
 │             LATE CASCADE  (global)                   │
 │                                                      │
 │  L2    Feature Table (31 cols)                       │
 │  L3    Sugeno Fuzzy Scoring   ────────→  l3_score    │
 │  L4    Semantic Relevance     ────────→  top 200     │
 │  L5a   Don'ts Penalty         ────────→  top 100     │
 │                                                      │
 │  FIS   Mamdani Final Score    ────────→  submission  │
 └──────────────────────────────────────────────────────┘
```

---

### Pre-Processing

**File:** `data/preprocessing/pipeline_1.py`

Ingests raw `candidates.jsonl` and classifies each profile into a structured folder tree:

```
<code|no_code> / <available|unavailable> / <0-3|4-9|10+> / <role_domain> / <bucket>.json
```

This tree is the foundation for heuristic folder pruning — entire irrelevant branches are dropped before any ML model is loaded.

---

### Folder Pruning

**File:** `pipeline/pruning.py`

Five deterministic pruning phases — **no AI model used**:

| Phase | Signal | What Gets Dropped |
|---|---|---|
| 1a — Code/No-code | JD code requirement | Buckets contradicting JD's need for a coder or non-coder |
| 1b — Inactive | Folder name label | Buckets flagged as unavailable candidates |
| 1c — Location | JD city vs. folder city | Buckets outside the JD's target location |
| 1d — Experience | JD minimum years | Buckets whose experience ceiling is below JD minimum |
| 1e — Role | JD role family | Buckets labelled as an unrelated role |
| 2 — Token Overlap | JD title + skills | Scores survivors by relevance; sorts for dispatch priority |

After pruning, surviving folders are dispatched to the early cascade in **parallel threads** (configurable stagger via `--stagger`).

---

### L1 — Fraud & Hard Reject

**File:** `pipeline/layers.py` → `l1_hard_reject()`

Eliminates candidates with mathematically impossible or fraudulent profiles using a **SQLite knowledge base** (`fraud_kb.db`) with fuzzy name matching (`rapidfuzz`, threshold: 85):

- Work experience at a company that was founded *after* the stated start date
- Education timelines that impossibly overlap with full-time employment
- Fictional company names (Dunder Mifflin, Hooli, Initech, Pied Piper, etc.) — in-memory blacklist + KB lookup
- Salary ranges where `min > max` (inverted)
- Expert-level skill proficiency claimed with 0 months of usage

---

### L1b — Profile Integrity

**File:** `pipeline/layers.py` → `l1b_profile_integrity()`

Applies **soft flags** (not hard rejects) to profiles with integrity concerns:

- Profile completeness below threshold
- Suspicious career history gap patterns
- Endorsement counts inconsistent with claimed proficiency level
- `open_to_work_flag = False`

Flagged candidates carry a penalty multiplier forward into L3 and are excluded from the top 100 if enough clean candidates exist.

---

### L1c — Skill Match

**File:** `pipeline/layers.py` → `l1c_skill_match()`

Scores each candidate's skills against JD requirements using:

- **Synonym expansion** (`pipeline/dictionary.py`) — maps aliases (`ML` → `machine learning`, `k8s` → `kubernetes`, etc.)
- **Proficiency weighting** — `expert` > `advanced` > `intermediate` > `beginner`
- **Endorsement & duration trust multiplier** — higher endorsements + months used → higher signal weight

Produces `l1c_score ∈ [0, 1]`. Candidates below the floor threshold are hard-rejected at this stage.

**Gate:** Top 50% by `l1c_score` pass forward, plus a random 25% from the bottom half (exploration insurance). **Total: 75% pass rate.**

---

### L2 — Feature Table Extraction

**File:** `pipeline/layers.py` → `l2_extract_features()`

Extracts **31 structured columns** from each surviving candidate profile for downstream scoring:

- Experience years, current title, company size, industry
- Skill count, endorsement totals, platform assessment scores
- Behavioral signals: `recruiter_response_rate`, `interview_completion_rate`, `offer_acceptance_rate`
- Platform signals: `github_activity_score`, `profile_completeness_score`, `open_to_work_flag`
- Notice period, salary range fit, willingness to relocate

These columns feed directly into L3's fuzzy membership functions.

---

### L3 — Sugeno Fuzzy Scoring

**File:** `pipeline/layers.py` → `l3_sugeno_score()`

A **Sugeno-type fuzzy inference system** evaluating eight conditions against the JD:

| Condition | Signal |
|---|---|
| a | Years of experience vs. JD requirement |
| b | Seniority match (title level vs. JD level) |
| c | Industry relevance |
| d | Company size match |
| e | Notice period acceptability |
| f | Salary range fit |
| g | Location / relocation alignment |
| h | Profile activity recency (`last_active_date`) |

Outputs `l3_score` as a weighted Sugeno aggregate. Significant mismatches apply a `0.85×` seniority penalty multiplier that carries into the FIS.

---

### L4 — Semantic Work Relevance

**File:** `pipeline/layers.py` → `l4_semantic_score()`

**Model:** `all-MiniLM-L6-v2` (INT8 ONNX, loaded from `models/decompressed/sentence_transformer/`)

Encodes each candidate's **work history descriptions** and computes cosine similarity against the JD embedding:

- Embedding dimension: 384
- Batch size: 64 candidates per forward pass
- Produces `l4_combined_score` — weighted blend of top-3 role cosine similarities

Top **200 candidates** by `l1c × l4` combined score pass to L5.

---

### L5a — Don'ts Penalty

**File:** `pipeline/layers.py` → `l5a_donts_penalty()`

Applies a **high-weight penalty** for explicit disqualifiers in the JD:

- Excluded titles (e.g. "no pure researchers")
- Excluded industries
- Skill anti-patterns (e.g. "must not have only academic project experience")

Surviving candidates are narrowed to the **top 100**.

---

### FIS — Mamdani Fuzzy Final Score

**File:** `pipeline/fis.py`

A **Mamdani fuzzy inference system** combining `l1c`, `l3`, `l4`, and `l6` (behavioral) signals via triangular membership functions:

```
LOW  → triangle(x,  -0.1,  0.0,  0.5)
MED  → triangle(x,   0.2,  0.5,  0.8)
HIGH → triangle(x,   0.5,  1.0,  1.1)
```

Assigns each candidate a tier before final ranking:

| Tier | Criteria | Effect |
|---|---|---|
| `very_good` | Meets ALL excellence thresholds across signals | Guaranteed top-10 placement |
| `eligible` | No flags, no L3 penalty | Ranked by FIS composite score |
| `penalized` | Has L1b flags OR L3 seniority penalty | Excluded from top 100 if 100 clean candidates exist |

The FIS output is the `score` column written to `submission.csv`.

---

## Backend API

**File:** `backend/api_server.py` · FastAPI · Default port: **8000**

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/healthz` | None | Health check — always public |
| `GET` | `/candidates` | Bearer | List all ranked candidates (paginated) |
| `GET` | `/candidates/{id}` | Bearer | Full profile + per-layer scores |
| `POST` | `/candidates/{id}/schedule-interview` | Bearer | Flag a candidate for interview |
| `GET` | `/pipeline/runs` | Bearer | List past ranking run metadata |
| `POST` | `/pipeline/runs` | Bearer | Trigger a new `rank.py` run (async subprocess) |
| `GET` | `/pipeline/runs/{run_id}/funnel` | Bearer | Layer-by-layer funnel counts for a run |
| `GET` | `/system/health` | Bearer | Model status, disk usage, runtime info |
| `GET` | `/system/events` | Bearer | Live log stream for the active run |

**Auth:** Bearer token header. Set `API_REQUIRE_AUTH=true` to enforce; accepts any token in local preview mode.

**CORS:** Pre-configured for Vite dev server ports `5173`, `5174`, `4173`.

The backend manages `rank.py` as a controlled subprocess, streams stdout logs in real time to `/system/events`, and caches the latest ranked output for the frontend to consume without re-running.

---

## Frontend Dashboard

**Stack:** React 18 · Vite · Tailwind CSS · Zustand · Three.js

**Directory:** `frontend/src/`

### Pages

| Page | Route | Purpose |
|---|---|---|
| `Overview` | `/` | KPI strip, funnel chart, risk cards summary |
| `RankedCandidates` | `/candidates` | Sortable, filterable ranked candidate table |
| `CandidateDetail` | `/candidates/:id` | Full profile view + per-layer score breakdown |
| `SystemHealth` | `/system` | Model status, pipeline timing, live log stream |
| `LoginPage` | `/login` | Auth gate |

### Key Components

| Component | Description |
|---|---|
| `EvidenceDrawer` | Slide-in panel showing per-layer scoring evidence for a candidate |
| `FunnelChart` | Visual layer-by-layer candidate count drop (L1 → FIS) |
| `LogicMatrix` | Displays FIS rule activations for a selected candidate |
| `ScoreOrb3D` | Three.js animated orb visualising the composite score |
| `FilterRail` | Multi-facet filter sidebar (tier, score range, skills, location) |
| `KPIStrip` | Live stats: total processed, pass rates, top-tier count |

---

## Models & Artifacts

All models are stored compressed in `models/compressed/` (tracked via **Git LFS**) and decompressed on first use by `setup.sh` into `models/decompressed/`.

| Artifact | Size | Purpose | Used in Layer |
|---|---|---|---|
| `sentence_transformer.tar.gz` | 173 MB | `all-MiniLM-L6-v2` bi-encoder | L4 |
| `spacy_model.tar.gz` | 13 MB | `en_core_web_sm` NLP | L1b, L5a |
| `fraud_kb.tar.gz` | 68 KB | SQLite company / institution KB | L1 |

To rebuild the fraud knowledge base from raw sources:

```bash
python pipeline/scripts/build_kb.py
```

To verify all models are correctly decompressed:

```bash
python pipeline/scripts/setup_check.py
```

---

## Performance Benchmarks

Measured on **Lenovo ThinkBook · 16 GB RAM · CPU-only · Windows 11 · Python 3.11**

| Stage | Time |
|---|---|
| Pre-processing (`pipeline_1.py`) | ~8s |
| Folder pruning | <1s |
| Early cascade (L1 → L1c, all folders, parallel) | ~12s |
| L4 semantic scoring (MiniLM bi-encoder) | ~22s |
| FIS + final ranking | <1s |
| **Total end-to-end (100K candidates)** | **~51s** |

---

## Team

| Member | Role |
|---|---|
| **Abin Mukherjee** | Integration, Deployment, Backend API |
| **Sreshtho** | Frontend Dashboard (React Command Center) |
| **Rishi** | Data Handling (cleaning pipeline, dataset preprocessing) |
| **Tina** | AI Logic (cascade layers, FIS, JD parser) |

---

*Built for the India Runs Data & AI Challenge 2026 · Redrob Hackathon*
