# RedRob Candidate Ranking Cascade

Offline, single-command pipeline that takes a pool of candidate JSON records and a parsed
job description (`jd.json`) and produces a **ranked Top-100 shortlist** with per-layer
scores, audit flags, and a human-readable reasoning string for every finalist.

```
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python rank.py --zip ./data/candidates.zip --out ./submission.csv
```

**Budget:** ~90s average wall-clock, <1GB peak memory, output capped at Top-100.

---

## 1. Pipeline Overview

```
                 ┌────────────────────────────────────────────────────────┐
                 │   STREAMING CASCADE (per-candidate, concurrent)         │
  raw candidates │   L1a → L1b → L1c → L1d → L2 → L3                       │
  ──────────────▶│                                                        │──▶ survivors
                 └────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌────────────────────────────────────────────────────────┐
                 │   POST-CASCADE (global, batch)                          │
                 │   L4 (semantic + donts)  →  L4b (skill-coverage penalty) │──▶ Top 100
                 └────────────────────────────────────────────────────────┘
```

Every candidate is pushed through a `ThreadPoolExecutor` and streams through
**L1a → L1b → L1c → L1d → L2 → L3 with no batch wait between stages** — candidate #7 can
be at L1c while candidate #2 is already at L3. The only synchronization point is the
`gather` at the end of `run_streaming_cascade()`, after which all survivors are handed to
L4 as one batch (L4 needs the full pool to normalize embeddings/percentiles).

`L5 (FlashRank cross-encoder)` exists in code but is **currently disabled** — the Top-100
from L4/L4b is the final ranking, no re-shuffle.

---

## 2. Layer-by-Layer Reference

| Layer | Name | Type | Can reject? | Approx. time* |
|---|---|---|---|---|
| L1a | Fraud KB (math + KB verification) | Knockout | ✅ Yes | ~15–20s |
| L1b | Profile integrity (ATS flags) | Knockout | ✅ Yes | <1s |
| L1c | Explicit skill match (NLP + synonyms) | Knockout + score | ✅ Yes | ~2–5s |
| L1d | Inferred skill match | Score only | ❌ No | ~1–3s |
| L2 | Table extract (31 columns) | Feature build | ❌ No | ~1–2s |
| L3 | Sugeno fuzzy inference (conditions a–h) | Score only | ❌ No | ~1–2s |
| L4 | Semantic work relevance + donts penalty | Score + reject | ✅ Yes (near-zero relevance / negative score) | ~30–40s |
| L4b | Explicit-requirement coverage penalty | Score adjustment | ❌ No | <1s |
| **Top-100 cut** | Sort by `candidate_final_score` | Selection | — | <1s |
| L5 | FlashRank cross-encoder | — | — | **disabled** |

*Times are indicative shares of the ~90s average run and scale with pool size and model
load time (embedding model + KB load account for a large fixed cost regardless of pool
size).

---

### L1a — Fraud KB (Hard Reject)

**Purpose:** catch fabricated or mathematically impossible profiles before spending any
compute on them — this is the **honeypot / fraud detection gate**.

Two parts:

1. **Mathematical consistency** — pre-computed honeypot and salary-inversion flags,
   degree-order sanity (PhD before Bachelor's, etc.), impossible certification dates
   (e.g., a LangChain cert dated before 2022), graduation-vs-experience overflow,
   future end-dates, work claimed before a company's founding year, senior titles
   claimed pre-graduation, overlapping full-time jobs, and experience exceeding an
   age-derived ceiling.
2. **KB verification** — fuzzy-matches (RapidFuzz, threshold 85) each company, university,
   and claimed publication against a local SQLite fraud knowledge base
   (`fraud_kb.sqlite`, built by `build_kb.py`):
   - `fictional_companies` — instant reject (also backed by an in-code blacklist:
     Hooli, Initech, Stark Industries, Wayne Enterprises, etc.)
   - `indian_companies` / `indian_startups` / `global_companies` — legitimate-company
     verification (MCA, DPIIT, PDL sources)
   - `universities` tables — institution verification
   - `research_papers` — authorship cross-check; a title match with a non-matching
     author → **reject** (`paper_authorship_mismatch`)

**Scoring:** each verifiable entity scores `1.0` (verified) / `0.5` (unknown/out-of-KB) /
`0.0`→reject (contradicts). `l1_score` = mean of all entity scores (defaults to `0.5` if
nothing to verify).

**Reject condition:** any hard-fail check above → candidate is removed, no further layers run.

---

### L1b — Profile Integrity

**Purpose:** structural ATS red flags and platform opt-outs, pre-computed upstream.

**Reject conditions** (any one fires → removed):
- `reverse_degree_order`, `duplicate_job_descriptions` (top-level ATS flags)
- `invalid_degree_field_combination` on any education entry
- `redrob_signals.willing_to_relocate is False`
- `redrob_signals.open_to_work_flag is False`

Survivors get `l1b_penalty = 1.0`. Softer integrity flags (fabrication bandwidth, low
engagement, etc.) are **not** rejected here — they are carried into the L2 table and folded
into condition **h** at L3.

---

### L1c — Explicit Skill Match (NLP + Synonym)

**Purpose:** does the candidate actually claim the JD's required/bonus skills, anywhere in
their skills list, work descriptions, profile summary, or project text?

**Method:** every JD skill is expanded through a synonym dictionary (`expand_skill()`), then
matched as a whole word/phrase (regex boundary match, not substring) against a normalized
candidate text blob.

**Score:**
```
raw   = 0.7 × |matched_required| + 0.3 × |matched_bonus| − 0.2 × |unmatched_candidate_skills|
l1c_score = clamp(raw / (0.7 × n_required + 0.3 × n_bonus), 0, 1)
```

**Reject condition:** if the JD has explicit required skills and the candidate matches
**zero** of them → hard reject. An optional global score-floor gate exists
(`C.L1C_MIN_SKILL_MATCH`, default off).

Also computed here: `l1c_explicit_proficiency_score` = `0.3 × skills-assessment score +
0.7 × self-reported proficiency` (beginner=0.1 … expert=0.4), restricted to JD-relevant skills.

---

### L1d — Inferred Skill Match (soft, never rejects)

**Purpose:** credit implicit/adjacent skills the JD didn't ask for explicitly but that
matter (e.g., tooling implied by a required framework).

```
inferred_ratio   = matched_inferred / n_inferred          (1.0 if JD has none)
leftover_penalty = 0.01 × (n_inferred − matched_inferred)
l1d_score        = max(0, inferred_ratio − leftover_penalty)
```

Never removes a candidate — purely a scoring input forwarded to L2/L3.

---

### L2 — Table Extract (31 columns)

**Purpose:** pure feature engineering. Consolidates everything computed so far (L1a–L1d),
plus derived fields (experience years, tenure, production/architecture/testing signals,
tool usage, open-source/research indicators, and the L1b soft-penalty flags) into one
`table_row` dict per candidate. **Never filters** — every candidate gets a row, which is
the direct input to L3's fuzzy system.

---

### L3 — Sugeno Fuzzy Inference System

**Purpose:** the core relevance scorer. Reduces the 31-column table into eight crisp
conditions `a`–`h` (each in `[0,1]`), then combines them with fixed Sugeno weights.

| Cond. | Meaning |
|---|---|
| a | Experience-in-sweet-spot signal (years, title seniority match, notice period) |
| b | Explicit skill signal — `0.8 × l1c_score + 0.2 × explicit_proficiency` |
| c | Inferred skill signal — `0.8 × l1d_score + 0.2 × inferred_proficiency` |
| d | Domain-specific term-hit signal (e.g., IR-domain keywords) |
| e | Absence of disqualifying traits (PhD-only, consulting-only, stagnant tenure, pure-researcher); penalized further if framework-heavy with no production evidence |
| f | Platform engagement signal (`redrob_cumulative`, min-max normalized across the batch) |
| g | Technical breadth — production + architecture + testing/eval + tools + open-source/research |
| h | Soft-penalty union (fabrication bandwidth + soft-penalty score); `0` = clean |

**Final formula:**
```
l3_score = 0.40·g + 0.20·b + 0.10·c + 0.10·f + 0.05·a + 0.05·d + 0.05·e − 0.05·h
```
clamped to `[0, 1]`. Classified into `strong_fit` (≥0.85) / `good_fit` (≥0.70) /
`moderate_fit` (≥0.55) / `weak_fit` (below). **Never rejects** — every candidate with a
`table_row` gets scored; only L1 removes candidates from the pool.

> **Note:** module-level docstrings in `layers.py` reference a "75% FIS gate" (top-50%
> plus a random 25% sample by `l3_score`) between L3 and L4. In the current `rank.py`
> entry point this gate is **not invoked** — all L3 survivors are passed straight to L4.
> Treat the gate as a documented-but-currently-inactive code path, not live behavior.

---

### L4 — Semantic Work Relevance + "Don'ts" Penalty

**Purpose:** does the candidate's actual work history read as relevant to what the JD says
the role *does* (`responsibilities`/`what_you'll_do` text) — and does it match anything the
JD explicitly says it does **not** want (`donts`)?

**Model:** `all-MiniLM-L6-v2` sentence-transformer, cosine similarity, embeddings computed
once per JD and batched across the candidate pool.

```
l4_work_relevance = cosine(candidate_work_text, jd_work_embedding)        ∈ [0,1]
l4_donts_sim       = cosine(candidate_title+work_text, jd_donts_embedding) ∈ [0,1]  (0 if JD has no donts)
```

**Donts penalty** — a self-calibrated ramp (noise floor = p10, full penalty = p75 of this
run's donts-similarity distribution, falling back to fixed 0.15 / 0.45 defaults for small
pools):
```
ramp           = clamp((sim − noise_floor) / (full_penalty_at − noise_floor), 0, 1)
donts_penalty  = 0.6 × ramp × sim                       (non-categorical candidates)
donts_penalty  = max(0.6 × ramp × sim, 0.06)             (CV/speech-domain candidates — floored)
donts_penalty  = 0                                       (sim below noise floor — cannot backfire)
```

**Final score:**
```
candidate_final_score = 0.5 × l3_score + 0.5 × l4_work_relevance − donts_penalty
final_score = max(0, candidate_final_score)     # clamped, never negative in output
```

**Reject conditions:**
- `l4_work_relevance < 0.05` (near-zero relevance to the role)
- resulting `candidate_final_score < 0` before clamping

---

### L4b — Explicit-Requirement Coverage Penalty

**Purpose:** final sanity check — even candidates who cleared L1c's "≥1 required skill"
gate can still be thin on explicit required-skill coverage. Anyone matching **fewer than 4**
explicit required skills is docked:
```
penalty = 0.005 × (4 − n_matched)     # capped at −0.015 (i.e. 1 match)
```
Score floor after penalty: `0.010` (never exactly zero). List is re-sorted after this step
so the Top-100 cut reflects the adjustment.

---

### Top-100 Cut

Final sort key: `(-candidate_final_score, candidate_id ascending)` — deterministic tie-break.
First 100 rows become the shortlist; rank 1…N assigned in that order.

### L5 — FlashRank Cross-Encoder (disabled)

Code retained but short-circuited (`return candidates` before the model call). If
re-enabled: `ms-marco-MiniLM-L-12-v2` cross-encodes the top-50 by `l4_combined_score`,
min-max normalizes raw logits per-batch, and recombines as
`l5_total_score = (l3_score + l4_score + flashrank_score) / 3`. Candidates outside the
top-50, or any run without FlashRank installed, fall back to `(l3_score + l4_score) / 2`.

---

## 3. Scoring Summary

| Score field | Formula | Range | Set by |
|---|---|---|---|
| `l1_score` | mean of KB entity checks | [0,1] | L1a |
| `l1c_score` | weighted skill match − unmatched penalty | [0,1] | L1c |
| `l1d_score` | inferred ratio − leftover penalty | [0,1] | L1d |
| `l3_score` | Sugeno-weighted a–h | [0,1] | L3 |
| `l4_work_relevance` | cosine(work text, JD text) | [0,1] | L4 |
| `l4_donts_penalty` | calibrated ramp penalty | [0, 0.6] | L4 |
| `candidate_final_score` | `0.5·l3 + 0.5·l4_wr − donts − l4b_penalty` | [0,1] (floored ≥0.01 post-L4b) | L4 / L4b |

---

## 4. `build_kb.py` — Offline Knowledge Base & Model Builder

Run **once**, ahead of time, to prepare everything `rank.py` loads at runtime:

- `build_sentence_transformer()` — downloads/caches `all-MiniLM-L6-v2` for L4.
- `build_flashrank()` — downloads/caches the (currently unused) L5 cross-encoder.
- `build_spacy()` — NLP pipeline used for skill-text normalization.
- `build_kenlm(order=5)` — n-gram language model (fabrication-bandwidth signal support).
- `build_fraud_kb()` — builds `fraud_kb.sqlite` with tables:
  - `fictional_companies (company_name PK, reason)`
  - `company_founding_dates (company_name PK, founding_year)`
  - `skill_aliases (skill_canonical, alias, category)`
  - plus the university/company/paper verification tables consumed by L1a.
- Checksums each artifact (`_update_checksum`) so `rank.py` can detect stale/corrupt caches.

This script is infrastructure setup, not part of the per-run scoring path — it does not
count against the 90s / <1GB runtime budget.

---

## 5. Runtime Budget

| Resource | Target |
|---|---|
| Average wall-clock (single run) | **~90 seconds** |
| Peak memory | **< 1 GB** |
| Output size | **Top 100 candidates** |
| Concurrency model | Thread pool over L1a–L3 (`C.PIPELINE_MAX_WORKERS`); L4 batches embeddings for the whole survivor pool |
| Hard pool cap | `C.HARD_POOL_CAP` candidates truncated before scoring begins (protects both time and memory budgets on oversized inputs) |

---

## 6. Output Artifacts

| File | Contents |
|---|---|
| `submission.csv` | `Cand_ID, rank, final_score, reasoning` — the graded deliverable |
| `output/ranked_<run_id>.json` | Full 21-field record per finalist: profile, education, career history, skills, projects, publications, all layer scores, flags, matched/unmatched skill lists, full L2 table row, and reasoning string |
| Run trace (via `RunTracer`) | Per-stage candidate counts, elapsed time, pruning decisions — for debugging/auditing a run |

---

