# Redrob Hackathon — Intelligent Candidate Discovery & Ranking
## Data Cleaning & Preprocessing Pipeline

> **Challenge:** Intelligent Candidate Discovery & Ranking  
> **Role being ranked for:** Senior AI Engineer — Redrob AI (Series A)  
> **Dataset:** 100,000 synthetic candidate profiles (`candidates.jsonl`)  
> **Pipeline file:** `pipeline.py`  
> **Reference date:** June 17, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Data Fixes](#data-fixes)
6. [Quality Flags](#quality-flags)
7. [Fabrication Bandwidth](#fabrication-bandwidth)
8. [Dataset Tree Structure](#dataset-tree-structure)
9. [Classification Logic](#classification-logic)
10. [Domain & Role Routing](#domain--role-routing)
11. [Computer Vision Special Routing](#computer-vision-special-routing)
12. [Engineering Domain 3-Tier Routing](#engineering-domain-3-tier-routing)
13. [Complete Field Reference](#complete-field-reference)
14. [Bugs Found & Fixed](#bugs-found--fixed)
15. [Design Decisions](#design-decisions)
16. [Threshold Reference](#threshold-reference)
17. [Tools Used](#tools-used)

---

## Overview

This pipeline takes the raw `candidates.jsonl` file and does three things in a single run:

1. **Cleans** every candidate — fixes sentinel values, detects structural impossibilities, flags fabricated content
2. **Stamps** a normalized fabrication bandwidth score on every candidate
3. **Routes** every candidate into a structured 5-dimension folder tree

**Nothing is deleted.** All 100,000 records are retained. Data quality issues are annotated as flags so the downstream ranking stage can apply soft penalties rather than hard exclusions.

```
candidates.jsonl (100K records)
        │
        ▼
   [ pipeline.py ]
        │
        ├── Pass 1: Read + Fix + Flag + Collect description frequencies
        ├── Pass 2: Stamp fabrication_bandwidth (normalized 0-1)
        └── Pass 3: Classify into 5 dimensions + Write Dataset/ tree
```

**Runtime:** Under 60 seconds on a single CPU core  
**Dependencies:** Python standard library only (`json`, `hashlib`, `os`, `datetime`, `collections`)

---

## Quick Start

```bash
# Place pipeline.py next to candidates.jsonl
cd data_Clean/

# Run
python pipeline.py
```

**Expected output:**

```
📂  Reading and cleaning candidates.jsonl ...
    ... 10,000 processed  (3.2s elapsed)
    ... 20,000 processed  (6.4s elapsed)
    ...
    ✅ 100,000 cleaned  |  0 skipped  |  35.1s

🔖  Stamping fabrication bandwidth ...
    max raw bandwidth across dataset : 48,200
    ✅ fabrication_bandwidth stamped for all candidates  |  0.4s

🌳  Building Dataset/ tree ...
    ✅ ~280 leaf files written  |  7.8s

══════════════════════════════════════════════════════════
  PIPELINE COMPLETE
══════════════════════════════════════════════════════════
  Candidates loaded        : 100,000
  Candidates cleaned       : 100,000
  Skipped (malformed)      : 0
  ── Quality ────────────────────────────────────────────
  Duplicate descriptions   : ~70,000  (70.0%)
  Avg fabrication_bandwidth: ~0.280   (0-1 scale)
  Max bandwidth (raw)      : 48,200
  ── Classification ─────────────────────────────────────
  code                     : ~45,000  (45.0%)
  no_code                  : ~55,000  (55.0%)
  available                : ~6,000   (6.0%)
  unavailable              : ~94,000  (94.0%)
  pure CV (no retrieval)   : ~1,200   (1.2%)
  ── Output ─────────────────────────────────────────────
  Unique descriptions      : ~10
  Leaf files written       : ~280
  Total wall time          : ~44s
```

---

## Project Structure

```
data_Clean/
├── candidates.jsonl              ← raw input (100,000 candidates)
├── pipeline.py                   ← this pipeline
└── Dataset/                      ← output (auto-created)
    ├── code/
    │   ├── available/
    │   │   ├── 0_to_3/
    │   │   │   ├── engineering/
    │   │   │   │   ├── ml_engineer.json
    │   │   │   │   ├── data_engineer.json
    │   │   │   │   ├── software_engineer.json
    │   │   │   │   ├── computer_vision_engineer.json
    │   │   │   │   └── ...
    │   │   │   ├── devops_and_cloud/
    │   │   │   │   ├── devops_engineer.json
    │   │   │   │   └── ...
    │   │   │   ├── product_and_design/
    │   │   │   ├── operations/
    │   │   │   ├── business/
    │   │   │   ├── marketing/
    │   │   │   ├── finance/
    │   │   │   ├── hr_and_people/
    │   │   │   ├── non_tech_engineering/
    │   │   │   └── other/
    │   │   ├── 4_to_9/           ← same domain/role structure
    │   │   └── 10_plus/          ← same domain/role structure
    │   └── unavailable/          ← same experience/domain/role structure
    └── no_code/                  ← same availability/experience/domain/role structure
```

Each leaf file (e.g. `ml_engineer.json`) is a **JSON array** of fully cleaned candidate objects with all flags and scores attached.

---

## Pipeline Architecture

### Why three passes?

```
Pass 1 — File Read (I/O bound)
  ├── json.loads() per line → fresh dict, no deepcopy needed
  ├── apply_fixes()         → correct sentinel values
  ├── process_candidate()   → compute all flags
  └── collect_desc_freq()   → MD5(description) → global count table

Pass 2 — In-Memory (CPU bound, fast)
  └── stamp_fabrication_bandwidth()
      → needs global max across all candidates
      → only possible after Pass 1 completes
      → normalized = raw_bandwidth / max_bandwidth

Pass 3 — In-Memory + Writes (I/O bound, minimal)
  └── build_tree()
      → classify() each candidate into 5 dimensions
      → group by leaf path using defaultdict(list)
      → write each group as one JSON array file
      → os.makedirs(exist_ok=True) creates folders automatically
```

### Performance decisions

| Decision | Why |
|---|---|
| No `copy.deepcopy()` | `json.loads()` already returns a fresh dict per line — deepcopy was wasting ~5-6 seconds |
| MD5 for description hashing | Fixed-length key, fast, negligible collision probability for ~1M strings |
| `defaultdict(list)` grouping | Group all candidates in-memory before writing — ~200-400 file writes instead of 100K |
| No intermediate batch files | Eliminates disk I/O and second file read entirely |

---

## Data Fixes

Fixes run **before** any flag. Flags must read corrected values, not raw sentinels.

### F1 — Salary Range Inversion

**Field:** `redrob_signals.expected_salary_range_inr_lpa.min / .max`

**Problem:** `min` and `max` were generated from independent distributions. Some records have `min > max` (e.g. min = 36.7, max = 34.3).

**Fix:** If `min > max`, swap them. Set `salary_was_inverted = True`.

---

### F2 — GitHub Score Sentinel

**Field:** `redrob_signals.github_activity_score`

**Problem:** `-1` means "no GitHub account linked", not a real score. `None` in raw data also means not linked.

**Fix:** If score is `-1` or `None`, replace with `None` and set `github_not_linked = True`.

> **Bug fixed:** Previous version set `github_not_linked = False` when score was already `None` — incorrectly implying GitHub was linked.

---

### F3 — Offer Acceptance Rate Sentinel

**Field:** `redrob_signals.offer_acceptance_rate`

**Problem:** `-1` means "no offer history on platform", not a real rate.

**Fix:** If value is `-1`, replace with `None` and set `no_offer_history = True`.

All three fix flags (`salary_was_inverted`, `github_not_linked`, `no_offer_history`) are bubbled to the **top level** of the candidate dict for direct access by ranking.

---

## Quality Flags

All flags are boolean. They **do not remove candidates**. All 100,000 records are retained.

### FL3 — Skill Career Domain Mismatch

**Field:** `skill_career_domain_mismatch` (bool)

Fires when a candidate claims 3+ AI/ML skills but has **zero AI/ML career history**.

**Threshold:** `ai_skill_count >= 3 AND _has_ai_career() == False`

**Why:** The JD explicitly warns against keyword stuffers — *"if your AI experience consists primarily of recent projects using LangChain to call OpenAI, we will probably not move forward."* Skills generated independently of career history in this synthetic dataset often produce this pattern.

---

### FL5 — Education Overlap

**Fields:** `education_overlap` (bool), `overlapping_education_indices` (list of pairs)

Fires when two education entries have **overlapping year ranges**.

**Threshold:** `max(start1, start2) < min(end1, end2)` for any pair

**Why:** Cannot be enrolled full-time in two degree programmes simultaneously. Indicates records from independent pools.

---

### FL6 — Reverse Degree Order

**Field:** `reverse_degree_order` (bool)

Fires when a **postgraduate degree was completed before an undergraduate degree started**.

**Threshold:** `min(pg_end_years) <= max(ug_start_years)` (uses `<=` to catch same-year edge cases)

**Why:** A Masters requires a Bachelors first. Impossible ordering confirms independently assigned records.

---

### FL7 — Second Undergraduate After First

**Field:** `second_undergrad_after_first` (bool)

Fires when two undergraduate degrees have a **2+ year gap** between them.

**Threshold:** `second_ug.start_year >= first_ug.end_year + 2`

**Why:** Starting a second bachelor's from scratch years after finishing the first has almost no legitimate explanation.

---

### FL8 / FL9 — Education Career Gap

**Fields:** `education_career_gap_flag` (bool), `education_career_gap_years` (float)

Fires when the gap between finishing education and starting career is **greater than 1 year**.

**Formula:** `gap = min(career_start_years) - max(edu_end_years)`

- Positive gap = career after graduation (normal direction)
- Negative gap = career started **before** graduation ended (impossible)
- `gap = 1.0` does **NOT** trigger (strict `>`)
- 1-year buffer allows for final-year internships and part-time work

---

### FL11 / FL12 — Active Before Signup

**Fields:** `active_before_signup` (bool), `signup_active_gap_days` (int)

Fires when `last_active_date` is **earlier than** `signup_date`.

**Why:** Cannot have platform activity before creating an account. Timestamps were generated independently. The `last_active_date` field is unreliable for these candidates.

---

### FL16 — Invalid Degree Field Combination

**Fields:** `invalid_degree_field_combination` (bool, per education entry), `_any_invalid_degree_field` (bool, at candidate root)

Fires when an **engineering degree** (B.Tech, M.Tech, B.E., M.E., etc.) is paired with a **non-engineering field of study** (MBA, Commerce).

**Why:** MBA and Commerce are entirely separate educational streams. They cannot be the "field of study" within an engineering degree programme.

---

### FL18 — Duplicate Job Descriptions (Within-Candidate)

**Field:** `duplicate_job_descriptions` (bool)

Fires when **2 or more** of this candidate's **own** career descriptions are character-for-character identical — whether just 2 match, or all of them.

**How:** `len(set(non_empty_descriptions)) < len(non_empty_descriptions)`

**Scope:** Within-candidate only. Checks this candidate's career entries against each other. Does not compare against the global dataset.

**Why merged from FL18 + FL19:** The old version had two flags — FL18 for "any duplicate" and FL19 for "all identical". Both detect the same underlying problem (template-assigned descriptions) with different severity, not different type. Merged into one.

---

### FL20 — Low Engagement

**Field:** `low_engagement_flag` (bool)

Fires when a candidate is **practically unreachable** for recruiters.

**Threshold:** `recruiter_response_rate < 0.10 AND avg_response_time_hours > 200` (both simultaneously)

**Boundary:** `rate = 0.10` exactly does **NOT** trigger (strict `<`)

**Why:** Per JD — *"a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is not actually available."* Not a fabrication indicator — a practical availability warning.

---

## Fabrication Bandwidth

`fabrication_bandwidth` is a **continuous normalized score in [0, 1]**, not a boolean. It measures how much a candidate's career description text relies on shared templates across the full dataset.

### How it's computed

```
Pass 1:  For every description across all 100K candidates:
           MD5(description) → increment global count

         For each candidate:
           raw_bandwidth = sum(global_count[MD5(desc)] for each career description)

         After Pass 1:
           max_bandwidth = max(raw_bandwidth) across entire dataset

Pass 2:  For each candidate:
           fabrication_bandwidth = raw_bandwidth / max_bandwidth
```

### Why divide by max (not full min-max)

The minimum `raw_bandwidth` is always `0` — a candidate with all-unique descriptions. So min-max normalization reduces to:

```
(value - 0) / (max - 0) = value / max
```

This gives a clean `[0, 1]` scale where:
- `0.0` = fully original career text, zero overlap with any other candidate
- `1.0` = maximum template reuse in the entire dataset
- `~0.3` = moderate template reuse (expected for many candidates)

### Why sum over career entries (not max)

A candidate with **three** recycled descriptions (frequencies 300, 200, 347) is more suspicious than one with a **single** description appearing 400 times. The sum captures the aggregate extent of template reuse across the full career history.

---

## Dataset Tree Structure

```
Dataset/
└── {code_status}          2 values: code | no_code
    └── {availability}     2 values: available | unavailable
        └── {experience}   3 values: 0_to_3 | 4_to_9 | 10_plus
            └── {domain}   10 values: engineering | devops_and_cloud |
            │              product_and_design | operations | business |
            │              marketing | finance | hr_and_people |
            │              non_tech_engineering | other
            └── {role}.json
```

**Max possible leaf files:** 2 × 2 × 3 × 10 × ~12 = **~1,440**  
**Actual files created:** Only buckets that have candidates are written. Empty buckets are skipped.

---

## Classification Logic

### Dimension 1 — Code Status

| Bucket | Logic |
|---|---|
| `code` | Current title IS a production coding role. Positive match against `CODE_TITLES` set. Covers: software engineer, backend/frontend engineer, full stack developer, mobile developer, ml engineer, data scientist, ai engineer, nlp engineer, computer vision engineer, mlops engineer, data engineer, qa engineer, devops engineer, cloud engineer, and senior/principal/staff variants. |
| `no_code` | Three cases: (1) Current title is NOT in `CODE_TITLES` — manager, analyst, HR, marketing, finance, civil/mechanical engineer, etc. (2) Zero career history — pure academic. (3) Entire career in pure research/academic titles — professor, postdoc, research fellow. |

**Why positive match (not duration-based):** Previous version checked if someone had been in a non-coding title for 18+ months. This let a Business Analyst with 8 months tenure land in `code`. The correct question is: *is this person writing production code right now?* Duration is irrelevant.

---

### Dimension 2 — Availability

| Bucket | Logic |
|---|---|
| `available` | **ALL THREE** must be true: `open_to_work_flag = True` AND `willing_to_relocate = True` AND `notice_period_days < 90` |
| `unavailable` | Any other combination |

**Why all three required:**
- `open_to_work` alone: candidate may be unwilling to relocate → not reachable for Pune/Noida role
- `willing_to_relocate` alone: may have 150-day notice → impractical hiring window
- `notice < 90` alone: short notice but not actively looking → won't engage

**Boundary:** `notice = 89` → `available`. `notice = 90` → `unavailable`.

---

### Dimension 3 — Experience

| Bucket | Range | Meaning |
|---|---|---|
| `0_to_3` | 0 – 3.0 years | Entry level / early career |
| `4_to_9` | 3.1 – 9.0 years | Mid-level (JD primary target: 5–9 years) |
| `10_plus` | 10.0+ years | Senior |

---

### Dimension 4 — Domain

| Domain | What It Covers |
|---|---|
| `engineering` | All software + AI/ML + data roles (merged — both require coding) |
| `devops_and_cloud` | DevOps, cloud, platform, SRE, systems, security |
| `product_and_design` | Product managers, UX/UI/graphic designers |
| `operations` | Ops managers, project managers, scrum masters, customer support |
| `business` | Business analysts, sales, consultants |
| `marketing` | Marketing, digital marketing, SEO, content writers |
| `finance` | Finance managers, analysts, accountants, auditors |
| `hr_and_people` | HR managers, recruiters, L&D, HRBPs |
| `non_tech_engineering` | Mechanical, civil, electrical, chemical, structural engineers |
| `other` | No domain rule matched → `unclassified.json` |

---

### Dimension 5 — Role (the filename)

The role slug becomes the `.json` filename at the leaf. For example:

```
Dataset/code/available/4_to_9/engineering/ml_engineer.json
```

Contains all active, relocatable, mid-level ML Engineers who actively write code.

---

## Domain & Role Routing

### Standard routing (non-engineering domains)

First keyword match in `DOMAIN_ROLE_RULES` wins. Rules are priority-ordered. Current title (lowercased) is checked against keyword lists.

### Engineering domain — 3-tier career routing

For candidates whose current title matches the `engineering` domain, standard title matching is overridden by a career-history-based system:

```
Priority 1 — AI/Data career (highest)
  └── ANY career job title matches AI/ML/Data keywords
      → routes to that AI/Data role file
      → most recent AI/Data entry determines the specific role
      → Example: title = "Software Engineer", career has "ML Engineer"
                 → ml_engineer.json

Priority 2 — Specialized tech career
  └── career contains specialized tech title (mobile, java, .net,
      embedded, qa, cloud, devops, etc.)
      → most recent specialized entry determines the route
      → Example: title = "Software Engineer", career has "Cloud Engineer"
                 → devops_and_cloud/cloud_engineer.json

Priority 3 — Pure Software Engineer (fallback)
  └── No AI/Data career, no specialized career
      → software_engineer.json
      → Only genuinely generalist engineers reach this file
```

**Why career titles (not descriptions)?**  
Descriptions in this dataset are template-generated and unreliable. Job titles are the primary signal of what someone actually did.

**Why most recent for Priority 2?**  
Career direction matters more than past experience. A candidate who was Java Developer → then QA Engineer is more likely to be hired as a QA Engineer today. Current direction > historical experience.

---

## Computer Vision Special Routing

From the JD: *"People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure — we respect your work but you'd be re-learning fundamentals here."*

| Condition | Routes To |
|---|---|
| CV title + **has** retrieval/NLP/IR skills (embeddings, FAISS, Pinecone, Qdrant, Weaviate, Milvus, sentence transformers, NLP, semantic search, IR, learning to rank, BM25, RAG, LLMs, fine-tuning LLMs, Haystack, pgvector) | `engineering/ml_engineer.json` — crossed the JD threshold |
| CV title + **no** retrieval/NLP/IR skills | `engineering/computer_vision_engineer.json` — pure CV, flagged by JD |

This rule runs **before** the 3-tier engineering routing.

---

## Complete Field Reference

Every candidate in every leaf file carries these fields in addition to the original schema. **No original fields are deleted.**

### Fix fields (computed in Pass 1, before flags)

| Field | Type | Meaning |
|---|---|---|
| `salary_was_inverted` | bool | Salary min/max were swapped in raw data and corrected |
| `github_not_linked` | bool | No GitHub linked (raw score was -1 or None) |
| `no_offer_history` | bool | No offer history on platform (raw rate was -1) |

### Boolean flags

| Flag | Type | Source | Meaning |
|---|---|---|---|
| `skill_career_domain_mismatch` | bool | FL3 | 3+ AI skills claimed but zero AI career history |
| `education_overlap` | bool | FL5 | Two education entries with overlapping year ranges |
| `reverse_degree_order` | bool | FL6 | Postgraduate degree completed before undergraduate started |
| `second_undergrad_after_first` | bool | FL7 | Two undergraduate degrees with 2+ year gap between them |
| `education_career_gap_flag` | bool | FL8 | Gap > 1 year between education end and career start |
| `active_before_signup` | bool | FL11 | last_active_date is earlier than signup_date |
| `duplicate_job_descriptions` | bool | FL18 | 2+ of this candidate's own career descriptions are identical |
| `low_engagement_flag` | bool | FL20 | response_rate < 0.10 AND response_time > 200 hours |
| `_any_invalid_degree_field` | bool | FL16 agg | Any education entry has invalid degree-field combination |
| `invalid_degree_field_combination` | bool (per edu) | FL16 | Engineering degree paired with MBA or Commerce |

### Value fields

| Field | Type | Source | Meaning |
|---|---|---|---|
| `education_career_gap_years` | float | FL9 | Actual gap in years. Positive = career after graduation. Negative = impossible (career before graduation ended). |
| `signup_active_gap_days` | int | FL12 | Days that last_active_date predates signup_date |
| `fabrication_bandwidth` | float 0–1 | Pass 2 | Normalized template-reuse score. 0 = fully original. 1 = maximum reuse in dataset. |

### Index fields

| Field | Meaning |
|---|---|
| `overlapping_education_indices` | List of `[i, j]` pairs identifying which education entries overlap (FL5) |

### Removed flags

| Removed | Why | Replacement |
|---|---|---|
| `possible_honeypot` | Boolean aggregate removed | Individual structural flags retained for ranking to use independently |
| `possible_fabrication` | Boolean threshold removed | `fabrication_bandwidth` (continuous 0–1) retained |
| `all_descriptions_identical` | Merged into FL18 | `duplicate_job_descriptions` covers both any-duplicate and all-identical |
| `honeypot_score` | Aggregate score removed | Individual flag fields retained for ranking to weight independently |

---

## Bugs Found & Fixed

| # | Severity | Area | Bug | Fix |
|---|---|---|---|---|
| 1 | 🔴 High | Domain routing | `"data engineer"` and `"analytics engineer"` in `AI_CAREER_KEYWORDS` — DE candidates with 3+ AI skills routed to `ai_ml` instead of `data_engineering` | Removed both from `AI_CAREER_KEYWORDS` |
| 2 | 🔴 High | Industry match | `"ai" in industry` substring matched `"retail"`, `"financial"`, `"pharmaceutical"` (all contain `"ai"`) | Replaced with exact set match against `AI_INDUSTRIES_EXACT` |
| 3 | 🟠 Medium | FL3 vs routing | `AI_SKILLS` (FL3) ≠ `AI_DOMAIN_SKILLS` (routing) — same candidate's AI skills counted differently by flag vs classifier | Merged into one unified `AI_SKILLS` set |
| 4 | 🟠 Medium | F2 fix | `_fix_github` set `github_not_linked = False` when score was already `None` — implied GitHub was linked | Also treat `None` as not linked |
| 5 | 🟠 Medium | FL8 threshold | `fl8` only flagged `gap > 5`, missing impossible negative gaps (career before graduation) | Changed threshold to `gap > 1` |
| 6 | 🔵 Logic | code/no_code | Checked title duration (18 months) — BA/HR/marketing landed in `code` if tenure was short | Positive match against `CODE_TITLES` — code = current title IS a coding role |
| 7 | 🔵 Logic | FL18/FL19 | Two separate flags for the same underlying problem | Merged into single within-candidate `duplicate_job_descriptions` |
| 8 | 🟡 Low | SWE title | `"sde "` with trailing space missed `"SDE"`, `"SDE-2"`, `"SDE2"` variants | Changed to plain `"sde"` substring |
| 9 | ⚡ Perf | deepcopy | `copy.deepcopy()` on every record — unnecessary since `json.loads()` returns a fresh dict | Removed, saving ~5-6 seconds on 100K records |

---

## Design Decisions

### Why non-destructive?
All 100,000 records are retained. The pipeline is a preprocessing stage, not a filter. Downstream ranking uses flag values as soft penalties. Hard filtering would remove candidates valuable for edge case analysis or training data.

### Why MD5 for description hashing?
Two character-identical descriptions always produce the same MD5. Fixed-length key regardless of description length (some descriptions are 300+ characters). Fast to compute. Collision probability negligible for ~1M strings.

### Why all three conditions for `available`?
Each alone is insufficient:
- `open_to_work` alone: candidate may be unwilling to relocate → not reachable for Pune/Noida
- `willing_to_relocate` alone: may have 150-day notice → impractical
- `notice < 90` alone: short notice but not actively looking → won't engage

All three together: *they want a job, they can come here, they can start soon.*

### Why sum for `fabrication_bandwidth`?
A candidate with three recycled descriptions is more suspicious than one with a single recycled description. The sum captures the aggregate extent of template reuse across the full career history.

### Why merge FL18 and FL19?
Both detected the same problem — template-assigned descriptions — with different severity, not a different type. The merged flag keeps detection simple. Severity is visible from `fabrication_bandwidth` and the number of career entries.

### Why positive match for code status?
Previous version: checked if someone had been in a non-coding title for 18+ months. This let a BA with 8 months tenure land in `code`. Correct question: *is this person writing production code right now?* Positive match against `CODE_TITLES` is unambiguous. Duration is irrelevant.

### Why career-based routing for the engineering domain?
Current title alone is misleading. A candidate titled "Software Engineer" who spent 3 years as an ML Engineer and 1 year as a SWE is more valuable to an AI team than a lifelong generic SWE. Career history reveals actual specialization. The 3-tier system surfaces this signal.

---

## Threshold Reference

All thresholds use strict operators (`>` not `>=`, `<` not `<=`) unless noted. A value exactly equal to the threshold does **NOT** trigger.

| Flag / Dimension | Threshold | Boundary |
|---|---|---|
| `education_career_gap_flag` | `gap > 1 year` | `gap = 1.0` → NOT flagged; `gap = 1.1` → flagged |
| `low_engagement_flag` | `rate < 0.10 AND hours > 200` | `rate = 0.10` exactly → NOT flagged |
| `available` | `open=True AND willing=True AND notice < 90` | `notice = 89` → available; `notice = 90` → unavailable |
| `experience 0_to_3` | `yoe <= 3` | `3.0` → `0_to_3`; `3.1` → `4_to_9` |
| `experience 4_to_9` | `yoe <= 9` | `9.0` → `4_to_9`; `9.1` → `10_plus` |
| `FL7 second undergrad` | `gap >= 2 years between UG degrees` | `gap = 2` → flagged; `gap = 1` → NOT flagged |
| `FL6 reverse degree` | `min(pg_end) <= max(ug_start)` | Same year → flagged (uses `<=`, not `<`) |
| `fabrication_bandwidth` | `raw / max_bandwidth` | `0.0` = fully original; `1.0` = most fabricated |

---

## Tools Used

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12+ | Pipeline runtime |
| `json` | stdlib | Parsing `candidates.jsonl` line by line |
| `hashlib` | stdlib | MD5 hashing of job descriptions for frequency table |
| `os` | stdlib | Creating Dataset/ folder tree, path joins |
| `datetime` | stdlib | Parsing date strings, computing gaps and recency |
| `collections.defaultdict` | stdlib | Grouping candidates by leaf path before writing |
| `re` | stdlib | Pattern matching in cleanup passes |

**No external libraries required.**  
**No GPU required.**  
**No network access required.**

---

## Re-running

The pipeline **overwrites** existing leaf files on each run (not appends). To re-run cleanly from scratch:

```bash
rm -rf Dataset/
python pipeline.py
```

---

*Built for the Redrob Intelligent Candidate Discovery & Ranking Hackathon — June 2026*
