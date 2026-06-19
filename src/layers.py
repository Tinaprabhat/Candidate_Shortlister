"""
layers.py — The 5-layer cascade (L1–L4, L6).

L1: Hard reject (fraud / impossibilities)         → binary
L2: Bi-encoder similarity (concat skills+work)    → score + 55% gate
L3: Seniority regression                          → soft penalty
L4: Semantic work-to-JD relevance (descs only)    → independent score
L6: Behavioral signals                            → normalized score
"""

import math
import random
import logging
from datetime import datetime
from typing import List, Dict

import numpy as np

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)

# In-code fictional company blacklist (backup to SQLite KB)
FICTIONAL_COMPANIES = {
    "dunder mifflin", "hooli", "acme corp", "acme corporation", "initech",
    "pied piper", "vehement capital", "globex", "soylent corp", "umbrella corp",
    "umbrella corporation", "stark industries", "wayne enterprises", "wonka industries",
    "cyberdyne systems", "weyland-yutani", "tyrell corporation", "oscorp",
    "massive dynamic", "aperture science", "black mesa", "vault-tec",
}


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1 — FRAUD KB  (mathematical consistency + KB verification)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_year(val):
    """Coerce an int, ISO-date string, or None to a 4-digit int year."""
    if val is None:
        return None
    try:
        return int(str(val)[:4])
    except (ValueError, TypeError):
        return None


def _company_founding_year(conn, company: str):
    if conn is None or not company:
        return None
    row = conn.execute(
        "SELECT founding_year FROM company_founding_dates WHERE LOWER(company_name)=?",
        (company.strip().lower(),),
    ).fetchone()
    return row["founding_year"] if row else None


def _fuzzy_kb_lookup(conn, table: str, name_col: str, query: str, threshold: int = 85):
    """
    Fuzzy name lookup in a SQLite table using rapidfuzz (ratio > threshold).
    Falls back to exact-match only when rapidfuzz is unavailable.
    Returns the first matching row, or None.
    """
    if not query or conn is None:
        return None
    q = query.strip().lower()
    if not q:
        return None
    # Fast path: exact match
    try:
        row = conn.execute(
            f"SELECT * FROM {table} WHERE LOWER({name_col})=?", (q,)
        ).fetchone()
        if row:
            return row
    except Exception:
        return None
    # Fuzzy path: prefix-filter in SQL, then rapidfuzz scoring
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    prefix = q[:3] + "%"
    try:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE LOWER({name_col}) LIKE ?", (prefix,)
        ).fetchall()
    except Exception:
        return None
    best_score, best_row = 0, None
    for row in rows:
        try:
            name = str(row[name_col]).lower()
        except (KeyError, TypeError):
            try:
                name = str(row[0]).lower()
            except Exception:
                continue
        s = fuzz.ratio(q, name)
        if s > best_score:
            best_score, best_row = s, row
    return best_row if best_score >= threshold else None


def _kb_company_status(conn, company: str) -> str:
    """
    Classify a company name against the fraud KB.
    Returns 'fictional' (hard reject) | 'verified' (1.0) | 'unknown' (0.5).
    """
    if not company or conn is None:
        return "unknown"
    # In-code fictional list (fast, no DB round-trip)
    if company.strip().lower() in FICTIONAL_COMPANIES:
        return "fictional"
    # KB fictional table
    if _fuzzy_kb_lookup(conn, "fictional_companies", "company_name", company):
        return "fictional"
    # Legitimate-company tables: MCA (indian_companies), DPIIT (indian_startups), PDL (global_companies)
    for tbl in ("indian_companies", "indian_startups", "global_companies"):
        try:
            if _fuzzy_kb_lookup(conn, tbl, "company_name", company):
                return "verified"
        except Exception:
            pass
    return "unknown"


def _kb_university_status(conn, institution: str) -> str:
    """Returns 'verified' (1.0) | 'unknown' (0.5)."""
    if not institution or conn is None:
        return "unknown"
    for tbl in ("universities", "indian_universities", "global_universities"):
        try:
            if _fuzzy_kb_lookup(conn, tbl, "institution_name", institution):
                return "verified"
        except Exception:
            pass
    return "unknown"


def _kb_paper_status(conn, title: str, claimed_author: str) -> str:
    """Returns 'verified' (1.0) | 'contradicts' (hard reject) | 'unknown' (0.5)."""
    if not title or conn is None:
        return "unknown"
    try:
        row = _fuzzy_kb_lookup(conn, "research_papers", "title", title)
    except Exception:
        return "unknown"
    if row is None:
        return "unknown"
    if claimed_author:
        try:
            authors_str = str(row["authors"]).lower()
        except (KeyError, TypeError):
            return "verified"
        tokens = {w for w in claimed_author.lower().split() if len(w) > 2}
        if any(t in authors_str for t in tokens):
            return "verified"
        return "contradicts"
    return "verified"


def _check_math_consistency(c: dict, fraud_kb, current_year: int):
    """
    Part 1: Pure arithmetic. Returns (flags: list[str], reject: bool).
    If ANY check fails → reject=True (hard reject, no further layers run).

    Checks (in order):
      - Pre-computed honeypot / salary-inversion flags
      - Education timeline (degree ordering)
      - Exp vs graduation: grad_year + exp_years <= current_year
      - Future end dates (end_date > current_year)
      - Worked before company founded
      - Senior role claimed before graduation
      - Overlapping full-time jobs (same date range)
      - Age vs experience: claimed_exp <= current_year - (birth_year + 18)
      - Total exp vs career span: total_exp <= years_since_graduation
    """
    flags = []

    # ── Pre-computed flags ─────────────────────────────────────────────────────
    if c.get("possible_honeypot") is True:
        return ["honeypot_flag"], True
    if c.get("salary_was_inverted") is True:
        return ["salary_inverted"], True
    sal = c.get("salary_expectation") or {}
    smin, smax = sal.get("min"), sal.get("max")
    if smin is not None and smax is not None:
        try:
            if float(smin) > float(smax):
                return ["salary_range_inverted"], True
        except (ValueError, TypeError):
            pass

    # ── Education timeline ─────────────────────────────────────────────────────
    edu = c.get("education") or []
    edu_years: dict = {}
    grad_year = None
    for e in edu:
        if not isinstance(e, dict):
            continue
        deg = str(e.get("degree", "")).lower()
        yr = _parse_year(e.get("end_year") or e.get("year"))
        if yr is None:
            continue
        if any(x in deg for x in ["phd", "doctor"]):
            edu_years.setdefault("phd", yr)
        elif any(x in deg for x in ["master", "m.tech", "m.sc", "m.e", "mba", "postgrad"]):
            edu_years.setdefault("master", yr)
        elif any(x in deg for x in ["bachelor", "b.tech", "b.sc", "b.e", "b.com", "b.a", "b.eng"]):
            edu_years.setdefault("bachelor", yr)
            if grad_year is None or yr < grad_year:
                grad_year = yr

    if edu_years.get("phd") and edu_years.get("bachelor") and edu_years["phd"] < edu_years["bachelor"]:
        return ["phd_before_bachelor"], True
    if edu_years.get("master") and edu_years.get("bachelor") and edu_years["master"] < edu_years["bachelor"]:
        return ["master_before_bachelor"], True

    # ── Extract experience / birth year ───────────────────────────────────────
    total_exp = utils.get_total_experience_years(c)
    birth_year = None
    for field in ("date_of_birth", "dob", "birth_year"):
        bval = c.get(field)
        if bval is not None:
            yr = _parse_year(bval)
            if yr and 1930 <= yr <= current_year - 16:
                birth_year = yr
                break

    # ── Check 1 & 5: experience vs graduation / career span ───────────────────
    # grad_year + exp_years <= current_year  ⟺  exp <= current_year - grad_year
    if grad_year and total_exp > 0:
        career_span = current_year - grad_year
        if total_exp > career_span:
            flags.append(
                f"exp_exceeds_career:grad_{grad_year}+{total_exp:.0f}yr>{current_year}"
            )
            return flags, True

    # ── Work-history checks ────────────────────────────────────────────────────
    work_entries = c.get("work_experience") or c.get("career_history") or []
    ft_intervals = []  # full-time job (start, end) for overlap detection

    for exp in work_entries:
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company", "")).strip()
        emp_type = str(exp.get("employment_type", "")).lower()
        is_part_time = any(
            t in emp_type
            for t in ["part-time", "part time", "intern", "freelance",
                       "contract", "consultant", "volunteer"]
        )
        start = _parse_year(exp.get("start_year") or exp.get("start_date"))

        # Check 3: future end date
        end_raw = exp.get("end_year") or exp.get("end_date")
        if end_raw is not None:
            end = _parse_year(end_raw)
            if end is not None and end > current_year:
                flags.append(f"future_end_date:{end}")
                return flags, True
            end = end if end is not None else current_year
        else:
            end = current_year  # ongoing / present

        # Work before company founded
        if start is not None and company:
            fy = _company_founding_year(fraud_kb, company)
            if fy is not None and start < fy:
                flags.append(f"worked_before_founding:{company}_{fy}")
                return flags, True

        # Senior role claimed before graduation
        if start is not None and grad_year is not None and start < grad_year - 1:
            title = str(exp.get("title", "")).lower()
            if any(k in title for k in ["senior", "lead", "principal", "manager", "director"]):
                flags.append(f"senior_role_before_graduation:{title}")
                return flags, True

        if not is_part_time and start is not None:
            ft_intervals.append((start, end))

    # Check 2: overlapping full-time jobs
    ft_intervals.sort()
    for i in range(len(ft_intervals) - 1):
        s1, e1 = ft_intervals[i]
        s2, e2 = ft_intervals[i + 1]
        if max(s1, s2) < min(e1, e2):  # genuine overlap (not just adjacent years)
            flags.append(f"overlapping_jobs:{s1}-{e1}_and_{s2}-{e2}")
            return flags, True

    # Check 4: age vs experience
    if birth_year is not None and total_exp > 0:
        max_exp = max(0, current_year - (birth_year + 18))
        if total_exp > max_exp:
            flags.append(f"exp_exceeds_age_limit:{total_exp:.0f}yr>max_{max_exp}yr")
            return flags, True

    return flags, False


def _run_kb_verification(c: dict, fraud_kb, current_year: int):
    """
    Part 2: Local KB verification (runs only when Part 1 passes).
    Scoring per entity: 1.0=verified, 0.5=out-of-KB, 0.0=contradicts (→ reject).
    l1_score = average of all entity scores.
    Returns (l1_score: float, flags: list[str], status: str).
    """
    kb_scores = []
    flags = []

    # ── Verify companies in work history ──────────────────────────────────────
    work_entries = c.get("work_experience") or c.get("career_history") or []
    for exp in work_entries:
        if not isinstance(exp, dict):
            continue
        company = str(exp.get("company", "")).strip()
        if not company:
            continue
        status = _kb_company_status(fraud_kb, company)
        if status == "fictional":
            flags.append(f"fictional_company:{company}")
            return 0.0, flags, "reject"
        kb_scores.append(1.0 if status == "verified" else 0.5)

    # ── Verify educational institutions ───────────────────────────────────────
    for e in (c.get("education") or []):
        if not isinstance(e, dict):
            continue
        inst = str(
            e.get("institution") or e.get("university") or e.get("school") or ""
        ).strip()
        if not inst:
            continue
        status = _kb_university_status(fraud_kb, inst)
        kb_scores.append(1.0 if status == "verified" else 0.5)

    # ── Verify research paper authorship ──────────────────────────────────────
    for pub in (c.get("publications") or c.get("research_papers") or []):
        if not isinstance(pub, dict):
            continue
        title = str(pub.get("title", "")).strip()
        author = str(
            pub.get("author") or pub.get("first_author") or c.get("name") or ""
        ).strip()
        if not title:
            continue
        status = _kb_paper_status(fraud_kb, title, author)
        if status == "contradicts":
            flags.append(f"paper_authorship_mismatch:{title[:60]}")
            return 0.0, flags, "reject"
        kb_scores.append(1.0 if status == "verified" else 0.5)

    l1_score = sum(kb_scores) / len(kb_scores) if kb_scores else 0.5
    return l1_score, flags, "pass"


def l1_hard_reject(candidates: List[dict], fraud_kb) -> List[dict]:
    """
    Layer 1 — Fraud KB:
      Part 1: Mathematical consistency checks (hard reject on any failure).
      Part 2: Local KB verification (score 0–1; hard reject on contradiction).

    Attaches l1_score [0–1], l1_flags [list], l1_status [pass|reject] to each
    candidate. Returns only candidates with l1_status == 'pass'.
    """
    current_year = datetime.now().year
    survivors = []
    rejected = 0

    for c in candidates:
        # Part 1: math consistency
        math_flags, is_reject = _check_math_consistency(c, fraud_kb, current_year)
        if is_reject:
            c["l1_score"] = 0.0
            c["l1_flags"] = math_flags
            c["l1_status"] = "reject"
            rejected += 1
            continue

        # Part 2: KB verification (only if Part 1 passes)
        l1_score, kb_flags, status = _run_kb_verification(c, fraud_kb, current_year)
        c["l1_score"] = l1_score
        c["l1_flags"] = math_flags + kb_flags
        c["l1_status"] = status

        if status == "reject":
            rejected += 1
        else:
            survivors.append(c)

    logger.info(f"L1: {len(candidates)} in → {len(survivors)} pass, {rejected} hard-rejected")
    return survivors


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 — BI-ENCODER SIMILARITY + 55% GATE
# ──────────────────────────────────────────────────────────────────────────────
def _build_jd_text(jd: dict) -> str:
    parts = []
    parts.append(jd.get("job_title", ""))
    parts += jd.get("explicit_required", [])
    parts += jd.get("inferred_required", [])
    parts += jd.get("explicit_bonus", [])
    parts += jd.get("inferred_bonus", [])
    return " ".join(parts).strip()


def l2_bi_encoder(candidates: List[dict], jd: dict, st_model) -> List[dict]:
    """Score each candidate on wholesome profile-JD similarity. Attaches 'l2_score'."""
    if not candidates:
        return candidates

    jd_text = _build_jd_text(jd)
    jd_emb = st_model.encode([jd_text], convert_to_numpy=True, normalize_embeddings=True)[0]

    texts = [utils.get_work_profile_text(c) or " " for c in candidates]
    embs = st_model.encode(
        texts,
        batch_size=C.L2_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    # cosine similarity (already normalized) → dot product
    scores = embs @ jd_emb
    for c, s in zip(candidates, scores):
        c["l2_score"] = float(max(0.0, s))  # clamp negatives to 0

    logger.info(f"L2: scored {len(candidates)} candidates "
                f"(mean={float(np.mean(scores)):.3f}, max={float(np.max(scores)):.3f})")
    return candidates


def l2_gate(candidates: List[dict], seed: int = 42) -> List[dict]:
    """Top 50% by l2_score + random 5% from bottom half = 55%."""
    if not candidates:
        return candidates

    ranked = sorted(candidates, key=lambda c: c["l2_score"], reverse=True)
    n = len(ranked)
    top_k = int(n * C.GATE_TOP_FRACTION)
    top = ranked[:top_k]
    bottom = ranked[top_k:]

    rng = random.Random(seed)
    rand_k = int(n * C.GATE_RANDOM_FRACTION)
    rescued = rng.sample(bottom, min(rand_k, len(bottom))) if bottom else []

    gated = top + rescued
    logger.info(f"L2 gate: {n} → {len(gated)} "
                f"(top {len(top)} + random {len(rescued)} = {len(gated)/max(n,1)*100:.0f}%)")
    return gated


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 — SENIORITY REGRESSION
# ──────────────────────────────────────────────────────────────────────────────
def l3_seniority(candidates: List[dict], jd: dict) -> List[dict]:
    """Apply soft penalty flag if candidate is under-qualified for JD seniority."""
    req_level_name = str(jd.get("required_seniority", "mid")).lower()
    req_level = C.SENIORITY_KEYWORDS.get(req_level_name, 3)

    penalized = 0
    for c in candidates:
        years = utils.get_total_experience_years(c)
        cand_level = C.years_to_seniority(years)
        if cand_level < req_level:
            c["l3_penalty"] = C.SENIORITY_PENALTY
            c["l3_flag"] = f"under-qualified ({years:.0f}y vs {req_level_name} required)"
            penalized += 1
        else:
            c["l3_penalty"] = 1.0
            c["l3_flag"] = ""
    logger.info(f"L3: {penalized}/{len(candidates)} candidates received seniority penalty")
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 4 — SEMANTIC WORK-TO-JD RELEVANCE (independent)
# ──────────────────────────────────────────────────────────────────────────────
def l4_semantic_work(candidates: List[dict], jd: dict, st_model) -> List[dict]:
    """Independent score: candidate work/project descriptions vs JD responsibilities."""
    if not candidates:
        return candidates

    # JD responsibilities text = required + inferred (the "what you'll do" signal)
    jd_resp = " ".join(jd.get("explicit_required", []) + jd.get("inferred_required", []))
    jd_emb = st_model.encode([jd_resp], convert_to_numpy=True, normalize_embeddings=True)[0]

    texts, idxs = [], []
    for i, c in enumerate(candidates):
        wt = utils.get_work_descriptions_only(c)
        if wt:
            texts.append(wt)
            idxs.append(i)
        else:
            c["l4_score"] = 0.0  # no work descriptions → score 0, no penalty

    if texts:
        embs = st_model.encode(
            texts, batch_size=C.L4_BATCH_SIZE,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        scores = embs @ jd_emb
        for i, s in zip(idxs, scores):
            candidates[i]["l4_score"] = float(max(0.0, s))

    logger.info(f"L4: scored {len(texts)} candidates with work descriptions "
                f"({len(candidates)-len(texts)} had none → score 0)")
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 6 — BEHAVIORAL SIGNALS
# ──────────────────────────────────────────────────────────────────────────────
def l6_behavioral(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Sum behavioral signals and normalize to [0,1] by dividing by the max
    possible score for the role type (absolute, not pool-relative).

    Tech roles: up to 5 signals (github + 4 others); max_possible = 5.0
    Non-tech roles: up to 4 signals; max_possible = 4.0
    Missing signals contribute 0. GitHub absence incurs a -0.1 penalty for
    tech roles (intentional: being on GitHub is expected for tech candidates).
    """
    is_tech = any(k in (jd.get("job_title", "") + " ".join(jd.get("explicit_required", []))).lower()
                  for k in ["engineer", "developer", "ml", "ai", "data", "software", "python"])
    max_possible = 5.0 if is_tech else 4.0

    for c in candidates:
        b = c.get("behavioral_signals", {}) or {}
        score = 0.0

        if is_tech:
            gh = b.get("github_activity_score")
            if gh is not None:
                score += float(gh)
            else:
                score -= 0.1  # mild absence penalty for tech roles

        for key in ["recruiter_response_rate", "profile_completeness",
                    "salary_fit", "notice_period_score"]:
            v = b.get(key)
            if v is not None:
                try:
                    score += float(v)
                except (ValueError, TypeError):
                    pass

        c["l6_score"] = max(0.0, min(1.0, score / max_possible))

    logger.info(f"L6: behavioral scored {len(candidates)} candidates (tech_role={is_tech})")
    return candidates
