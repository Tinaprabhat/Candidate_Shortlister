"""
layers.py — The ranking cascade layers.

Early cascade (per-folder):
  L1   Hard reject (fraud / impossibilities)           → knockout
  L1b  Profile integrity (ATS pre-computed flags)      → hard reject only; soft flags → L2 cols
  L1c  Skill match (NLP + synonym)                     → score [0–1]; explicit-skill hard-reject gate
  L1d  Inferred skill match + leftover penalty         → l1d_score; soft, no rejections

Late cascade (global):
  L2   Table extract (31 cols)                         → feeds L3 FIS
  L3   Sugeno fuzzy inference (conditions a–h)         → l3_score; 75% gate after
  L4   Semantic work relevance (all-MiniLM-L6-v2)      → l4_combined_score; top-200 forwarded
  L5a  Donts penalty layer (high penalty)              → l5_donts_score; top-100
  L5b  FlashRank cross-encoder (top-50)                → l5_total_score = donts + flashrank
"""

import math
import random
import logging
import re
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)

# Per-run caches for fraud KB lookups.
# Keyed by normalised (lowercased, stripped) name → result.
# Thread-safe: values are deterministic (read-only DB), so a double-write from
# two racing threads is harmless — both write the same value.
_company_status_cache: dict = {}   # company_name → 'fictional'|'verified'|'unknown'
_founding_year_cache: dict = {}    # company_name → int|None
_university_status_cache: dict = {}  # institution_name → 'verified'|'unknown'

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
    key = company.strip().lower()
    if key in _founding_year_cache:
        return _founding_year_cache[key]
    with utils.FRAUD_KB_LOCK:
        row = conn.execute(
            "SELECT founding_year FROM company_founding_dates WHERE LOWER(company_name)=?",
            (key,),
        ).fetchone()
    result = row["founding_year"] if row else None
    _founding_year_cache[key] = result
    return result


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
        with utils.FRAUD_KB_LOCK:
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
        with utils.FRAUD_KB_LOCK:
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
    key = company.strip().lower()
    if key in _company_status_cache:
        return _company_status_cache[key]
    # In-code fictional list (fast, no DB round-trip)
    if key in FICTIONAL_COMPANIES:
        _company_status_cache[key] = "fictional"
        return "fictional"
    # KB fictional table
    if _fuzzy_kb_lookup(conn, "fictional_companies", "company_name", company):
        _company_status_cache[key] = "fictional"
        return "fictional"
    # Legitimate-company tables: MCA (indian_companies), DPIIT (indian_startups), PDL (global_companies)
    for tbl in ("indian_companies", "indian_startups", "global_companies"):
        try:
            if _fuzzy_kb_lookup(conn, tbl, "company_name", company):
                _company_status_cache[key] = "verified"
                return "verified"
        except Exception:
            pass
    _company_status_cache[key] = "unknown"
    return "unknown"


def _kb_university_status(conn, institution: str) -> str:
    """Returns 'verified' (1.0) | 'unknown' (0.5)."""
    if not institution or conn is None:
        return "unknown"
    key = institution.strip().lower()
    if key in _university_status_cache:
        return _university_status_cache[key]
    for tbl in ("universities", "indian_universities", "global_universities"):
        try:
            if _fuzzy_kb_lookup(conn, tbl, "institution_name", institution):
                _university_status_cache[key] = "verified"
                return "verified"
        except Exception:
            pass
    _university_status_cache[key] = "unknown"
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
    if grad_year and total_exp > 0:
        career_span = current_year - grad_year
        if total_exp > career_span:
            flags.append(
                f"exp_exceeds_career:grad_{grad_year}+{total_exp:.0f}yr>{current_year}"
            )
            return flags, True

    # ── Work-history checks ────────────────────────────────────────────────────
    work_entries = c.get("work_experience") or c.get("career_history") or []
    ft_intervals = []

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
            end = current_year

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
        if max(s1, s2) < min(e1, e2):
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
    """
    current_year = datetime.now().year
    survivors = []
    rejected = 0

    for c in candidates:
        math_flags, is_reject = _check_math_consistency(c, fraud_kb, current_year)
        if is_reject:
            c["l1_score"] = 0.0
            c["l1_flags"] = math_flags
            c["l1_status"] = "reject"
            rejected += 1
            continue

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
# LAYER 1b — PROFILE INTEGRITY FLAGS
# ──────────────────────────────────────────────────────────────────────────────

_L1B_HARD_REJECT_FLAGS = [
    "reverse_degree_order",
    "all_descriptions_identical",
    "invalid_degree_field_combination",
]

def l1b_profile_integrity(candidates: List[dict]) -> List[dict]:
    """
    Layer 1b — Profile integrity: hard-reject gate only.

    Hard-reject flags (removed from pool):
      reverse_degree_order, all_descriptions_identical, invalid_degree_field_combination

    Soft-penalty flags are NO LONGER applied here.  They are stored as boolean
    columns in the L2 table (cols 29-31) and unified into condition 'h' by L3.

    All surviving candidates get l1b_penalty=1.0, l1b_flags=[], l1b_status='pass'.
    """
    survivors = []
    rejected = 0

    for c in candidates:
        reject_flag = next(
            (f for f in _L1B_HARD_REJECT_FLAGS if c.get(f) is True), None
        )
        if reject_flag:
            c["l1b_penalty"] = 0.0
            c["l1b_flags"]   = [reject_flag]
            c["l1b_status"]  = "reject"
            rejected += 1
            continue

        c["l1b_penalty"] = 1.0
        c["l1b_flags"]   = []
        c["l1b_status"]  = "pass"
        survivors.append(c)

    logger.info(
        f"L1b: {len(candidates)} in → {len(survivors)} pass ({rejected} hard-rejected)"
    )
    return survivors


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1c — SKILL MATCH (NLP STRING + SYNONYM MATCH)
# ──────────────────────────────────────────────────────────────────────────────
from .dictionary import expand_skill, JD_REQUIREMENTS  # noqa: E402 (import after module-level constants)


def _candidate_search_text(c: dict) -> str:
    """Build a single normalized text blob from skills, work descriptions, and profile."""
    parts = []
    # Skills list
    skills = c.get("skills", [])
    if isinstance(skills, list):
        for s in skills:
            parts.append(str(s.get("name", "")) if isinstance(s, dict) else str(s))
    elif isinstance(skills, str):
        parts.append(skills)
    # Work history titles + descriptions
    for exp in utils._iter_work_history(c):
        if not isinstance(exp, dict):
            continue
        for field in ("title", "role", "description"):
            v = exp.get(field, "")
            if v:
                parts.append(str(v))
    # Profile summary / headline
    profile = c.get("profile") or {}
    if isinstance(profile, dict):
        for field in ("summary", "headline", "title", "current_title"):
            v = profile.get(field, "")
            if v:
                parts.append(str(v))
    # Project descriptions
    for proj in (c.get("projects") or []):
        if isinstance(proj, dict):
            d = proj.get("description", "")
            if d:
                parts.append(str(d))
    return " ".join(p for p in parts if p).lower()


def _skill_in_text(skill: str, text: str) -> bool:
    """Return True if the skill (or any synonym) appears in text as a whole word/phrase."""
    for form in expand_skill(skill):
        pattern = r"(?<![a-z0-9\-])" + re.escape(form) + r"(?![a-z0-9\-])"
        if re.search(pattern, text):
            return True
    return False


def compute_skill_match(candidate: dict, requirements: dict) -> tuple:
    """
    Returns fraction of JD requirements satisfied.
    Checks skills[] names AND career description text.
    """
    cand_skills = {
        s["name"].lower().strip()
        for s in candidate.get("skills", [])
    }

    cand_desc = " ".join(
        job.get("description", "")
        for job in candidate.get("career_history", [])
    ).lower()

    satisfied = 0
    results = {}

    for req_name, req_data in requirements.items():
        skill_hit = bool(cand_skills & req_data["skill_tokens"])
        desc_hit = any(term in cand_desc for term in req_data["desc_terms"])

        if skill_hit or desc_hit:
            satisfied += 1
            results[req_name] = "skill" if skill_hit else "desc"
        else:
            results[req_name] = "MISS"

    score = satisfied / len(requirements)
    return score, results


def l1c_skill_match(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 1c — NLP string skill match using the synonym dictionary.

    Score:
        If both required and bonus skills exist:
            l1c_score = 0.75 * (req_matched / req_total) + 0.25 * (bon_matched / bon_total)
        If only required:  l1c_score = req_matched / req_total
        If only bonus:     l1c_score = bon_matched / bon_total
        If no skills in JD: l1c_score = 1.0

    Hard-reject gate: If JD has explicit_required skills and candidate matches NONE → hard reject.
    """
    explicit_req = jd.get("explicit_required", [])
    inferred_req = jd.get("inferred_required", [])
    required     = explicit_req + inferred_req
    bonus        = jd.get("explicit_bonus", []) + jd.get("inferred_bonus", [])
    n_req, n_bon = len(required), len(bonus)
    n_explicit   = len(explicit_req)

    for c in candidates:
        text = _candidate_search_text(c)

        req_match    = {s: _skill_in_text(s, text) for s in required}
        matched_req  = [s for s, m in req_match.items() if m]
        missing_req  = [s for s, m in req_match.items() if not m]
        matched_bon  = [s for s in bonus if _skill_in_text(s, text)]
        matched_expl = [s for s in explicit_req if req_match.get(s)]

        req_ratio = len(matched_req) / n_req if n_req else 1.0
        bon_ratio = len(matched_bon) / n_bon if n_bon else 0.0

        if n_req > 0 and n_bon > 0:
            score = 0.75 * req_ratio + 0.25 * bon_ratio
        elif n_req > 0:
            score = req_ratio
        elif n_bon > 0:
            score = bon_ratio
        else:
            score = 1.0

        c["l1c_score"]             = round(score, 4)
        c["l1c_matched_required"]  = matched_req
        c["l1c_missing_required"]  = missing_req
        c["l1c_matched_bonus"]     = matched_bon
        c["l1c_matched_explicit"]  = matched_expl

        jd_req_score, jd_req_results = compute_skill_match(c, JD_REQUIREMENTS)
        c["jd_req_score"]   = round(jd_req_score, 4)
        c["jd_req_results"] = jd_req_results

    before = len(candidates)

    # Hard reject: zero explicit_required matches when JD has explicit skills
    if n_explicit > 0:
        candidates = [c for c in candidates if c.get("l1c_matched_explicit")]
        dropped_explicit = before - len(candidates)
        if dropped_explicit:
            logger.info(
                f"L1c hard-reject: {dropped_explicit} candidates matched 0/{n_explicit} "
                f"explicit required skills → removed"
            )

    # Optional score gate
    if C.L1C_MIN_SKILL_MATCH > 0:
        before_score = len(candidates)
        candidates = [c for c in candidates if c.get("l1c_score", 0.0) >= C.L1C_MIN_SKILL_MATCH]
        if len(candidates) < before_score:
            logger.info(
                f"L1c score gate: {before_score} → {len(candidates)} "
                f"(dropped {before_score - len(candidates)} below min_score={C.L1C_MIN_SKILL_MATCH})"
            )

    avg = sum(c.get("l1c_score", 0.0) for c in candidates) / max(len(candidates), 1)
    logger.info(
        f"L1c: {before} in → {len(candidates)} pass "
        f"(avg_score={avg:.3f}, explicit={n_explicit}, req_skills={n_req}, bonus_skills={n_bon})"
    )
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1d — INFERRED SKILL MATCH + LEFTOVER PENALTY (soft, no rejections)
# ──────────────────────────────────────────────────────────────────────────────

def l1d_inferred_match(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 1d — Inferred skill match with leftover penalty.

    Matches all inferred JD skills (inferred_required + inferred_bonus) against
    each candidate's full text.  No candidates are rejected.

    Score:
      inferred_ratio  = matched_inferred / n_inferred   (1.0 when JD has none)
      leftover_count  = n_inferred − matched_inferred   (unmatched inferred)
      leftover_penalty = 0.01 × leftover_count
      l1d_score       = max(0, inferred_ratio − leftover_penalty)

    Attaches per-candidate:
      l1d_matched_inferred   list[str]  — inferred skills found
      l1d_unmatched_inferred list[str]  — inferred skills not found
      l1d_inferred_ratio     float[0,1] — matched / total
      l1d_leftover_count     int        — count of unmatched inferred skills
      l1d_score              float[0,1] — net score forwarded to L2
    """
    inferred = jd.get("inferred_required", []) + jd.get("inferred_bonus", [])
    n_inferred = len(inferred)

    for c in candidates:
        text = _candidate_search_text(c)
        matched   = [s for s in inferred if _skill_in_text(s, text)]
        unmatched = [s for s in inferred if not _skill_in_text(s, text)]
        ratio     = len(matched) / n_inferred if n_inferred > 0 else 1.0
        leftover  = len(unmatched)
        penalty   = 0.01 * leftover
        score     = max(0.0, ratio - penalty)

        c["l1d_matched_inferred"]   = matched
        c["l1d_unmatched_inferred"] = unmatched
        c["l1d_inferred_ratio"]     = round(ratio, 4)
        c["l1d_leftover_count"]     = leftover
        c["l1d_score"]              = round(score, 4)

    avg = sum(c.get("l1d_inferred_ratio", 1.0) for c in candidates) / max(len(candidates), 1)
    logger.info(
        f"L1d: {len(candidates)} candidates — {n_inferred} inferred skills "
        f"(avg_inferred_match={avg:.3f})"
    )
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# L3 GLOBAL GATE — 75% (top 50% + random 25%)
# Applied after L3 fuzzy scoring, before expensive L4 semantic encoding.
# ──────────────────────────────────────────────────────────────────────────────
def l3_gate(candidates: List[dict], seed: int = 42) -> List[dict]:
    """
    Keep the top 50% by l3_score plus a random 25% from the remainder = 75%.
    """
    if not candidates:
        return candidates

    ranked = sorted(candidates, key=lambda c: c.get("l3_score", 0.0), reverse=True)
    n = len(ranked)
    top_k = math.ceil(n * C.GATE_TOP_FRACTION)
    top = ranked[:top_k]
    bottom = ranked[top_k:]

    rng = random.Random(seed)
    rand_k = math.ceil(n * C.GATE_RANDOM_FRACTION)
    rescued = rng.sample(bottom, min(rand_k, len(bottom))) if bottom else []

    gated = top + rescued
    logger.info(
        f"L3 gate: {n} → {len(gated)} "
        f"(top {len(top)} + random {len(rescued)} = {len(gated)/max(n,1)*100:.0f}%)"
    )
    return gated


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 2 — TABLE EXTRACT  (29-column row per candidate)
# ──────────────────────────────────────────────────────────────────────────────

_PRODUCTION_KW: Dict[str, float] = {
    "deployed": 0.20, "deploy": 0.15, "deployment": 0.15,
    "shipped": 0.18, "ship": 0.15, "shipping": 0.15,
    "production": 0.20,
    "real users": 0.25, "live users": 0.20, "end users": 0.15,
    "at scale": 0.15, "large scale": 0.15,
    "a/b test": 0.15, "ab test": 0.15, "ab testing": 0.15,
    "rollout": 0.12, "launched": 0.15, "launch": 0.12,
    "serving": 0.12,
    "qps": 0.20,
    "p95 latency": 0.22,
    "p99 latency": 0.22,
    "ranking pipeline": 0.22,
    "retrieval system": 0.22,
    "search at scale": 0.25,
    "online serving": 0.20,
}

_ARCHITECTURE_KW: Dict[str, float] = {
    "designed system": 0.20, "system design": 0.18,
    "architected": 0.20, "architecture": 0.12,
    "distributed system": 0.15, "distributed": 0.10,
    "microservices": 0.12, "microservice": 0.12,
    "scalable": 0.10, "scalability": 0.10,
    "high availability": 0.12, "fault tolerant": 0.12,
    "low latency": 0.10, "throughput": 0.10,
    "system architecture": 0.18, "pipeline design": 0.12,
}

_TESTING_EVAL_KW: Dict[str, float] = {
    "ndcg": 0.25, "mrr": 0.25,
    "mean average precision": 0.25, "map": 0.20,
    "a/b test": 0.20, "ab test": 0.20, "ab testing": 0.20,
    "recall@k": 0.20, "recall@": 0.18,
    "benchmark": 0.15,
    "precision@": 0.15, "precision": 0.12,
    "evaluation": 0.10, "metrics": 0.10,
    "beir": 0.20, "trec": 0.20,
    "offline evaluation": 0.15, "online evaluation": 0.15,
    "f1 score": 0.12,
}

_EMBEDDING_TOOLS: Dict[str, float] = {
    "sentence-transformers": 1.00,
    "fastembed":             1.00,
    "huggingface":           0.90,
    "cohere":                0.90,
}

_VECTOR_DB_TOOLS: Dict[str, float] = {
    "faiss":         0.85,
    "qdrant":        0.85,
    "milvus":        0.85,
    "weaviate":      0.85,
    "pinecone":      0.85,
    "chromadb":      0.85,
    "pgvector":      0.85,
    "lancedb":       0.85,
    "vespa":         0.85,
    "elasticsearch": 0.80,
    "opensearch":    0.80,
    "solr":          0.70,
}

_RANKING_TOOLS: Dict[str, float] = {
    "flashrank":     0.90,
    "colbert":       0.90,
    "cross-encoder": 0.85,
    "bm25":          0.80,
    "reranker":      0.80,
}

_NLP_MODEL_TOOLS: Dict[str, float] = {
    "bert":        0.70,
    "llama":       0.70,
    "spacy":       0.65,
    "mistral":     0.65,
    "openai":      0.65,
    "gensim":      0.60,
    "pytorch":     0.60,
    "anthropic":   0.60,
    "nltk":        0.55,
    "tensorflow":  0.55,
    "keras":       0.50,
    "onnx":        0.50,
    "tensorrt":    0.50,
}

_DEPLOYMENT_TOOLS: Dict[str, float] = {
    "docker":     0.50,
    "kubernetes": 0.50,
    "triton":     0.50,
    "bentoml":    0.50,
    "ray":        0.45,
    "mlflow":     0.45,
    "kubeflow":   0.45,
    "wandb":      0.40,
    "dvc":        0.40,
}

_ORCHESTRATION_TOOLS: Dict[str, float] = {
    "langchain":  0.30,
    "llamaindex": 0.30,
    "haystack":   0.30,
    "crewai":     0.25,
}

_CLOUD_TOOLS: Dict[str, float] = {
    "sagemaker":  0.20,
    "bigquery":   0.20,
    "databricks": 0.20,
    "snowflake":  0.20,
    "redshift":   0.20,
}

_TOOLS_WEIGHTED: Dict[str, float] = {
    **_EMBEDDING_TOOLS,
    **_VECTOR_DB_TOOLS,
    **_RANKING_TOOLS,
    **_NLP_MODEL_TOOLS,
    **_DEPLOYMENT_TOOLS,
    **_ORCHESTRATION_TOOLS,
    **_CLOUD_TOOLS,
}

_IR_DOMAIN_TERMS: frozenset = frozenset({
    "bm25", "faiss", "rerank", "retrieval", "ranking",
    "vector search", "embedding",
})

_JD_REQ_TITLES: frozenset = frozenset({
    "ai engineer", "senior ai engineer", "staff ai engineer", "principal ai engineer",
    "ml engineer", "machine learning engineer", "senior ml engineer",
    "principal ml engineer", "staff ml engineer",
    "applied scientist", "applied ai engineer", "applied ai",
    "applied machine learning engineer", "applied research scientist",
    "nlp engineer", "nlp scientist", "natural language processing engineer",
    "search engineer", "senior search engineer", "search scientist",
    "ranking engineer", "relevance engineer", "information retrieval engineer",
    "research engineer", "senior research engineer", "ml research engineer",
    "ai tech lead", "ml tech lead", "ai lead", "ml lead",
    "ai architect", "ml architect",
})

_CONSULTING_FIRMS: frozenset = frozenset({
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "hcl technologies", "tech mahindra",
    "mphasis", "mindtree", "hexaware", "ltimindtree", "lti mindtree",
    "l&t infotech", "kpmg", "deloitte", "pwc", "ernst & young", "ey",
    "mckinsey", "boston consulting group", "bcg", "bain",
    "niit technologies", "zensar", "cyient", "persistent systems",
})

_CONSULTING_INDUSTRIES: frozenset = frozenset({
    "it services", "consulting", "professional services", "outsourcing",
    "it consulting", "management consulting", "technology services",
    "bpo", "kpo", "ites", "information technology services",
})

_RESEARCH_KW: frozenset = frozenset({
    "paper", "published", "arxiv", "proceedings", "neurips", "nips",
    "icml", "iclr", "cvpr", "acl", "emnlp", "journal", "preprint",
    "citation", "dissertation", "thesis", "conference paper", "research paper",
    "peer reviewed", "peer-reviewed",
})


def _safe_float(val, default=None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _norm(val: Optional[float], cap: float) -> float:
    if val is None or val < 0:
        return 0.0
    return min(val, cap) / cap


def _kw_accumulate(text: str, kw_dict: Dict[str, float]) -> float:
    score = 0.0
    for kw, weight in kw_dict.items():
        if kw in text:
            score += weight
    return min(score, 1.0)


def _work_text_lower(c: dict) -> str:
    parts = []
    for exp in utils._iter_work_history(c):
        if isinstance(exp, dict):
            d = exp.get("description", "")
            if d:
                parts.append(str(d))
    return " ".join(parts).lower()


def _parse_date_to_dt(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    s = str(val).strip()
    for fmt, ln in (
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%d", 10),
        ("%Y/%m/%d", 10),
        ("%d-%m-%Y", 10),
        ("%Y-%m", 7),
        ("%Y", 4),
    ):
        try:
            return datetime.strptime(s[:ln], fmt)
        except ValueError:
            continue
    return None


def _redrob_cumulative(signals: dict) -> float:
    def _bool(key: str) -> float:
        v = signals.get(key)
        return 1.0 if v is True or str(v).lower() in ("1", "true", "yes") else 0.0

    github_raw = _safe_float(signals.get("github_activity_score"))
    github_val = 0.5 if github_raw == -1.0 else _norm(github_raw, 100.0)

    offer_raw = _safe_float(signals.get("offer_acceptance_rate"))
    offer_val = 0.5 if offer_raw == -1.0 else _norm(offer_raw, 1.0)

    speed_h = _safe_float(signals.get("average_response_time_hours"))
    if speed_h is None or speed_h < 0:
        speed_val = 0.5
    else:
        speed_val = max(0.0, 1.0 - _norm(speed_h, 48.0))

    return round(min(max((
        _norm(_safe_float(signals.get("profile_completeness_score")), 100.0) * 0.08
        + _norm(_safe_float(signals.get("profile_recency_score")), 1.0)      * 0.12
        + _bool("open_to_work")                                               * 0.08
        + _norm(_safe_float(signals.get("applications_submitted_30d")), 30.0) * 0.06
        + _norm(_safe_float(signals.get("recruiter_response_rate")), 1.0)    * 0.07
        + speed_val                                                            * 0.05
        + _norm(_safe_float(signals.get("linkedin_connections")), 500.0)     * 0.04
        + _norm(_safe_float(signals.get("linkedin_endorsements")), 100.0)    * 0.04
        + github_val                                                           * 0.10
        + _norm(_safe_float(signals.get("search_appearances_30d")), 50.0)    * 0.05
        + _norm(_safe_float(signals.get("saved_by_recruiters")), 20.0)       * 0.05
        + _norm(_safe_float(signals.get("profile_views_30d")), 100.0)        * 0.04
        + _norm(_safe_float(signals.get("interview_conversion_rate")), 1.0)  * 0.07
        + offer_val                                                            * 0.06
        + _bool("is_verified")                                                * 0.05
        + _norm(_safe_float(signals.get("linkedin_profile_score")), 100.0)   * 0.04
    ), 0.0), 1.0), 4)


def _build_table_row(
    c: dict,
    jd_inferred: List[str],
    n_req: int,
) -> dict:
    profile    = c.get("profile") or {}
    signals    = c.get("redrob_signals") or {}
    edu_list   = c.get("education") or []
    work_hist  = list(utils._iter_work_history(c))
    skills_raw = c.get("skills") or []
    certs      = c.get("certifications") or []
    langs      = c.get("languages") or []

    candidate_id = str(c.get("candidate_id") or c.get("id") or "")
    total_exp = utils.get_total_experience_years(c)

    loc     = str(profile.get("location") or "").strip()
    country = str(profile.get("country") or "").strip()
    location = f"{loc}, {country}" if loc and country else loc or country or ""

    sal_range   = signals.get("expected_salary_range_inr_lpa") or {}
    max_salary  = _safe_float(sal_range.get("max"))
    min_salary  = _safe_float(sal_range.get("min"))
    _sal_inverted = (
        min_salary is not None and max_salary is not None and min_salary > max_salary
    )

    assessment = signals.get("skill_assessment_scores")
    if isinstance(assessment, dict) and assessment:
        skill_assessment_score = round(sum(assessment.values()) / len(assessment) / 100.0, 4)
    else:
        skill_assessment_score = 0.0

    matched_req = c.get("l1c_matched_required") or []
    skill_match_score = round(len(matched_req) / n_req, 4) if n_req > 0 else 0.0

    cand_text = _candidate_search_text(c)
    inferred_skill_match_score = sum(
        1 for skill in jd_inferred if _skill_in_text(skill, cand_text)
    )

    redrob_cumulative = _redrob_cumulative(signals)

    industry = str(profile.get("current_industry") or profile.get("industry") or "")

    is_phd = any(
        any(kw in str(e.get("degree", "")).lower() for kw in ("phd", "doctor", "d.phil"))
        for e in edu_list if isinstance(e, dict)
    )

    wtext = _work_text_lower(c)
    production_score        = round(_kw_accumulate(wtext, _PRODUCTION_KW), 4)
    architecture_score      = round(_kw_accumulate(wtext, _ARCHITECTURE_KW), 4)
    testing_evaluation_score = round(_kw_accumulate(wtext, _TESTING_EVAL_KW), 4)

    no_certifications = len(certs)
    no_languages = len(langs)

    github_raw = _safe_float(signals.get("github_activity_score"))
    open_source_score = (
        0.0 if (github_raw is None or github_raw == -1.0)
        else round(github_raw / 100.0, 4)
    )

    research_text = wtext + " " + " ".join(
        str(cert.get("name") or "").lower() for cert in certs if isinstance(cert, dict)
    )
    research_published = True if any(kw in research_text for kw in _RESEARCH_KW) else None

    skills_text = " ".join(
        str(s.get("name") if isinstance(s, dict) else s).lower() for s in skills_raw
    )
    full_text = skills_text + " " + wtext
    tools_score = sum(w for t, w in _TOOLS_WEIGHTED.items() if t in full_text)
    ir_domain_score = sum(1 for t in _IR_DOMAIN_TERMS if t in full_text)
    orchestration_score = sum(w for t, w in _ORCHESTRATION_TOOLS.items() if t in full_text)

    consulting_only = bool(work_hist) and all(
        (
            str(exp.get("industry") or "").strip().lower() in _CONSULTING_INDUSTRIES
            or str(exp.get("company") or "").strip().lower() in _CONSULTING_FIRMS
        )
        for exp in work_hist if isinstance(exp, dict)
    )

    current_job = next(
        (e for e in work_hist if isinstance(e, dict) and e.get("is_current") is True),
        None,
    )
    if current_job is None and work_hist:
        current_job = max(
            (e for e in work_hist if isinstance(e, dict)),
            key=lambda e: str(e.get("start_date") or e.get("start_year") or ""),
            default=None,
        )
    last_career_tenure  = _safe_float((current_job or {}).get("duration_months"))
    last_career_company = str((current_job or {}).get("company") or "")

    offer_raw       = _safe_float(signals.get("offer_acceptance_rate"))
    no_offer_history = offer_raw == -1.0

    edu_ends = [
        _parse_year(e.get("end_year") or e.get("year"))
        for e in edu_list if isinstance(e, dict)
    ]
    edu_ends = [y for y in edu_ends if y is not None]

    career_starts = [
        _parse_year(e.get("start_date") or e.get("start_year"))
        for e in work_hist if isinstance(e, dict)
    ]
    career_starts = [y for y in career_starts if y is not None]

    education_overlap   = False
    edu_career_gap_flag = False
    if edu_ends and career_starts:
        latest_edu_end       = max(edu_ends)
        first_career_start   = min(career_starts)
        education_overlap    = latest_edu_end > first_career_start
        gap                  = first_career_start - latest_edu_end
        edu_career_gap_flag  = gap > 1.5
    edu_career_gap_flag = edu_career_gap_flag or bool(c.get("edu_career_gap_flag"))

    apps_30d      = _safe_float(signals.get("applications_submitted_30d"))
    response_rate = _safe_float(signals.get("recruiter_response_rate"))
    low_engagement_flag = (
        apps_30d is not None and apps_30d < 1
        and response_rate is not None and response_rate < 0.2
    )

    fabrication_bandwidth = float(c.get("fabrication_bandwidth_score") or 0.0) / 100.0
    possible_fabrication = bool(c.get("possible_fabrication"))

    notice_period_days = (
        signals.get("notice_period_days")
        or c.get("notice_period_days")
        or c.get("notice_period")
    )
    l1d_inferred_score = round(float(c.get("l1d_score", 1.0)), 4)
    l1c_fwd_score = round(float(c.get("l1c_score", 0.0)), 4)
    title = (
        str(profile.get("current_title") or profile.get("title") or profile.get("headline") or "")
        .strip().lower()
    )
    if not title and current_job:
        title = str(current_job.get("title") or "").strip().lower()

    return {
        "candidate_id":               candidate_id,
        "total_exp":                  total_exp,
        "location":                   location,
        "max_salary":                 max_salary,
        "skill_assessment_score":     skill_assessment_score,
        "skill_match_score":          skill_match_score,
        "inferred_skill_match_score": inferred_skill_match_score,
        "redrob_cumulative":          redrob_cumulative,
        "industry":                   industry,
        "is_phd":                     is_phd,
        "production_score":           production_score,
        "architecture_score":         architecture_score,
        "testing_evaluation_score":   testing_evaluation_score,
        "no_certifications":          no_certifications,
        "no_languages":               no_languages,
        "open_source_score":          open_source_score,
        "research_published":         research_published,
        "tools_score":                tools_score,
        "ir_domain_score":            ir_domain_score,
        "orchestration_score":        orchestration_score,
        "consulting_only":            consulting_only,
        "last_career_tenure":         last_career_tenure,
        "last_career_company":        last_career_company,
        "no_offer_history":           no_offer_history,
        "education_overlap":          education_overlap,
        "edu_career_gap_flag":        edu_career_gap_flag,
        "low_engagement_flag":        low_engagement_flag,
        "fabrication_bandwidth":      fabrication_bandwidth,
        "possible_fabrication":       possible_fabrication,
        "skill_career_domain_mismatch": bool(c.get("skill_career_domain_mismatch")),
        "second_undergrad_after_first": bool(c.get("second_undergrad_after_first")),
        "l1c_score":                  l1c_fwd_score,
        "l1d_inferred_score":         l1d_inferred_score,
        "notice_period_days":         notice_period_days,
        "title":                      title,
    }


def l2_table_extract(
    candidates: List[dict],
    jd: dict,
) -> List[dict]:
    """
    Layer 2 — Table Extract:
    Builds a 31-column `table_row` dict on every candidate and attaches it as
    c['table_row'].  No candidates are filtered; the full list passes through.
    """
    n_explicit_req = len(jd.get("explicit_required", []))
    jd_inferred    = jd.get("inferred_required", []) + jd.get("inferred_bonus", [])

    for c in candidates:
        c["table_row"] = _build_table_row(c, jd_inferred, n_explicit_req)

    logger.info(f"L2 table extract: {len(candidates)} rows built (31 cols each)")
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 — SUGENO FUZZY INFERENCE SYSTEM  (reads table_row from L2)
# ──────────────────────────────────────────────────────────────────────────────

_FUZZY_RULES = [
    #  a    b    c    d    e    f    g    h    out
    ("H", "H", "H", "H", "H", "H", "H", "L", 1.00),  # R01
    ("H", "H", "H", "L", "H", "H", "H", "L", 0.95),  # R02
    ("H", "H", "H", "H", "H", "H", "P", "L", 0.91),  # R03
    ("H", "H", "H", "H", "H", "W", "H", "L", 0.82),  # R04
    ("H", "H", "M", "H", "H", "H", "H", "L", 0.95),  # R05
    ("H", "M", "H", "H", "H", "H", "H", "L", 0.95),  # R06
    ("M", "H", "H", "H", "H", "H", "H", "L", 0.97),  # R07
    ("H", "H", "H", "H", "H", "H", "P", "L", 0.91),  # R08
    ("H", "H", "H", "H", "M", "H", "H", "L", 0.95),  # R09
    ("H", "H", "H", "H", "H", "H", "L", "L", 0.55),  # R10
    ("H", "H", "H", "H", "H", "L", "H", "L", 0.75),  # R11
    ("H", "H", "M", "L", "H", "H", "P", "L", 0.78),  # R12
    ("L", "H", "H", "H", "H", "H", "M", "L", 0.85),  # R13
    ("H", "M", "M", "H", "H", "H", "M", "L", 0.77),  # R14
    ("H", "H", "L", "H", "H", "H", "M", "L", 0.77),  # R15
    ("H", "H", "H", "H", "L", "H", "M", "M", 0.77),  # R16
    ("H", "M", "M", "H", "H", "H", "L", "M", 0.45),  # R17
    ("H", "H", "M", "L", "L", "H", "L", "M", 0.42),  # R18
    ("H", "M", "H", "H", "L", "W", "M", "M", 0.62),  # R19
    ("L", "M", "M", "L", "H", "H", "M", "M", 0.68),  # R20
    ("H", "L", "M", "H", "H", "H", "M", "M", 0.72),  # R21
    ("H", "H", "L", "H", "L", "W", "L", "M", 0.35),  # R22
    ("H", "L", "L", "H", "L", "H", "M", "M", 0.53),  # R23
    ("L", "M", "L", "L", "L", "H", "L", "M", 0.27),  # R24
    ("H", "M", "L", "L", "L", "W", "L", "M", 0.24),  # R25
    ("H", "L", "L", "H", "H", "W", "L", "H", 0.27),  # R26
    ("L", "L", "M", "L", "L", "H", "L", "H", 0.22),  # R27
    ("H", "L", "L", "H", "L", "L", "L", "H", 0.16),  # R28
    ("H", "M", "L", "L", "L", "L", "L", "H", 0.17),  # R29
    ("L", "L", "L", "L", "L", "H", "L", "H", 0.20),  # R30
    ("H", "L", "L", "H", "L", "L", "L", "H", 0.16),  # R31
    ("L", "L", "L", "L", "L", "L", "L", "H", 0.04),  # R32
]

_FIS_WEIGHTS = {"g": 0.40, "b": 0.20, "f": 0.05, "e": 0.10, "a": 0.05, "c": 0.10, "d": 0.05, "h": 0.05}

_SOFT_PENALTY_COLS: tuple = (
    "skill_career_domain_mismatch",
    "education_overlap",
    "second_undergrad_after_first",
    "edu_career_gap_flag",
)


def _bell_mf(x: float, center: float, width: float, slope: float = 2.0) -> float:
    if width <= 0:
        return 1.0 if abs(x - center) < 1e-9 else 0.0
    return 1.0 / (1.0 + abs((x - center) / width) ** (2.0 * slope))


def _pct(sv: list, p: float) -> float:
    n = len(sv)
    if n == 0:
        return 0.0
    if n == 1:
        return sv[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def _compute_conditions(row: dict, n_inferred: int) -> Dict[str, float]:
    """
    Derive 8 crisp conditions a–h from a L2 table_row. All outputs in [0, 1].

    a — experience in JD sweet spot (5–9 yr = 1.0, 3–12 yr = 0.5, else 0.0)
    b — explicit skill signal (skill_match + skill_assessment)
    c — inferred signal + platform engagement
    d — IR domain signal (0.0 / 0.5 / 1.0 based on retrieval-specific term hits)
    e — absence of disqualifying traits; −0.25 if framework-heavy with no production
    f — profile integrity (deductions for anomaly flags)
    g — technical breadth (production + arch + testing + tools + open-source/research)
    h — soft-penalty union (1.0 = any flag fires; ideal candidate has h=0.0)
    """
    exp = float(row.get("total_exp") or 0.0)
    a = 1.0 if 5.0 <= exp <= 9.0 else (0.5 if 3.0 <= exp <= 12.0 else 0.0)

    b = (0.8 * float(row.get("skill_match_score") or 0.0)
         + 0.2 * float(row.get("skill_assessment_score") or 0.0))

    inf_norm = (
        min(float(row.get("inferred_skill_match_score") or 0.0) / n_inferred, 1.0)
        if n_inferred > 0 else 0.0
    )
    c = (inf_norm + float(row.get("redrob_cumulative") or 0.0)) / 2.0

    ir_hits = float(row.get("ir_domain_score") or 0.0)
    d = 1.0 if ir_hits >= 3 else (0.5 if ir_hits >= 1 else 0.0)

    stagnant_lead = (
        float(row.get("last_career_tenure") or 0.0) > 60.0
        and float(row.get("production_score") or 0.0) < 0.25
    )
    pure_researcher = (
        row.get("research_published") is True
        and float(row.get("production_score") or 0.0) < 0.20
    )
    e = (
        (0.0 if row.get("is_phd") else 1.0)
        + (0.0 if row.get("consulting_only") else 1.0)
        + (0.0 if stagnant_lead else 1.0)
        + (0.0 if pure_researcher else 1.0)
    ) / 4.0
    total_tool = float(row.get("tools_score") or 0.0)
    orch_score = float(row.get("orchestration_score") or 0.0)
    framework_ratio = orch_score / total_tool if total_tool > 0.0 else 0.0
    if framework_ratio > 0.6 and float(row.get("production_score") or 0.0) < 0.15:
        e = max(0.0, e - 0.25)

    f = 1.0
    if row.get("low_engagement_flag"):         f -= 0.20
    if row.get("active_before_signup"):        f -= 0.20
    f -= float(row.get("fabrication_bandwidth") or 0.0) * 0.20
    f = max(0.0, f)

    tools_norm      = min(float(row.get("tools_score") or 0.0) / 3.0, 1.0)
    open_or_research = max(
        float(row.get("open_source_score") or 0.0),
        1.0 if row.get("research_published") is True else 0.0,
    )
    g = (
        float(row.get("production_score") or 0.0)*0.35
        + float(row.get("architecture_score") or 0.0)*0.25
        + float(row.get("testing_evaluation_score") or 0.0)*0.25
        + tools_norm*0.05
        + open_or_research*0.10
    )

    h = 1.0 if any(row.get(flag) for flag in _SOFT_PENALTY_COLS) else 0.0

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h}


def _calibrate_mf_params(all_conds: List[Dict[str, float]]) -> Dict[str, dict]:
    MIN_W = 0.05
    params: Dict[str, dict] = {}

    for var in ("b", "c", "e", "f", "g"):
        sv = sorted(c[var] for c in all_conds)
        p10 = _pct(sv, 10); p25 = _pct(sv, 25); p50 = _pct(sv, 50)
        p65 = _pct(sv, 65); p75 = _pct(sv, 75); p90 = _pct(sv, 90)

        w_lo = max((p25 - p10) / 2.0, MIN_W)
        w_md = max((p75 - p25) / 2.0, MIN_W)
        w_hi = max((p90 - p75) / 2.0, MIN_W)

        if var == "g":
            params[var] = {
                "L": (p10, w_lo, 2.0),
                "M": (p50, w_md, 2.0),
                "P": (p65, max((p90 - p50) / 3.0, MIN_W), 2.0),
                "H": (p90, w_hi, 2.0),
            }
        elif var == "f":
            params[var] = {
                "L": (p10, w_lo, 2.0),
                "W": (p25, max((p65 - p10) / 3.0, MIN_W), 2.0),
                "H": (p90, w_hi, 2.0),
            }
        else:
            params[var] = {
                "L": (p10, w_lo, 2.0),
                "M": (p50, w_md, 2.0),
                "H": (p90, w_hi, 2.0),
            }

    params["a"] = {"L": (0.0, 0.15, 2.0), "M": (0.5, 0.15, 2.0), "H": (1.0, 0.15, 2.0)}
    params["d"] = {"L": (0.0, 0.15, 2.0), "M": (0.5, 0.20, 2.0), "H": (1.0, 0.15, 2.0)}

    return params


def _fuzzify(conds: Dict[str, float], params: Dict[str, dict]) -> Dict[str, dict]:
    return {
        var: {
            lvl: _bell_mf(val, center, width, slope)
            for lvl, (center, width, slope) in params[var].items()
        }
        for var, val in conds.items()
    }


def _sugeno_infer(mf_vals: Dict[str, dict]):
    ws = 0.0
    ts = 0.0
    for a_l, b_l, c_l, d_l, e_l, f_l, g_l, _, out in _FUZZY_RULES:
        s = min(
            mf_vals["a"][a_l], mf_vals["b"][b_l], mf_vals["c"][c_l],
            mf_vals["d"][d_l], mf_vals["e"][e_l], mf_vals["f"][f_l],
            mf_vals["g"][g_l],
        )
        ws += s * out
        ts += s
    return (ws / ts if ts > 1e-9 else None), ts


def _linear_fallback(conds: Dict[str, float]) -> float:
    return sum(_FIS_WEIGHTS[k] * v for k, v in conds.items())


def _apply_ceilings(mf_vals: Dict[str, dict], raw_score: float) -> float:
    gv = mf_vals["g"]
    fv = mf_vals["f"]
    g_lo = gv["L"];      g_hi = gv["H"];  g_pa = gv.get("P", 0.0);  g_md = gv.get("M", 0.0)
    f_hi = fv["H"];      f_wa = fv.get("W", 0.0);  f_lo = fv["L"]

    if g_lo > max(g_hi, g_pa, g_md):
        raw_score = min(raw_score, 0.55)
    elif max(f_wa, f_lo) > f_hi and max(g_hi, g_pa) > 0.4:
        raw_score = min(raw_score, 0.82)
    return raw_score


def _post_fis_adjust(raw_score: float, row: dict, h: float = 0.0) -> float:
    if h > 0.5:                                          raw_score *= 0.90
    if row.get("research_published") is True:            raw_score *= 1.05
    if 5.0 <= float(row.get("total_exp") or 0) <= 9.0:  raw_score *= 1.08
    return max(0.0, min(1.0, raw_score))


def _fuzzy_class(score: float) -> str:
    if score >= 0.85:  return "strong_fit"
    if score >= 0.70:  return "good_fit"
    if score >= 0.55:  return "moderate_fit"
    return "weak_fit"


def _build_fuzzy_reasoning(row: dict, conds: Dict[str, float], score: float) -> str:
    flags = []
    if row.get("possible_fabrication"):                  flags.append("fabrication")
    if row.get("consulting_only"):                        flags.append("consulting_only")
    if row.get("low_engagement_flag"):                    flags.append("low_engagement")
    if row.get("active_before_signup"):                   flags.append("active_before_signup")
    if conds.get("h", 0.0) > 0.5:                        flags.append("soft_penalty")
    if 5.0 <= float(row.get("total_exp") or 0) <= 9.0:  flags.append("ideal_exp")
    if row.get("research_published"):                     flags.append("research")
    tag = f"[{', '.join(flags)}]" if flags else "clean"
    a, b, c, d, e, f, g, h = (conds.get(k, 0.0) for k in "abcdefgh")
    return (
        f"score={score:.3f} "
        f"a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f} e={e:.2f} f={f:.2f} g={g:.2f} h={h:.0f} "
        f"{tag}"
    )


def l3_fuzzy_score(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 3 — Sugeno Fuzzy Inference System.
    Input:  candidates with table_row from L2.
    Output: same list with l3_score, l3_class, l3_reasoning attached.
            Notice period bonus (+0.05) and title-match bonus (+0.01) applied post-FIS.
    """
    n_inferred  = len(jd.get("inferred_required", []) + jd.get("inferred_bonus", []))

    active: List[dict] = []
    for c in candidates:
        row = c.get("table_row")
        if not isinstance(row, dict):
            logger.warning(
                f"L3 fuzzy: no table_row on {c.get('candidate_id', '?')} — assign 0.0"
            )
            c["l3_score"] = 0.0; c["l3_class"] = "no_table_row"
            c["l3_reasoning"] = "missing_l2_output"
            continue
        active.append(c)

    if not active:
        logger.info(f"L3 fuzzy: no valid candidates with table_row")
        return candidates

    all_conds: List[Dict[str, float]] = [
        _compute_conditions(c["table_row"], n_inferred) for c in active
    ]

    mf_params = _calibrate_mf_params(all_conds)

    scored: List[float] = []
    for c, conds in zip(active, all_conds):
        row = c["table_row"]

        h         = conds["h"]
        fis_conds = {k: v for k, v in conds.items() if k != "h"}

        mf_vals = _fuzzify(fis_conds, mf_params)
        raw, _  = _sugeno_infer(mf_vals)

        if raw is None:
            raw = _linear_fallback(fis_conds)
        else:
            raw = _apply_ceilings(mf_vals, raw)

        score = _post_fis_adjust(raw, row, h)

        # Notice period bonus: ≤30 days available → +0.05
        notice_days = row.get("notice_period_days")
        notice_bonus = notice_days is not None and notice_days <= 30
        if notice_bonus:
            score = min(1.0, score + 0.05)

        # Title bonus: candidate's current title matches a JD-aligned AI/NLP/IR title → +0.01
        cand_title = str(row.get("title") or "")
        title_bonus = bool(cand_title) and any(kw in cand_title for kw in _JD_REQ_TITLES)
        if title_bonus:
            score = min(1.0, score + 0.01)

        cls   = _fuzzy_class(score)
        rsn   = _build_fuzzy_reasoning(row, conds, score)
        if notice_bonus:
            rsn += " [notice_bonus+0.05]"
        if title_bonus:
            rsn += f" [title_bonus+0.01:{cand_title}]"

        c["l3_score"]     = round(score, 4)
        c["l3_class"]     = cls
        c["l3_reasoning"] = rsn
        row.update(
            l7_fis_score    = round(score, 4),
            fuzzy_class     = cls,
            fuzzy_penalty   = round(fis_conds["f"], 4),
            l3_h_penalty    = bool(h > 0.5),
            reasoning       = rsn,
        )
        scored.append(score)

    avg = sum(scored) / len(scored) if scored else 0.0
    logger.info(
        f"L3 fuzzy: {len(candidates)} in — {len(active)} FIS-scored "
        f"(avg={avg:.3f}), {len(candidates) - len(active)} skipped (no table_row)"
    )
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — SEMANTIC WORK RELEVANCE
# ══════════════════════════════════════════════════════════════════════════════

def _jd_responsibilities_text(jd: dict) -> str:
    for key in (
        "what_you_will_do", "what_youll_do", "what_will_you_do",
        "responsibilities", "job_responsibilities", "key_responsibilities",
        "role_responsibilities", "duties", "role_description",
    ):
        val = jd.get(key)
        if val:
            if isinstance(val, list):
                return " ".join(str(v) for v in val if v).strip()
            s = str(val).strip()
            if s:
                return s
    return str(jd.get("description") or jd.get("job_description") or "").strip()


_L4_CHAR_LIMIT = 2000


def l4_semantic_work(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 4 — Semantic Work Relevance.
    Attaches l4_work_relevance and l4_combined_score = l3_score + l4_work_relevance.
    """
    model = utils.load_sentence_transformer()

    jd_text = _jd_responsibilities_text(jd)
    if not jd_text:
        logger.warning("L4: JD has no responsibilities/description text — work_relevance=0 for all")

    if jd_text:
        jd_vec: np.ndarray = model.encode(
            jd_text[:_L4_CHAR_LIMIT],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    else:
        jd_vec = np.zeros(C.EMBED_DIM, dtype=np.float32)

    work_texts: List[str] = [
        utils.get_work_descriptions_only(c)[:_L4_CHAR_LIMIT] for c in candidates
    ]

    non_empty_idx = [i for i, t in enumerate(work_texts) if t.strip()]
    work_relevances = [0.0] * len(candidates)

    if non_empty_idx and jd_text:
        batch_texts = [work_texts[i] for i in non_empty_idx]

        mat: np.ndarray = model.encode(
            batch_texts,
            batch_size=C.L4_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        sims: np.ndarray = mat @ jd_vec

        for list_pos, cand_idx in enumerate(non_empty_idx):
            work_relevances[cand_idx] = max(0.0, float(sims[list_pos]))

    for i, c in enumerate(candidates):
        wr  = round(work_relevances[i], 4)
        l3  = float(c.get("l3_score") or 0.0)
        c["l4_work_relevance"] = wr
        c["l4_combined_score"] = round(l3 + wr, 4)

    candidates.sort(key=lambda c: c["l4_combined_score"], reverse=True)

    avg_wr  = sum(c["l4_work_relevance"] for c in candidates) / max(len(candidates), 1)
    avg_cmb = sum(c["l4_combined_score"]  for c in candidates) / max(len(candidates), 1)
    logger.info(
        f"L4 semantic: {len(candidates)} scored "
        f"(avg_relevance={avg_wr:.3f}, avg_combined={avg_cmb:.3f})"
    )
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — FLASHRANK CROSS-ENCODER RE-RANK
# ══════════════════════════════════════════════════════════════════════════════

_DONTS_STOPWORDS: frozenset = frozenset({
    "with", "only", "that", "this", "from", "have", "their", "they", "more",
    "than", "been", "will", "about", "when", "some", "what", "which", "such",
    "does", "just", "very", "also", "into", "over", "your", "those", "these",
    "candidate", "candidates", "background", "experience",
})


def _donts_penalty(c: dict, donts: List[str]) -> float:
    if not donts:
        return 1.0

    profile = (
        utils.get_work_descriptions_only(c) + " " + utils.get_skills_text(c)
    ).lower()

    matches = 0
    for dont in donts:
        words = [w for w in re.findall(r"[a-z]{4,}", dont.lower())
                 if w not in _DONTS_STOPWORDS]
        if not words:
            continue
        hit = sum(1 for w in words if w in profile)
        if hit / len(words) >= 0.50:
            matches += 1

    return max(0.20, 1.0 - 0.25 * matches)


def donts_penalty_layer(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Donts penalty layer — applied to top-200 before FlashRank.
    Re-sorts by l5_donts_score and returns top 100.
    """
    donts = [str(d) for d in (jd.get("donts") or []) if d]
    for c in candidates:
        mult  = _donts_penalty(c, donts)
        base  = float(c.get("l4_combined_score") or 0.0)
        c["l5_donts_mult"]  = round(mult, 4)
        c["l5_donts_score"] = round(base * mult, 4)

    candidates.sort(key=lambda c: c["l5_donts_score"], reverse=True)
    top100 = candidates[:100]

    n_pen = sum(1 for c in top100 if c.get("l5_donts_mult", 1.0) < 1.0)
    logger.info(
        f"Donts penalty: {len(candidates)} in → top {len(top100)} out "
        f"({n_pen} penalised, {len(donts)} dont rules)"
    )
    return top100


def l5_flashrank_rerank(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 5 — FlashRank cross-encoder re-rank.
    total_score = 0.4*l4_work_relevance + 0.3*l3_score + 0.3*flashrank_score
    """
    for c in candidates:
        c["l5_flashrank_score"] = 0.0
        c["l5_total_score"]     = round(
            0.4 * float(c.get("l4_work_relevance") or 0.0)
            + 0.3 * float(c.get("l3_score") or 0.0),
            4,
        )

    try:
        from flashrank import RerankRequest
        ranker = utils.load_flashrank()
        if ranker is None:
            raise RuntimeError("FlashRank ranker is None")
    except Exception as exc:
        logger.warning(f"L5 FlashRank unavailable ({exc}); total_score = donts_score for all")
        return candidates

    ranker  = utils.load_flashrank()
    jd_text = _jd_responsibilities_text(jd)

    if not jd_text:
        logger.warning("L5: JD has no responsibilities text — flashrank_score=0 for all")
        return candidates

    top_n    = min(C.FLASHRANK_TOP_N, len(candidates))
    top_pool = candidates[:top_n]

    passages = [
        {"id": i, "text": utils.get_work_descriptions_only(c) or ""}
        for i, c in enumerate(top_pool)
    ]
    results = ranker.rerank(RerankRequest(query=jd_text, passages=passages))

    # min-max normalize within the batch so the best candidate always gets 1.0
    # and worst gets 0.0 — sigmoid was collapsing near-zero logits to uniform 0.5
    raw_scores = {r["id"]: float(r.get("score", 0.0)) for r in results}
    if raw_scores:
        lo = min(raw_scores.values())
        hi = max(raw_scores.values())
        span = (hi - lo) or 1.0
        id_to_score: Dict[int, float] = {k: (v - lo) / span for k, v in raw_scores.items()}
    else:
        id_to_score = {}

    for i, c in enumerate(top_pool):
        fr_score = id_to_score.get(i, 0.0)
        c["l5_flashrank_score"] = round(fr_score, 4)
        c["l5_total_score"]     = round(
            0.4 * float(c.get("l4_work_relevance") or 0.0)
            + 0.3 * float(c.get("l3_score") or 0.0)
            + 0.3 * fr_score,
            4,
        )

    top_reranked = sorted(top_pool, key=lambda c: c["l5_total_score"], reverse=True)
    tail         = candidates[top_n:]

    avg_fr = sum(c["l5_flashrank_score"] for c in top_reranked) / top_n
    logger.info(
        f"L5 flashrank: cross-encoded top {top_n} of {len(candidates)} "
        f"(avg_fr={avg_fr:.3f}); total_score = 0.4*L4 + 0.3*L3 + 0.3*FlashRank"
    )
    return top_reranked + tail
