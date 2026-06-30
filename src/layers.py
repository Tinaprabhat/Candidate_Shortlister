"""
layers.py — The ranking cascade layers.

Streaming pipeline (per-candidate, continuous, concurrent — see
run_candidate_pipeline / run_streaming_cascade): every candidate flows
through L1a→L1b→L1c→L1d→L2→L3 immediately with no batch wait in between, and
candidates run concurrently against each other via a worker pool. The single
compilation/synchronisation point for the whole pre-gate pipeline is the
gather right before the 75% FIS gate.

  L1a  Hard reject (fraud / impossibilities)           → knockout
  L1b  Profile integrity (ATS pre-computed flags)      → hard reject only; soft flags → L2 cols
  L1c  Skill match (NLP + synonym)                     → score [0–1]; explicit-skill hard-reject gate
  L1d  Inferred skill match                            → l1d_score; never rejects
  L2   Table extract (31 cols)                         → feeds L3 FIS; never rejects
  L3   Sugeno fuzzy inference (conditions a–h)         → l3_score

Post-gate cascade (global, after the one compilation point):
  gate  Top 50% + random 25% by l3_score = 75% kept    → the only synchronisation barrier
  L4    Semantic work relevance (all-MiniLM-L6-v2)      → l4_combined_score (donts penalty baked in); top-100 forwarded
  L5    FlashRank cross-encoder (top-50), min-max norm  → l5_total_score = (l3_score + l4_score + flashrank) / 3
"""

import math
import random
import logging
import re
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

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

# Ordered longest-first so "pvt ltd" is stripped before "ltd"
_COMPANY_SUFFIXES = (
    "pvt ltd", "private limited", "pvt. ltd.", "pvt. ltd", "pvt ltd.",
    "corporation", "incorporated", "limited liability company",
    "corp.", "corp", "inc.", "inc", "ltd.", "ltd", "llc", "llp",
    "plc", "gmbh", "s.a.", "s.a", "nv", "bv", "ag", "kg",
)

def _strip_company_suffix(name: str) -> str:
    """Remove common legal suffixes from a company name (case-insensitive)."""
    s = name.strip().lower()
    for suffix in _COMPANY_SUFFIXES:
        if s.endswith(" " + suffix):
            s = s[: -(len(suffix) + 1)].rstrip(" ,.")
            break  # strip at most one suffix layer
    return s


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
    with utils.FRAUD_KB_LOCK:
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
    company = _strip_company_suffix(company)
    # In-code fictional list (fast, no DB round-trip)
    if company in FICTIONAL_COMPANIES:
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


"""def _check_math_consistency(c: dict, fraud_kb, current_year: int):
    
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
    
    flags = []

    # ── Pre-computed flags ─────────────────────────────────────────────────────
    if c.get("possible_honeypot") is True:
        return ["honeypot_flag"], True
    if c.get("salary_was_inverted") is True:
        return ["salary_inverted"], True

    # Salary range inversion — read from actual data path (redrob_signals)
    _sig = c.get("redrob_signals") or {}
    _sal_rng = _sig.get("expected_salary_range_inr_lpa") or {}
    try:
        _smin = float(_sal_rng["min"]) if _sal_rng.get("min") is not None else None
        _smax = float(_sal_rng["max"]) if _sal_rng.get("max") is not None else None
        if _smin is not None and _smax is not None and _smin > _smax:
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

    # Cert date sanity: LangChain cannot predate its public release (2022)
    for cert in (c.get("certifications") or []):
        if not isinstance(cert, dict):
            continue
        cname = str(cert.get("name") or cert.get("title") or "").lower()
        if "langchain" in cname:
            cert_year = _parse_year(
                cert.get("year") or cert.get("date") or cert.get("issue_date")
            )
            if cert_year is not None and cert_year < 2022:
                flags.append(f"cert_impossible:langchain_before_2022:{cert_year}")
                return flags, True

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

    return flags, False """


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


def _l1a_process_one(c: dict, fraud_kb, current_year: int) -> bool:
    """Mutate c with l1_score/l1_flags/l1_status. Return True if it survives L1a."""
    l1_score, kb_flags, status = _run_kb_verification(c, fraud_kb, current_year)
    c["l1_score"] = l1_score
    c["l1_flags"] = kb_flags
    c["l1_status"] = status
    return status != "reject"


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
        if _l1a_process_one(c, fraud_kb, current_year):
            survivors.append(c)
        else:
            rejected += 1

    logger.info(f"L1: {len(candidates)} in → {len(survivors)} pass, {rejected} hard-rejected")
    return survivors


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 1b — PROFILE INTEGRITY FLAGS
# ──────────────────────────────────────────────────────────────────────────────

def l1b_profile_integrity(candidates: List[dict]) -> List[dict]:
    """
    Layer 1b — Profile integrity: hard-reject gate only.

    Hard-reject conditions (any fires → remove from pool):
      - reverse_degree_order or all_descriptions_identical (top-level ATS flags)
      - invalid_degree_field_combination on any education entry (per-entry check)
      - redrob_signals.willing_to_relocate is False
      - redrob_signals.open_to_work_flag is False

    Soft-penalty flags are stored as boolean columns in the L2 table and unified
    into condition 'h' by L3 — they are NOT applied here.

    All surviving candidates get l1b_penalty=1.0, l1b_flags=[], l1b_status='pass'.
    """
    survivors = []
    rejected = 0

    for c in candidates:
        if _l1b_process_one(c):
            survivors.append(c)
        else:
            rejected += 1

    logger.info(
        f"L1b: {len(candidates)} in → {len(survivors)} pass ({rejected} hard-rejected)"
    )
    return survivors


def _l1b_process_one(c: dict) -> bool:
    """Mutate c with l1b_penalty/l1b_flags/l1b_status. Return True if it survives L1b."""
    flags = []
    sig = c.get("redrob_signals") or {}

    # Top-level ATS flags
    if c.get("reverse_degree_order") is True:
        flags.append("reverse_degree_order")
    if c.get("duplicate_job_descriptions") is True:
        flags.append("duplicate_job_descriptions")

    # Per-entry degree+field combination check
    if any(
        isinstance(e, dict) and e.get("invalid_degree_field_combination") is True
        for e in (c.get("education") or [])
    ):
        flags.append("invalid_degree_field_combination")

    # Opt-out signals — explicit False only; None/missing = pass through
    if sig.get("willing_to_relocate") is False:
        flags.append("not_willing_to_relocate")
    if sig.get("open_to_work_flag") is False:
        flags.append("not_open_to_work")

    if flags:
        c["l1b_penalty"] = 0.0
        c["l1b_flags"]   = flags
        c["l1b_status"]  = "reject"
        return False

    c["l1b_penalty"] = 1.0
    c["l1b_flags"]   = []
    c["l1b_status"]  = "pass"
    return True


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
        # Use negative look-behind/ahead for alphanumeric and hyphen to avoid substrings
        pattern = r"(?<![a-z0-9\-])" + re.escape(form) + r"(?![a-z0-9\-])"
        if re.search(pattern, text):
            return True
    return False


_PROFICIENCY_MAP: Dict[str, float] = {
    "beginner": 0.1, "intermediate": 0.2, "advanced": 0.3, "expert": 0.4
}


def _proficiency_score_for_skills(
    c: dict, jd_skills: List[str], assessment_scores: dict
) -> float:
    """
    proficiency_score = 0.3 * skills_assessment_score + 0.7 * skill_proficiency

    skills_assessment_score: avg of assessment_scores entries matching any jd_skill / 100
                             (0 if no matches or field absent)
    skill_proficiency: avg proficiency of candidate skills[] entries matching any jd_skill
                       beginner=0.1, intermediate=0.2, advanced=0.3, expert=0.4
                       (0 if no matches or proficiency field absent)
    """
    if not jd_skills:
        return 0.0

    jd_expanded: set = set()
    for skill in jd_skills:
        jd_expanded.update(expand_skill(skill))

    # skills_assessment_score filtered to JD skills
    relevant_assessments = []
    for skill_name, score in (assessment_scores or {}).items():
        if set(expand_skill(skill_name)) & jd_expanded:
            try:
                relevant_assessments.append(min(float(score), 100.0) / 100.0)
            except (ValueError, TypeError):
                pass
    skills_assessment_score = (
        sum(relevant_assessments) / len(relevant_assessments)
        if relevant_assessments else 0.0
    )

    # skill_proficiency from candidate skills[] section
    skills_raw = c.get("skills") or []
    proficiency_values = []
    seen: set = set()
    for s in skills_raw:
        if not isinstance(s, dict):
            continue
        name = str(s.get("name") or "").lower().strip()
        prof = str(s.get("proficiency") or "").lower().strip()
        if not name or name in seen:
            continue
        if set(expand_skill(name)) & jd_expanded and prof in _PROFICIENCY_MAP:
            proficiency_values.append(_PROFICIENCY_MAP[prof])
            seen.add(name)
    skill_proficiency = (
        sum(proficiency_values) / len(proficiency_values)
        if proficiency_values else 0.0
    )

    return round(0.3 * skills_assessment_score + 0.7 * skill_proficiency, 4)


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


def build_pipeline_ctx(jd: dict) -> dict:
    """
    Precompute JD-derived static lists/counts ONCE per run. The returned dict
    is read-only and shared (never mutated) across the concurrent per-candidate
    pipeline, so it is safe to hand to every worker thread.
    """
    explicit_req = jd.get("explicit_required", [])
    explicit_bon = jd.get("explicit_bonus", [])
    inferred_req = jd.get("inferred_required", [])
    inferred_bon = jd.get("inferred_bonus", [])
    inferred     = inferred_req + inferred_bon
    return {
        "explicit_req":   explicit_req,
        "explicit_bon":   explicit_bon,
        "n_explicit_req": len(explicit_req),
        "n_explicit_bon": len(explicit_bon),
        "n_explicit":     len(explicit_req),  # hard-reject gate uses explicit required only
        "inferred_req":   inferred_req,
        "inferred_bon":   inferred_bon,
        "inferred":       inferred,
        "n_inferred":     len(inferred),
    }


def _l1c_process_one(c: dict, ctx: dict) -> bool:
    """Mutate c with l1c_* fields. Return True if it survives L1c's gates."""
    explicit_req   = ctx["explicit_req"]
    explicit_bon   = ctx["explicit_bon"]
    n_explicit_req = ctx["n_explicit_req"]
    n_explicit_bon = ctx["n_explicit_bon"]
    n_explicit     = ctx["n_explicit"]
    inferred_req   = ctx["inferred_req"]
    inferred_bon   = ctx["inferred_bon"]

    text = _candidate_search_text(c)

    # Explicit required — score numerator and hard-reject gate
    expl_req_match = {s: _skill_in_text(s, text) for s in explicit_req}
    matched_req    = [s for s, m in expl_req_match.items() if m]
    missing_req    = [s for s, m in expl_req_match.items() if not m]

    # Explicit bonus — contributes to l1c_score (0.25 weight)
    matched_bon    = [s for s in explicit_bon if _skill_in_text(s, text)]

    # Inferred skills — stored here, matched and scored in L1d
    matched_inferred = (
        [s for s in inferred_req if _skill_in_text(s, text)]
        + [s for s in inferred_bon if _skill_in_text(s, text)]
    )

    req_ratio = len(matched_req) / n_explicit_req if n_explicit_req else 1.0
    bon_ratio = len(matched_bon) / n_explicit_bon if n_explicit_bon else 0.0

    if n_explicit_req > 0 and n_explicit_bon > 0:
        score = 0.75 * req_ratio + 0.25 * bon_ratio
    elif n_explicit_req > 0:
        score = req_ratio
    elif n_explicit_bon > 0:
        score = bon_ratio
    else:
        score = 1.0

    c["l1c_score"]            = round(score, 4)
    c["l1c_matched_required"] = matched_req    # explicit required only
    c["l1c_missing_required"] = missing_req
    c["l1c_matched_bonus"]    = matched_bon    # explicit bonus only
    c["l1c_matched_explicit"] = matched_req    # alias used by hard-reject gate
    c["l1c_matched_inferred"] = matched_inferred  # forwarded to L1d

    _signals = c.get("redrob_signals") or {}
    c["l1c_explicit_proficiency_score"] = _proficiency_score_for_skills(
        c, explicit_req + explicit_bon, _signals.get("skill_assessment_scores") or {}
    )

    jd_req_score, jd_req_results = compute_skill_match(c, JD_REQUIREMENTS)
    c["jd_req_score"]   = round(jd_req_score, 4)
    c["jd_req_results"] = jd_req_results

    # Hard reject: zero explicit_required matches when JD has explicit skills
    if n_explicit > 0 and not matched_req:
        return False

    # Optional score gate
    if C.L1C_MIN_SKILL_MATCH > 0 and c["l1c_score"] < C.L1C_MIN_SKILL_MATCH:
        return False

    return True


def l1c_skill_match(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 1c — NLP string skill match using the synonym dictionary.

    For each candidate, searches the candidate's skills list + work descriptions
    + profile summary for every JD-required and JD-bonus skill, expanding each
    through the synonym dictionary before matching.

    Score:
        If both required and bonus skills exist:
            l1c_score = 0.75 * (req_matched / req_total) + 0.25 * (bon_matched / bon_total)
        If only required:  l1c_score = req_matched / req_total
        If only bonus:     l1c_score = bon_matched / bon_total
        If no skills in JD: l1c_score = 1.0

    Hard-reject gate (applied before the optional score gate):
        If the JD has explicit_required skills and the candidate matches NONE of
        them → hard reject (removed from pool, not just penalised).

    Optional score gate (C.L1C_MIN_SKILL_MATCH > 0): candidates scoring below
    the threshold are additionally filtered out. Defaults to 0.0 (off).

    Attaches: l1c_score, l1c_matched_required, l1c_missing_required,
              l1c_matched_bonus, l1c_matched_explicit.
    """
    ctx = build_pipeline_ctx(jd)
    before = len(candidates)
    dropped_explicit = 0
    dropped_score = 0
    survivors = []

    for c in candidates:
        if _l1c_process_one(c, ctx):
            survivors.append(c)
        elif ctx["n_explicit"] > 0 and not c.get("l1c_matched_explicit"):
            dropped_explicit += 1
        else:
            dropped_score += 1

    if dropped_explicit:
        logger.info(
            f"L1c hard-reject: {dropped_explicit} candidates matched 0/{ctx['n_explicit']} "
            f"explicit required skills → removed"
        )
    if dropped_score:
        logger.info(
            f"L1c score gate: dropped {dropped_score} below min_score={C.L1C_MIN_SKILL_MATCH}"
        )

    avg = sum(c.get("l1c_score", 0.0) for c in survivors) / max(len(survivors), 1)
    logger.info(
        f"L1c: {before} in → {len(survivors)} pass "
        f"(avg_score={avg:.3f}, explicit_req={ctx['n_explicit_req']}, explicit_bon={ctx['n_explicit_bon']})"
    )
    return survivors


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
      l1d_matched_inferred  list[str]   — inferred skills found
      l1d_unmatched_inferred list[str]  — inferred skills not found
      l1d_inferred_ratio    float[0,1]  — matched / total
      l1d_leftover_count    int         — count of unmatched inferred skills
      l1d_score             float[0,1]  — net score forwarded to L2

    l1d_score is forwarded to the L2 table as l1d_inferred_score.
    """
    ctx = build_pipeline_ctx(jd)
    for c in candidates:
        _l1d_process_one(c, ctx)

    avg = sum(c.get("l1d_inferred_ratio", 1.0) for c in candidates) / max(len(candidates), 1)
    logger.info(
        f"L1d: {len(candidates)} candidates — {ctx['n_inferred']} inferred skills "
        f"(avg_inferred_match={avg:.3f})"
    )
    return candidates


def _l1d_process_one(c: dict, ctx: dict) -> None:
    """Mutate c with l1d_* fields. Never rejects (soft layer)."""
    inferred   = ctx["inferred"]
    n_inferred = ctx["n_inferred"]

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

    _signals = c.get("redrob_signals") or {}
    c["l1d_inferred_proficiency_score"] = _proficiency_score_for_skills(
        c, inferred, _signals.get("skill_assessment_scores") or {}
    )


# ──────────────────────────────────────────────────────────────────────────────
# L3 GLOBAL GATE — 75% (top 50% + random 25%)
# Applied after L3 fuzzy scoring, before expensive L4 semantic encoding.
# ──────────────────────────────────────────────────────────────────────────────
def l3_gate(candidates: List[dict], seed: int = 42) -> List[dict]:
    """
    Keep the top 50% by l3_score plus a random 25% from the remainder = 75%.
    The random rescue prevents a strong folder monopolising slots and preserves
    diversity (edge-case candidates with low fuzzy scores but other strengths).
    Uses math.ceil so a pool of 1 always yields at least 1 candidate."""
    
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

# ── Keyword vocabularies for cols 11–13 ──────────────────────────────────────
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
    "mean average precision": 0.25,
    "a/b test": 0.20, "ab test": 0.20, "ab testing": 0.20,
    "recall@k": 0.20, "recall@": 0.18,
    "benchmark": 0.15,
    "precision@": 0.15, "precision": 0.12,
    "evaluation": 0.10, "metrics": 0.10,
    "beir": 0.20, "trec": 0.20,
    "offline evaluation": 0.15, "online evaluation": 0.15,
    "f1 score": 0.12,
}

# ── Tools vocabulary for col 18 — weighted by stack relevance ────────────────
# CV-specific tools (opencv, torchvision, detectron2, yolo, albumentations …)
# are deliberately excluded.  Categories and weights:
#   Embedding tools          → 1.0–0.90  (highest: core IR capability)
#   Vector DB / ANN indexes  → 0.85–0.70  (high)
#   Ranking / reranking      → 0.90–0.80  (high)
#   NLP & LLM models         → 0.70–0.50  (medium)
#   Deployment / MLOps       → 0.50–0.40  (medium)
#   AI orchestration         → 0.30–0.25  (low)
#   Cloud ML platforms       → 0.20       (lowest)

# Embedding tools — highest weight
_EMBEDDING_TOOLS: Dict[str, float] = {
    "sentence-transformers": 1.00,
    "fastembed":             1.00,
    "huggingface":           0.90,   # HF Hub / Transformers ecosystem
    "cohere":                0.90,   # Cohere embed + rerank API
}

# Vector database / ANN index tools — high weight
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

# Ranking / reranking tools — high weight
_RANKING_TOOLS: Dict[str, float] = {
    "flashrank":     0.90,
    "colbert":       0.90,
    "cross-encoder": 0.85,
    "bm25":          0.80,
    "reranker":      0.80,
}

# NLP and LLM model tools — medium weight
_NLP_MODEL_TOOLS: Dict[str, float] = {
    "bert":        0.70,
    "llama":       0.70,
    "spacy":       0.65,
    "mistral":     0.65,
    "openai":      0.65,   # GPT / embeddings API
    "gensim":      0.60,
    "pytorch":     0.60,
    "anthropic":   0.60,
    "nltk":        0.55,
    "tensorflow":  0.55,
    "keras":       0.50,
    "onnx":        0.50,
    "tensorrt":    0.50,
}

# Deployment / serving / MLOps tools — medium weight
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

# AI orchestration frameworks — low weight
_ORCHESTRATION_TOOLS: Dict[str, float] = {
    "langchain":  0.30,
    "llamaindex": 0.30,
    "haystack":   0.30,
    "crewai":     0.25,
}

# Cloud ML platforms — lowest weight
_CLOUD_TOOLS: Dict[str, float] = {
    "sagemaker":  0.20,
    "bigquery":   0.20,
    "databricks": 0.20,
    "snowflake":  0.20,
    "redshift":   0.20,
}

# Merged lookup: tool_name → weight  (used for tools_score computation)
_TOOLS_WEIGHTED: Dict[str, float] = {
    **_EMBEDDING_TOOLS,
    **_VECTOR_DB_TOOLS,
    **_RANKING_TOOLS,
    **_NLP_MODEL_TOOLS,
    **_DEPLOYMENT_TOOLS,
    **_ORCHESTRATION_TOOLS,
    **_CLOUD_TOOLS,
}

# IR domain terms for condition d scoring (0/0.5/1.0 based on hit count)
_IR_DOMAIN_TERMS: frozenset = frozenset({
    "bm25", "faiss", "rerank", "retrieval", "ranking",
    "vector search", "embedding",
})

# ── Consulting firms / industries for col 19 ─────────────────────────────────
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

# ── Research publication keywords for col 17 ─────────────────────────────────
_RESEARCH_KW: frozenset = frozenset({
    "paper", "published", "arxiv", "proceedings", "neurips", "nips",
    "icml", "iclr", "cvpr", "acl", "emnlp", "journal", "preprint",
    "citation", "dissertation", "thesis", "conference paper", "research paper",
    "peer reviewed", "peer-reviewed",
})

# ── JD-relevant title keywords (NLP / IR / Applied AI — NOT CV / Speech) ─────
# Titles matching these indicate the candidate's role is closely aligned with
# the JD target (Senior AI Engineer, IR/NLP stack).  Used in L3 for +0.02 bonus.
# CV-specific titles (computer vision, vision scientist, etc.) are intentionally
# excluded — they signal a different domain.
_JD_REQ_TITLES: frozenset = frozenset({
    # Core AI / ML engineering
    "ai engineer", "senior ai engineer", "staff ai engineer", "principal ai engineer",
    "ml engineer", "machine learning engineer", "senior ml engineer",
    "principal ml engineer", "staff ml engineer",
    # Applied AI / Applied Science
    "applied scientist", "applied ai engineer", "applied ai",
    "applied machine learning engineer", "applied research scientist",
    # NLP / Information Retrieval / Search
    "nlp engineer", "nlp scientist", "natural language processing engineer",
    "search engineer", "senior search engineer", "search scientist",
    "ranking engineer", "relevance engineer", "information retrieval engineer",
    # Research Engineering (AI/ML-focused, not domain-locked; "ai research engineer"
    # excluded — too broad, fires on CV/speech candidates)
    "research engineer", "senior research engineer",
    "ml research engineer",
    # AI / ML leadership
    "ai tech lead", "ml tech lead", "ai lead", "ml lead",
    "ai architect", "ml architect",
})


# ── Private helpers ───────────────────────────────────────────────────────────

def _safe_float(val, default=None) -> Optional[float]:
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default



def _kw_accumulate(text: str, kw_dict: Dict[str, float]) -> float:
    """Weighted keyword accumulation over lowercased text, capped at 1.0."""
    score = 0.0
    for kw, weight in kw_dict.items():
        if kw in text:
            score += weight
    return min(score, 1.0)


def _work_text_lower(c: dict) -> str:
    """All career_history description strings, concatenated and lowercased."""
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
    """
    Direct weighted composite from redrob_signals fields (all expected [0,1]).
    Weights sum to 1.0.
    -1 sentinel → subtract that field's weight (bad signal).
    Missing field → contributes 0 (neutral).
    """
    _FIELDS = [
        ("applications_submitted_30d",  0.01),
        ("avg_response_time_hours",     0.01),
        ("connection_count",            0.05),
        ("endorsements_received",       0.03),
        ("github_activity_score",       0.30),
        ("interview_completion_rate",   0.05),
        ("offer_acceptance_rate",       0.05),
        ("profile_completeness_score",  0.10),
        ("profile_views_received_30d",  0.05),
        ("recruiter_response_rate",     0.20),
        ("saved_by_recruiters_30d",     0.10),
        ("search_appearance_30d",       0.05),
    ]
    score = 0.0
    for field, weight in _FIELDS:
        v = signals.get(field)
        if v is None:
            continue
        fv = float(v)
        score += (-1.0 if fv == -1.0 else fv) * weight
    return round(score, 4)


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
    summary    = str(profile.get("summary") or "").lower()

    # 01 — candidate_id
    candidate_id = str(c.get("candidate_id") or c.get("id") or "")

    # 02 — total_exp
    total_exp = utils.get_total_experience_years(c)

    # 03 — location
    loc     = str(profile.get("location") or "").strip()
    country = str(profile.get("country") or "").strip()
    location = f"{loc}, {country}" if loc and country else loc or country or ""

    # 04 — max_salary + internal salary-inversion flag (feeds col 28)
    sal_range   = signals.get("expected_salary_range_inr_lpa") or {}
    max_salary  = _safe_float(sal_range.get("max"))
    min_salary  = _safe_float(sal_range.get("min"))
    _sal_inverted = (
        min_salary is not None and max_salary is not None and min_salary > max_salary
    )

    # 05 — skill_assessment_score
    assessment = signals.get("skill_assessment_scores")
    if isinstance(assessment, dict) and assessment:
        skill_assessment_score = round(sum(assessment.values()) / len(assessment) / 100.0, 4)
    else:
        skill_assessment_score = 0.0

    # 06 — skill_match_score  (carried from L1c)
    matched_req = c.get("l1c_matched_required") or []
    skill_match_score = round(len(matched_req) / n_req, 4) if n_req > 0 else 0.0

    # 07 — inferred_skill_match_score  (NLP + synonym match; raw count)
    cand_text = _candidate_search_text(c)  # lowercased skills + descs + summary
    inferred_skill_match_score = sum(
        1 for skill in jd_inferred if _skill_in_text(skill, cand_text)
    )

    # 08 — redrob_cumulative
    redrob_cumulative = _redrob_cumulative(signals)

    # 09 — industry
    industry = str(profile.get("current_industry") or profile.get("industry") or "")

    # 10 — is_phd
    is_phd = any(
        any(kw in str(e.get("degree", "")).lower() for kw in ("phd", "ph.d", "ph. d", "doctor", "d.phil"))
        for e in edu_list if isinstance(e, dict)
    )

    # 11–13 — keyword scores over all work descriptions
    wtext = _work_text_lower(c)
    production_score        = round(_kw_accumulate(wtext, _PRODUCTION_KW), 4)
    architecture_score      = round(_kw_accumulate(wtext, _ARCHITECTURE_KW), 4)
    testing_evaluation_score = round(_kw_accumulate(wtext, _TESTING_EVAL_KW), 4)

    # 14 — no_certifications
    no_certifications = len(certs)

    # 15 — no_languages
    no_languages = len(langs)

    # 16 — open_source_score
    github_raw = _safe_float(signals.get("github_activity_score"))
    open_source_score = (
        0.0 if (github_raw is None or github_raw == -1.0)
        else round(github_raw / 100.0, 4)
    )

    # 17 — research_published  (True | None, never False)
    research_text = wtext + " " + " ".join(
        str(cert.get("name") or "").lower() for cert in certs if isinstance(cert, dict)
    )
    research_published = True if any(kw in research_text for kw in _RESEARCH_KW) else None

    # 18 — tools_score  (weighted sum; normalized by 3.0 in L3 _compute_conditions)
    skills_text = " ".join(
        str(s.get("name") if isinstance(s, dict) else s).lower() for s in skills_raw
    )
    full_text = skills_text + " " + wtext
    tools_score = sum(w for t, w in _TOOLS_WEIGHTED.items() if t in full_text)

    # ir_domain_score — count of retrieval/IR-specific terms hit in full_text
    ir_domain_score = sum(1 for t in _IR_DOMAIN_TERMS if t in full_text)

    # orchestration_score — weighted sum of orchestration-only hits (sub-score of tools_score)
    orchestration_score = sum(w for t, w in _ORCHESTRATION_TOOLS.items() if t in full_text)

    # 19 — consulting_only
    consulting_only = bool(work_hist) and all(
        (
            str(exp.get("industry") or "").strip().lower() in _CONSULTING_INDUSTRIES
            or str(exp.get("company") or "").strip().lower() in _CONSULTING_FIRMS
        )
        for exp in work_hist if isinstance(exp, dict)
    )

    # 20–21 — last_career_tenure / last_career_company
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

    # 22 — no_offer_history
    offer_raw       = _safe_float(signals.get("offer_acceptance_rate"))
    no_offer_history = offer_raw == -1.0

    # 23–24 — education timeline vs career start
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
    # Merge with upstream ATS flag (field is education_career_gap_flag in the data)
    edu_career_gap_flag = edu_career_gap_flag or bool(c.get("education_career_gap_flag"))

    # 26 — low_engagement_flag
    apps_30d      = _safe_float(signals.get("applications_submitted_30d"))
    response_rate = _safe_float(signals.get("recruiter_response_rate"))
    low_engagement_flag = (
        apps_30d is not None and apps_30d < 1
        and response_rate is not None and response_rate < 0.2
    )

    # 28 — fabrication_bandwidth  [0, 1]
    """fab = 0.0
    if _sal_inverted:
        fab += 0.30
    cur_title  = str(profile.get("current_title") or profile.get("title") or "").strip().lower()
    job_title  = str((current_job or {}).get("title") or "").strip().lower()
    if cur_title and job_title and cur_title != job_title:
        fab += 0.20
    if "marketing manager" in summary:
        fab += 0.10"""
    fabrication_bandwidth = float(c.get("fabrication_bandwidth") or 0.0)

    # 29 — possible_fabrication
    possible_fabrication = bool(c.get("possible_fabrication"))

    # notice_period_days — read from redrob_signals (actual data path)
    notice_period_days = (
        signals.get("notice_period_days")
        or c.get("notice_period_days")
        or c.get("notice_period")
    )
    # l1d_inferred_score — forwarded from L1d (1.0 if L1d hasn't run yet)
    l1d_inferred_score = round(float(c.get("l1d_score", 1.0)), 4)
    # l1c_score — forwarded from L1c
    l1c_fwd_score = round(float(c.get("l1c_score", 0.0)), 4)
    # proficiency scores — computed at L1c / L1d respectively
    explicit_proficiency_score = round(float(c.get("l1c_explicit_proficiency_score", 0.0)), 4)
    inferred_proficiency_score = round(float(c.get("l1d_inferred_proficiency_score", 0.0)), 4)

    # title — current job title, lowercased for L3 matching against _JD_REQ_TITLES
    title = (
        str(profile.get("current_title") or profile.get("title") or profile.get("headline") or "")
        .strip().lower()
    )
    if not title and current_job:
        title = str(current_job.get("title") or "").strip().lower()

    soft_penalty_score = round(
        (0.03 if low_engagement_flag else 0.0)
        + (0.04 if edu_career_gap_flag else 0.0)
        + (0.01 if education_overlap else 0.0)
        + (0.02 if bool(c.get("skill_career_domain_mismatch")) else 0.0),
        4,
    )

    return {
        "candidate_id":               candidate_id,              # 01
        "total_exp":                  total_exp,                 # 02
        "location":                   location,                  # 03
        "max_salary":                 max_salary,                # 04
        "skill_assessment_score":     skill_assessment_score,    # 05
        "skill_match_score":          skill_match_score,         # 06
        "inferred_skill_match_score": inferred_skill_match_score, # 07
        "redrob_cumulative":          redrob_cumulative,         # 08
        "industry":                   industry,                  # 09
        "is_phd":                     is_phd,                    # 10
        "production_score":           production_score,          # 11
        "architecture_score":         architecture_score,        # 12
        "testing_evaluation_score":   testing_evaluation_score,  # 13
        "no_certifications":          no_certifications,         # 14
        "no_languages":               no_languages,              # 15
        "open_source_score":          open_source_score,         # 16
        "research_published":         research_published,        # 17
        "tools_score":                tools_score,               # 18
        "ir_domain_score":            ir_domain_score,           # IR term hit count (feeds condition d)
        "orchestration_score":        orchestration_score,       # orchestration sub-score (feeds condition e)
        "consulting_only":            consulting_only,           # 19
        "last_career_tenure":         last_career_tenure,        # 20
        "last_career_company":        last_career_company,       # 21
        "no_offer_history":           no_offer_history,          # 22
        "education_overlap":          education_overlap,         # 23
        "edu_career_gap_flag":        edu_career_gap_flag,       # 24 (local OR ATS)
        "low_engagement_flag":        low_engagement_flag,       # 26
        "fabrication_bandwidth":      fabrication_bandwidth,     # 27
        "possible_fabrication":       possible_fabrication,      # 28
        # Soft-penalty flags (read from ATS/upstream; feed L3 condition h)
        "skill_career_domain_mismatch": bool(c.get("skill_career_domain_mismatch")), # 29
        "second_undergrad_after_first": bool(c.get("second_undergrad_after_first")), # 30
        "soft_penalty_score":           soft_penalty_score,         # weighted sum [0, 0.10]; feeds L3 h
        # Layer scores forwarded + notice period + title
        "l1c_score":                 l1c_fwd_score,               # explicit skill match score from L1c
        "l1d_inferred_score":        l1d_inferred_score,          # inferred skill match score from L1d
        "explicit_proficiency_score": explicit_proficiency_score, # 0.3*assessment + 0.7*proficiency for explicit skills
        "inferred_proficiency_score": inferred_proficiency_score, # 0.3*assessment + 0.7*proficiency for inferred skills
        "notice_period_days": notice_period_days,  # from redrob_signals; None if absent
        "title":              title,               # current job title (lowercased)
    }

def l2_table_extract(
    candidates: List[dict],
    jd: dict,
) -> List[dict]:
    """
    Layer 2 — Table Extract:
    Builds a 31-column `table_row` dict on every candidate and attaches it as
    c['table_row'].  No candidates are filtered; the full list passes through.

    Cols 29-31 are the L1b soft-penalty flags read from the upstream ATS dict;
    they are used by L3 to compute condition 'h' (soft-penalty union).
    """

    ctx = build_pipeline_ctx(jd)
    for c in candidates:
        _l2_process_one(c, ctx)

    logger.info(f"L2 table extract: {len(candidates)} rows built (31 cols each)")
    return candidates


def _l2_process_one(c: dict, ctx: dict) -> None:
    """Mutate c with table_row (31-col L2 extract). Never rejects."""
    # denominator for skill_match_score = explicit required only (not bonus)
    c["table_row"] = _build_table_row(c, ctx["inferred"], ctx["n_explicit_req"])


# ──────────────────────────────────────────────────────────────────────────────
# LAYER 3 — SUGENO FUZZY INFERENCE SYSTEM  (reads table_row from L2)
# ──────────────────────────────────────────────────────────────────────────────


# ── Condition computation (Step 3) ───────────────────────────────────────────

def _compute_conditions(row: dict) -> Dict[str, float]:
    """
    Derive 8 crisp conditions a–h from a L2 table_row. All outputs in [0, 1].

    a — experience in JD sweet spot
    b — explicit skill signal: 0.8*l1c_score + 0.2*explicit_proficiency_score
    c — inferred skill signal: 0.8*l1d_inferred_score + 0.2*inferred_proficiency_score
    d — IR domain signal (0.0/0.5/1.0 based on retrieval-specific term hits in work+skills)
    e — absence of disqualifying traits (phd-only, consulting, stagnant, pure-researcher); −0.25 if framework-heavy with no production
    f — redrob cumulative platform signal (redrob_cumulative from L2)
    g — technical breadth (production + arch + testing + tools + open-source/research)
    h — soft-penalty union (1.0 = any flag fires; ideal candidate has h=0.0)
    """
    exp = float(row.get("total_exp") or 0.0)
    a = 1.0 if 5.0 <= exp <= 9.0 else (0.5 if 3.0 <= exp <= 12.0 else 0.0)

    b = (0.8 * float(row.get("l1c_score") or 0.0)
         + 0.2 * float(row.get("explicit_proficiency_score") or 0.0))

    c = (0.8 * float(row.get("l1d_inferred_score") or 0.0)
         + 0.2 * float(row.get("inferred_proficiency_score") or 0.0))

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

    f = max(0.0, min(1.0, float(row.get("redrob_cumulative") or 0.0)))

    tools_norm      = min(float(row.get("tools_score") or 0.0) / 3.0, 1.0)
    open_or_research = max(
        float(row.get("open_source_score") or 0.0),
        1.0 if row.get("research_published") is True else 0.0,
    )
    g = (
        float(row.get("production_score") or 0.0)*0.3
        + float(row.get("architecture_score") or 0.0)*0.3
        + float(row.get("testing_evaluation_score") or 0.0)*0.3
        + tools_norm*0.05
        + open_or_research*0.05
    )

    h = (0.06 * float(row.get("fabrication_bandwidth") or 0.0)
         + 0.04 * float(row.get("soft_penalty_score") or 0.0))

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "g": g, "h": h}



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
    if conds.get("h", 0.0) > 0.0:                        flags.append("soft_penalty")
    if 5.0 <= float(row.get("total_exp") or 0) <= 9.0:  flags.append("ideal_exp")
    if row.get("research_published"):                     flags.append("research")
    tag = f"[{', '.join(flags)}]" if flags else "clean"
    a, b, c, d, e, f, g, h = (conds.get(k, 0.0) for k in "abcdefgh")
    return (
        f"score={score:.3f} "
        f"a={a:.2f} b={b:.2f} c={c:.2f} d={d:.2f} e={e:.2f} f={f:.2f} g={g:.2f} h={h:.3f} "
        f"{tag}"
    )


# ── Public entry point ────────────────────────────────────────────────────────

def l3_fuzzy_score(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 3 — Sugeno Fuzzy Inference System.
    Input:  candidates that have already passed through L2 (table_row present).
    Output: same list with l3_score, l3_class, l3_reasoning attached to each c.
            Also writes l7_fis_score, fuzzy_class, fuzzy_penalty, l3_h_penalty, reasoning
            into each candidate's table_row.

    No pre-FIS knockouts — all candidates with a valid table_row are scored.
    Conditions a–g feed the 32-rule Sugeno table; condition h (weighted soft-penalty
    score [0, 0.10]) is subtracted directly from the FIS output in _post_fis_adjust.
    No candidates are removed; every candidate exits with l3_score set.
    """
    # All candidates enter FIS — only L1 hard-rejects are knockouts
    active = 0
    scored: List[float] = []
    for c in candidates:
        score = _l3_process_one(c)
        if score is not None:
            active += 1
            scored.append(score)

    avg = sum(scored) / len(scored) if scored else 0.0
    logger.info(
        f"L3 fuzzy: {len(candidates)} in — {active} FIS-scored "
        f"(avg={avg:.3f}), {len(candidates) - active} skipped (no table_row)"
    )
    return candidates


def _l3_process_one(c: dict) -> Optional[float]:
    """
    Score a single candidate's table_row via the Sugeno FIS. Mutates c with
    l3_score/l3_class/l3_reasoning (and table_row with FIS columns). Never
    rejects (FIS scores everyone; only L1 removes candidates from the pool).
    Returns the l3_score, or None if there was no table_row to score (L2 not run).
    """
    row = c.get("table_row")
    if not isinstance(row, dict):
        logger.warning(
            f"L3 fuzzy: no table_row on {c.get('candidate_id', '?')} — assign 0.0"
        )
        c["l3_score"] = 0.0
        c["l3_class"] = "no_table_row"
        c["l3_reasoning"] = "missing_l2_output"
        return None

    conds = _compute_conditions(row)
    h     = conds["h"]

    score = (
        0.40 * conds["g"]
        + 0.20 * conds["b"]
        - 0.05 * conds["e"]
        + 0.10 * conds["c"]
        + 0.05 * conds["a"]
        + 0.05 * conds["d"]
        + 0.10 * conds["f"]
        - 0.05 * h
    )
    score = max(0.0, min(1.0, score))

    # Notice period bonus: ≤ 30 days available → +0.05 (immediate or short notice)
    notice_days = row.get("notice_period_days")
    notice_bonus = notice_days is not None and notice_days <= 30
    if notice_bonus:
        score = min(1.0, score + 0.05)

    # Title bonus: candidate's current title matches a JD-aligned AI/NLP/IR title → +0.01
    cand_title = str(row.get("title") or "")
    title_bonus = bool(cand_title) and any(kw in cand_title for kw in _JD_REQ_TITLES)
    if title_bonus:
        score = min(1.0, score + 0.01)

    cls = _fuzzy_class(score)
    rsn = _build_fuzzy_reasoning(row, conds, score)
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
        fuzzy_penalty   = round(conds["f"], 4),
        l3_h_penalty    = round(h, 4),
        reasoning       = rsn,
    )
    return score


# ──────────────────────────────────────────────────────────────────────────────
# STREAMING PIPELINE — L1a→L1b→L1c→L1d→L2→L3, continuous per candidate
#
# Each candidate flows through every stage immediately, with no batch wait in
# between (unlike the standalone l1_hard_reject/l1b_.../l3_fuzzy_score
# wrappers above, which still exist for direct/unit-test use and iterate the
# whole list per stage). Candidates run concurrently against each other via a
# worker pool, so e.g. candidate 2 can already be in L1a while candidate 1 is
# in L1c. The ONLY synchronisation point is the gather at the end of
# run_streaming_cascade — immediately before the 75% FIS gate (l3_gate).
# ──────────────────────────────────────────────────────────────────────────────

def run_candidate_pipeline(c: dict, fraud_kb, ctx: dict, current_year: int) -> Optional[dict]:
    """
    Stream ONE candidate continuously through L1a→L1b→L1c→L1d→L2→L3.
    Returns the scored candidate dict, or None if hard-rejected at any stage.
    Thread-safe: only mutates `c` itself; `ctx` is read-only; fraud-KB access
    is internally serialised via utils.FRAUD_KB_LOCK.
    """
    if not _l1a_process_one(c, fraud_kb, current_year):
        return None
    if not _l1b_process_one(c):
        return None
    if not _l1c_process_one(c, ctx):
        return None
    _l1d_process_one(c, ctx)
    _l2_process_one(c, ctx)
    _l3_process_one(c)
    return c


def run_streaming_cascade(
    candidates: List[dict], jd: dict, fraud_kb, max_workers: Optional[int] = None
) -> List[dict]:
    """
    Run every candidate through L1a→L1b→L1c→L1d→L2→L3 as a continuous,
    concurrent, per-candidate pipeline (no batch wait between stages).

    This is the single "compilation" point before the FIS gate: candidates
    are dispatched to a worker pool, each one streams through every stage on
    its own, and results are gathered here as they complete — right before
    the caller applies the 75% l3_gate shortlist.
    """
    if not candidates:
        return []

    ctx          = build_pipeline_ctx(jd)
    current_year = datetime.now().year
    workers      = max_workers or min(C.PIPELINE_MAX_WORKERS, max(4, len(candidates)))

    survivors: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_candidate_pipeline, c, fraud_kb, ctx, current_year)
            for c in candidates
        ]
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                survivors.append(result)

    logger.info(
        f"Streaming cascade (L1a→L3): {len(candidates)} in → {len(survivors)} "
        f"survived to FIS gate ({workers} workers)"
    )
    return survivors


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — SEMANTIC WORK RELEVANCE
# ══════════════════════════════════════════════════════════════════════════════

# CV / Speech domain skills and title keywords — candidates whose profile is
# dominated by these are hard-penalized by the donts layer regardless of
# embedding similarity, which is too noisy at 384-dim to resolve domains.
_CV_SPEECH_SKILLS: frozenset = frozenset({
    "asr", "tts", "speech recognition", "automatic speech recognition",
    "text to speech", "text-to-speech", "computer vision", "opencv",
    "yolo", "object detection", "image classification", "image segmentation",
    "detectron", "torchvision", "face recognition", "face detection",
    "optical character recognition", "ocr",
})
_CV_SPEECH_TITLE_KW: frozenset = frozenset({
    "computer vision", "cv engineer", "vision scientist", "vision engineer",
    "speech recognition", "asr engineer", "tts engineer", "speech engineer",
    "image processing", "visual recognition",
})
_CATEGORICAL_DONTS_PENALTY = 0.60  # full _L4_DONTS_WEIGHT applied to CV/speech candidates


def _is_cv_speech_candidate(c: dict) -> bool:
    """Return True when ≥2 CV/speech skills OR a CV/speech title is present."""
    profile = c.get("profile") or {}
    title   = str(profile.get("current_title") or profile.get("title") or "").lower()
    if any(kw in title for kw in _CV_SPEECH_TITLE_KW):
        return True
    skills_raw  = c.get("skills") or []
    skills_text = " ".join(
        str(s.get("name") if isinstance(s, dict) else s).lower() for s in skills_raw
    )
    return sum(1 for kw in _CV_SPEECH_SKILLS if kw in skills_text) >= 2


def _candidate_title_work_text(c: dict) -> str:
    """Title + work titles/descriptions — sharper donts signal than full-profile blob."""
    profile = c.get("profile") or {}
    parts   = []
    t = str(
        profile.get("current_title") or profile.get("title") or profile.get("headline") or ""
    ).strip()
    if t:
        parts.append(t)
    for exp in utils._iter_work_history(c):
        if not isinstance(exp, dict):
            continue
        for field in ("title", "description"):
            v = exp.get(field, "")
            if v:
                parts.append(str(v))
    return " ".join(parts)


def _jd_responsibilities_text(jd: dict) -> str:
    """Extract the JD 'what you'll do' / responsibilities text."""
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
    # Fallback: use the full JD description
    return str(jd.get("description") or jd.get("job_description") or "").strip()



_L4_CHAR_LIMIT = 2000            # pre-truncate before tokenisation; model caps at 256 wp anyway

# Donts gate: cosine sim below this is treated as noise and ignored entirely.
_DONTS_SIM_THRESHOLD      = 0.30
# candidate_final_score = 0.5·l3 + 0.5·l4_work_relevance − 0.6·donts_penalty
# Negative final score → HARD REJECT
_L4_WORK_WEIGHT           = 0.5
_L4_DONTS_WEIGHT          = 0.6
# Near-zero work similarity → hard reject (candidate removed from output)
_L4_WORK_SIM_HARD_REJECT  = 0.05


def l4_semantic_work(candidates: List[dict], jd: dict) -> List[dict]:
    """
    Layer 4 — Semantic Work Relevance + Donts Penalty.

    Two JD embeddings are built once:
      1. responsibilities / candidate_work text  → jd_work_vec
      2. all jd.donts joined as one text         → jd_donts_vec  (None if donts absent)

    Per candidate:
      l4_work_relevance  = cosine(work_descriptions, jd_work_vec)    [0, 1]
      l4_donts_sim       = cosine(title+work text,   jd_donts_vec)   [0, 1]  (0 if no donts)

      Hard-reject        : l4_work_relevance < _L4_WORK_SIM_HARD_REJECT → removed from output

      l4_donts_penalty   = _L4_DONTS_WEIGHT × l4_donts_sim   if donts present & sim > threshold
                         = _L4_DONTS_WEIGHT                   if categorical CV/speech candidate
                         = 0                                   otherwise  (cannot backfire)

      candidate_final_score = 0.5 × l3_score + 0.5 × l4_work_relevance − l4_donts_penalty
      Hard-reject           : candidate_final_score < 0  (negative → removed from output)
      l4_score              = 0.5 × l4_work_relevance   (L4 positive contribution only)
      l4_combined_score     = candidate_final_score      (sort key; alias for downstream)

    Donts component can only subtract — it is zero when JD has no donts and zero when
    donts similarity is below the noise threshold, so it can never inflate scores.
    Returns surviving candidates (hard-rejects removed) sorted by candidate_final_score.
    """
    model = utils.load_sentence_transformer()
    n     = len(candidates)

    # ── JD embeddings (one encode call) ──────────────────────────────────────
    jd_work_text  = _jd_responsibilities_text(jd)
    if not jd_work_text:
        logger.warning("L4: JD has no responsibilities text — work_relevance=0 for all")

    donts_list    = [str(d) for d in (jd.get("donts") or []) if d]
    jd_donts_text = " . ".join(donts_list) if donts_list else ""

    # Build a compact list of non-empty JD texts, track which index is which
    jd_to_encode: List[str] = []
    _jd_work_idx = _jd_donts_idx = None
    if jd_work_text:
        _jd_work_idx = len(jd_to_encode)
        jd_to_encode.append(jd_work_text[:_L4_CHAR_LIMIT])
    if jd_donts_text:
        _jd_donts_idx = len(jd_to_encode)
        jd_to_encode.append(jd_donts_text[:_L4_CHAR_LIMIT])

    jd_work_vec  = np.zeros(C.EMBED_DIM, dtype=np.float32)
    jd_donts_vec = None

    if jd_to_encode:
        jd_vecs = model.encode(
            jd_to_encode,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if _jd_work_idx is not None:
            jd_work_vec = jd_vecs[_jd_work_idx]
        if _jd_donts_idx is not None:
            jd_donts_vec = jd_vecs[_jd_donts_idx]

    # ── Candidate work texts → work relevance ────────────────────────────────
    work_texts      = [utils.get_work_descriptions_only(c)[:_L4_CHAR_LIMIT] for c in candidates]
    non_empty_work  = [i for i, t in enumerate(work_texts) if t.strip()]
    work_relevances = [0.0] * n

    if non_empty_work and jd_work_text:
        work_mat: np.ndarray = model.encode(
            [work_texts[i] for i in non_empty_work],
            batch_size=C.L4_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sims = work_mat @ jd_work_vec
        for pos, idx in enumerate(non_empty_work):
            work_relevances[idx] = max(0.0, float(sims[pos]))

    # ── Candidate title+work texts → donts similarity ────────────────────────
    # Narrower text (title + work only) gives sharper domain signal than full blob.
    donts_sims = [0.0] * n

    if jd_donts_vec is not None:
        title_work_texts = [_candidate_title_work_text(c)[:_L4_CHAR_LIMIT] for c in candidates]
        non_empty_prof   = [i for i, t in enumerate(title_work_texts) if t.strip()]

        if non_empty_prof:
            prof_mat: np.ndarray = model.encode(
                [title_work_texts[i] for i in non_empty_prof],
                batch_size=C.L4_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            sims = prof_mat @ jd_donts_vec
            for pos, idx in enumerate(non_empty_prof):
                donts_sims[idx] = max(0.0, float(sims[pos]))

    # Pre-compute categorical CV/speech flags (one pass, no repeated calls)
    is_cv_speech = [_is_cv_speech_candidate(c) for c in candidates]

    # ── Combine: candidate_final_score = 0.5·l3 + 0.5·work − donts_penalty ──
    # Hard-reject (1): near-zero work relevance.
    # Hard-reject (2): negative final score (donts penalty exceeds positive signals).
    approved: List[dict] = []
    n_rejected = 0

    for i, c in enumerate(candidates):
        wr = round(float(work_relevances[i]), 4)
        ds = round(float(donts_sims[i]), 4)

        c["l4_work_relevance"] = wr
        c["l4_donts_sim"]      = ds

        # Hard reject: near-zero work relevance → not a viable candidate
        if wr < _L4_WORK_SIM_HARD_REJECT:
            c["l4_hard_reject"]        = True
            c["l4_donts_penalty"]      = 0.0
            c["l4_score"]              = 0.0
            c["candidate_final_score"] = 0.0
            c["l4_combined_score"]     = 0.0
            n_rejected += 1
            continue

        # Donts penalty: subtracts _L4_DONTS_WEIGHT × sim when donts are active
        # and above the noise floor.  Zero otherwise → cannot backfire.
        if is_cv_speech[i]:
            donts_penalty = _CATEGORICAL_DONTS_PENALTY   # == _L4_DONTS_WEIGHT
        elif jd_donts_vec is not None and ds > _DONTS_SIM_THRESHOLD:
            donts_penalty = _L4_DONTS_WEIGHT * ds
        else:
            donts_penalty = 0.0

        l3  = float(c.get("l3_score") or 0.0)
        # candidate_final_score = 0.5·l3 + 0.5·work_relevance − donts_penalty
        final_score = round(_L4_WORK_WEIGHT * l3 + _L4_WORK_WEIGHT * wr - donts_penalty, 4)

        # Hard reject: negative final score (donts penalty wiped out positive signals)
        if final_score < 0.0:
            c["l4_hard_reject"]        = True
            c["l4_donts_penalty"]      = round(donts_penalty, 4)
            c["l4_score"]              = 0.0
            c["candidate_final_score"] = final_score
            c["l4_combined_score"]     = final_score
            n_rejected += 1
            continue

        c["l4_hard_reject"]        = False
        c["l4_donts_penalty"]      = round(donts_penalty, 4)
        c["l4_score"]              = round(_L4_WORK_WEIGHT * wr, 4)  # L4 positive contribution
        c["candidate_final_score"] = final_score
        c["l4_combined_score"]     = final_score  # sort key; alias for downstream
        approved.append(c)

    approved.sort(key=lambda c: c["candidate_final_score"], reverse=True)

    n_kept  = len(approved)
    n_pen   = sum(1 for c in approved if c["l4_donts_penalty"] > 0)
    avg_wr  = sum(c["l4_work_relevance"]      for c in approved) / max(n_kept, 1)
    avg_cmb = sum(c["candidate_final_score"]  for c in approved) / max(n_kept, 1)
    logger.info(
        f"L4 semantic: {n} in → {n_kept} kept, {n_rejected} hard-rejected "
        f"(avg_work={avg_wr:.3f}, avg_final={avg_cmb:.3f}, "
        f"donts_penalised={n_pen}/{n_kept}, donts_rules={len(donts_list)})"
    )
    return approved


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — FLASHRANK CROSS-ENCODER RE-RANK
# ══════════════════════════════════════════════════════════════════════════════


def l5_flashrank_rerank(candidates: List[dict], _jd: dict) -> List[dict]:
    """
    Layer 5 — FlashRank cross-encoder re-rank (DISABLED).

    Returns candidates unchanged. Re-enable by restoring the body below.
    """
    # L5 DISABLED — top-100 from L4 is returned as-is, no FlashRank reshuffling.
    return candidates

    # ── DISABLED BODY (re-enable by removing the `return candidates` above) ──
    #
    # Runs ms-marco-MiniLM-L-12-v2 on the top C.FLASHRANK_TOP_N (50) candidates.
    # Raw cross-encoder logits are min-max normalised across that batch (not
    # sigmoid-squashed) so l5_flashrank_score always spans the full [0, 1] range
    # relative to its own batch before being averaged in.
    #
    # total_score = (l3_score + l4_score + flashrank_score) / 3   — for the
    # reranked top-50 (l4_score already has the donts penalty baked in from L4).
    #
    # Candidates outside top-50 are never cross-encoded; total_score for those
    # falls back to (l3_score + l4_score) / 2.  Same fallback applies for
    # everyone if FlashRank is not installed or fails.
    #
    # # Initialise keys so they always exist regardless of degradation path
    # for c in candidates:
    #     c["l5_flashrank_score"] = 0.0
    #     c["l5_total_score"]     = round(
    #         (float(c.get("l3_score") or 0.0) + float(c.get("l4_score") or 0.0)) / 2.0,
    #         4,
    #     )
    #
    # try:
    #     from flashrank import RerankRequest
    #     ranker = utils.load_flashrank()
    #     if ranker is None:
    #         raise RuntimeError("FlashRank ranker is None")
    # except Exception as exc:
    #     logger.warning(f"L5 FlashRank unavailable ({exc}); total_score = (l3_score + l4_score) / 2")
    #     return candidates
    #
    # ranker  = utils.load_flashrank()
    # jd_text = _jd_responsibilities_text(jd)
    #
    # if not jd_text:
    #     logger.warning("L5: JD has no responsibilities text — flashrank_score=0 for all")
    #     return candidates
    #
    # top_n    = min(C.FLASHRANK_TOP_N, len(candidates))
    # top_pool = candidates[:top_n]
    #
    # passages = [
    #     {"id": i, "text": utils.get_work_descriptions_only(c) or ""}
    #     for i, c in enumerate(top_pool)
    # ]
    # results = ranker.rerank(RerankRequest(query=jd_text, passages=passages))
    #
    # raw_scores: Dict[int, float] = {
    #     r["id"]: float(r.get("score", 0.0)) for r in results
    # }
    # # Min-max normalise the raw logits across this batch → [0, 1].
    # # A degenerate batch (max == min, e.g. all-identical or single candidate)
    # # normalises to a neutral 0.5 rather than dividing by zero.
    # raw_vals = list(raw_scores.values())
    # lo, hi   = (min(raw_vals), max(raw_vals)) if raw_vals else (0.0, 0.0)
    # span     = hi - lo
    #
    # def _minmax(v: float) -> float:
    #     return 0.5 if span <= 0 else max(0.0, min(1.0, (v - lo) / span))
    #
    # id_to_score: Dict[int, float] = {i: _minmax(v) for i, v in raw_scores.items()}
    #
    # for i, c in enumerate(top_pool):
    #     fr_score = id_to_score.get(i, 0.0)
    #     c["l5_flashrank_score"] = round(fr_score, 4)
    #     c["l5_total_score"]     = round(
    #         (
    #             float(c.get("l3_score") or 0.0)
    #             + float(c.get("l4_score") or 0.0)
    #             + fr_score
    #         ) / 3.0,
    #         4,
    #     )
    #
    # top_reranked = sorted(top_pool, key=lambda c: c["l5_total_score"], reverse=True)
    # tail         = candidates[top_n:]
    #
    # avg_fr = sum(c["l5_flashrank_score"] for c in top_reranked) / top_n
    # logger.info(
    #     f"L5 flashrank: cross-encoded top {top_n} of {len(candidates)} "
    #     f"(avg_fr={avg_fr:.3f}, min-max normalised); "
    #     f"total_score = (l3_score + l4_score + FlashRank) / 3"
    # )
    # return top_reranked + tail

