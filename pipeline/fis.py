"""
fis.py — Layer 7: Mamdani Fuzzy Inference System.

Combines L1c, L3, L4, L6 signals via fuzzy reasoning into a final
composite score, assigns a tier, ranks by tier then score, applies
tie-breaks.

Tiers (assigned in run_fis, enforced in rank_candidates):
  "very_good"  — meets ALL excellence criteria → guaranteed top-10 placement
  "eligible"   — clean candidate (no flags, no L3 penalty), ranked by FIS score
  "penalized"  — has L1b flags OR L3 penalty → excluded from top 100 if 100 clean exist
"""

import logging
from typing import List, Dict

import numpy as np

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# FUZZY MEMBERSHIP FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def _tri(x, a, b, c):
    """Triangular membership."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _low(x):      return _tri(x, -0.1, 0.0, 0.5)
def _med(x):      return _tri(x, 0.2, 0.5, 0.8)
def _high(x):     return _tri(x, 0.5, 1.0, 1.1)
def _exp_high(x): return _tri(x, 5.0, 12.0, 30.0)  # experience years: HIGH at ~12+


def _fuzzy_score(
    l1c: float,
    l4: float,
    l6: float,
    composite: float = 0.0,
    l3_penalty: float = 1.0,
    exp_years: float = 0.0,
    company_age_years: float = 0.0,
) -> float:
    """
    Mamdani inference. Inputs already in [0,1] except exp_years and company_age_years.
    Rules:
      R1: l1c HIGH & l4 HIGH                                       → fit HIGH
      R2: l1c HIGH & l4 MED                                        → fit HIGH
      R3: l1c MED  & l4 HIGH                                       → fit HIGH
      R4: l1c MED  & l4 MED                                        → fit MED
      R5: l1c LOW                                                   → fit LOW
      R6: l6 HIGH                                                   → nudge fit up
      R7: composite HIGH & l6 HIGH & L3 ok & l1c HIGH & l4 HIGH
          & exp HIGH & company >10yrs                               → fit VERY_HIGH
    Defuzzify via weighted centroid of {LOW=0.2, MED=0.55, HIGH=0.9, VERY_HIGH=0.97}.
    """
    l1cL, l1cM, l1cH = _low(l1c), _med(l1c), _high(l1c)
    l4L, l4M, l4H = _low(l4), _med(l4), _high(l4)
    l6H = _high(l6)

    # rule firing strengths
    high_fire = max(
        min(l1cH, l4H),
        min(l1cH, l4M),
        min(l1cM, l4H),
        0.6 * l6H,           # behavioral nudge
    )
    med_fire = max(
        min(l1cM, l4M),
        min(l1cH, l4L),
        min(l1cL, l4H),
    )
    low_fire = max(l1cL, min(l1cL, l4L))

    # R7 — "very good candidate" rule
    l3_ok = 1.0 if l3_penalty >= 1.0 else 0.0
    co_age_membership = min(1.0, company_age_years / 10.0) if company_age_years > 0 else 0.0
    very_high_fire = min(
        _high(composite),
        l6H,
        l3_ok,
        l1cH,
        l4H,
        _exp_high(exp_years),
        co_age_membership,
    )

    num = very_high_fire * 0.97 + high_fire * 0.9 + med_fire * 0.55 + low_fire * 0.2
    den = very_high_fire + high_fire + med_fire + low_fire
    return float((num / den) if den > 0 else (0.5 * l1c + 0.3 * l4 + 0.2 * l6))


def _get_oldest_company_age(c: dict, fraud_kb, current_year: int) -> float:
    """Return age in years of the oldest company in the candidate's work history."""
    oldest_year = None
    for e in utils._iter_work_history(c):
        if not isinstance(e, dict):
            continue
        comp = str(e.get("company", "")).strip().lower()
        if fraud_kb is not None and comp:
            with utils.FRAUD_KB_LOCK:
                row = fraud_kb.execute(
                    "SELECT founding_year FROM company_founding_dates WHERE LOWER(company_name)=?",
                    (comp,),
                ).fetchone()
            if row and row["founding_year"]:
                yr = int(row["founding_year"])
                if oldest_year is None or yr < oldest_year:
                    oldest_year = yr
    if oldest_year is None:
        return 0.0
    return max(0.0, float(current_year - oldest_year))


def _is_very_good(l1c: float, l4: float, l6: float,
                  exp_years: float, company_age: float, c: dict) -> bool:
    """
    Return True if a candidate meets ALL excellence criteria:
      - l1c_score >= L7_VERY_GOOD_L1C_MIN  (skill match high)
      - l4_score  >= L7_VERY_GOOD_L4_MIN   (semantic work relevance high)
      - l6_score  >= L7_VERY_GOOD_L6_MIN   (behavioral signals high)
      - exp_years >= L7_VERY_GOOD_EXP_MIN  (experience high)
      - company_age >= L7_VERY_GOOD_COMPANY_AGE_MIN OR no company data (age == 0)
      - no L1b flags (profile integrity clean)
      - no L3 penalty (seniority fit)
    """
    company_ok = (company_age == 0.0) or (company_age >= C.L7_VERY_GOOD_COMPANY_AGE_MIN)
    return (
        l1c >= C.L7_VERY_GOOD_L1C_MIN
        and l4 >= C.L7_VERY_GOOD_L4_MIN
        and l6 >= C.L7_VERY_GOOD_L6_MIN
        and exp_years >= C.L7_VERY_GOOD_EXP_MIN
        and company_ok
        and not c.get("l1b_flags")
        and c.get("l3_penalty", 1.0) >= 1.0
    )


def run_fis(candidates: List[dict], fraud_kb=None) -> List[dict]:
    """
    Compute composite_score, fis_score, and l7_tier for each candidate.

    l7_tier values:
      "very_good"  — meets all excellence criteria (→ top 10 in ranked output)
      "eligible"   — clean candidate, ranked by FIS score (→ top 100)
      "penalized"  — has L1b flags OR L3 penalty (→ excluded from top 100 if
                     100 clean candidates exist; fills remaining slots if not)
    """
    import datetime
    current_year = datetime.datetime.now().year

    n_very_good = n_eligible = n_penalized = 0

    for c in candidates:
        l1c = c.get("l1c_score", 0.0)
        l4 = c.get("l4_score", 0.0)
        l6 = c.get("l6_score", 0.0)
        l3_pen = c.get("l3_penalty", 1.0)
        l1b_pen = c.get("l1b_penalty", 1.0)

        composite = 0.5 * l4 + 0.3 * l1c + 0.1 * l6 + 0.1 * l3_pen
        c["composite_score"] = round(composite, 4)

        exp_years = utils.get_total_experience_years(c)
        company_age = _get_oldest_company_age(c, fraud_kb, current_year)

        fuzzy = _fuzzy_score(l1c, l4, l6, composite, l3_pen, exp_years, company_age)
        c["fis_score"] = float(fuzzy * l3_pen * l1b_pen)

        # Assign tier
        is_penalized = bool(c.get("l1b_flags")) or l3_pen < 1.0
        if is_penalized:
            c["l7_tier"] = "penalized"
            n_penalized += 1
        elif _is_very_good(l1c, l4, l6, exp_years, company_age, c):
            c["l7_tier"] = "very_good"
            n_very_good += 1
        else:
            c["l7_tier"] = "eligible"
            n_eligible += 1

    logger.info(
        f"FIS: {len(candidates)} scored — "
        f"very_good={n_very_good}, eligible={n_eligible}, penalized={n_penalized}"
    )
    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# TIE-BREAKING
# ──────────────────────────────────────────────────────────────────────────────
def _tiebreak_key(c: dict, fraud_kb):
    """
    Sort key: higher fis_score, then higher experience,
    then OLDER company (smaller founding year), then candidate_id asc.
    Returned tuple is for DESC sort on score/exp, ASC on founding & id.
    """
    score = c.get("fis_score", 0.0)
    exp = utils.get_total_experience_years(c)

    # oldest company founding year among work experience
    oldest = 9999
    for e in utils._iter_work_history(c):
        if isinstance(e, dict):
            comp = str(e.get("company", "")).strip().lower()
            if fraud_kb is not None and comp:
                with utils.FRAUD_KB_LOCK:
                    row = fraud_kb.execute(
                        "SELECT founding_year FROM company_founding_dates WHERE LOWER(company_name)=?",
                        (comp,),
                    ).fetchone()
                if row and row["founding_year"]:
                    oldest = min(oldest, int(row["founding_year"]))
    cid = str(c.get("candidate_id", c.get("id", "")))
    # negate score & exp for descending; oldest ascending; cid ascending
    return (-score, -exp, oldest, cid)


_TIER_ORDER = {"very_good": 0, "eligible": 1, "penalized": 2}


def rank_candidates(candidates: List[dict], fraud_kb) -> List[dict]:
    """
    Tiered sort:
      Tier 0 — very_good   (all excellence criteria met, no penalties)
      Tier 1 — eligible    (clean, no flags/penalties, ranked by FIS score)
      Tier 2 — penalized   (L1b flags OR L3 penalty present)

    Within each tier, the full tiebreak chain applies.

    Effect on output:
      • very_good candidates occupy the first slots → guaranteed top 10 if ≥10 exist.
      • eligible candidates fill the rest of the top 100.
      • penalized candidates appear only if fewer than 100 clean candidates exist.
    """
    def _sort_key(c: dict):
        tier = _TIER_ORDER.get(c.get("l7_tier", "eligible"), 1)
        return (tier,) + _tiebreak_key(c, fraud_kb)

    return sorted(candidates, key=_sort_key)


# ──────────────────────────────────────────────────────────────────────────────
# REASONING GENERATION
# ──────────────────────────────────────────────────────────────────────────────
def generate_reasoning(c: dict, jd: dict) -> str:
    """Build a specific, honest 1-2 sentence reasoning string."""
    bits = []
    years = utils.get_total_experience_years(c)
    title = ""
    we = list(utils._iter_work_history(c))
    if we and isinstance(we[0], dict):
        title = str(we[0].get("title", "")).strip()

    lead = f"{years:.0f}y experience" + (f" as {title}" if title else "")
    bits.append(lead)

    if c.get("l7_tier") == "very_good":
        bits.append("exceptional profile: meets all top-tier criteria")

    l1c = c.get("l1c_score", 0.0)
    l4 = c.get("l4_score", 0.0)
    matched = c.get("l1c_matched_required", [])
    missing = c.get("l1c_missing_required", [])
    if l1c >= 0.7:
        bits.append(f"strong skill match ({len(matched)} required skills matched)")
    elif l1c >= 0.4:
        bits.append(
            f"partial skill match ({len(matched)} of {len(matched)+len(missing)} required skills)"
        )
    else:
        miss_str = ", ".join(missing[:3])
        bits.append(
            f"weak skill match (missing: {miss_str})" if missing else "weak skill match"
        )

    if l4 >= 0.5:
        bits.append("work history closely aligns with JD responsibilities")
    elif l4 > 0:
        bits.append("some work-history alignment")

    # honest concerns
    if c.get("l1b_flags"):
        readable = [f.replace("_", " ") for f in c["l1b_flags"]]
        bits.append("profile flags: " + ", ".join(readable))
    if c.get("l3_flag"):
        bits.append(c["l3_flag"])

    l6 = c.get("l6_score", 0.0)
    if l6 >= 0.7:
        bits.append("strong engagement signals")
    elif l6 <= 0.3:
        bits.append("weak engagement signals")

    text = "; ".join(bits)
    return text[0].upper() + text[1:] if text else "Candidate evaluated across all layers."
