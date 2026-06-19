"""
fis.py — Layer 7: Mamdani Fuzzy Inference System + FlashRank polish.

Combines L2, L3, L4, L6 signals via fuzzy reasoning into a final
composite score, ranks, polishes top 50 with FlashRank, applies tie-breaks.
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


def _low(x):  return _tri(x, -0.1, 0.0, 0.5)
def _med(x):  return _tri(x, 0.2, 0.5, 0.8)
def _high(x): return _tri(x, 0.5, 1.0, 1.1)


def _fuzzy_score(l2, l4, l6) -> float:
    """
    Mamdani inference. Inputs already in [0,1].
    Rules express recruiter-style reasoning:
      R1: l2 HIGH & l4 HIGH      → fit HIGH
      R2: l2 HIGH & l4 MED       → fit HIGH
      R3: l2 MED  & l4 HIGH      → fit HIGH
      R4: l2 MED  & l4 MED       → fit MED
      R5: l2 LOW                 → fit LOW
      R6: l6 HIGH                → nudge fit up
    Defuzzify via weighted centroid of {LOW=0.2, MED=0.55, HIGH=0.9}.
    """
    l2L, l2M, l2H = _low(l2), _med(l2), _high(l2)
    l4L, l4M, l4H = _low(l4), _med(l4), _high(l4)
    l6H = _high(l6)

    # rule firing strengths
    high_fire = max(
        min(l2H, l4H),
        min(l2H, l4M),
        min(l2M, l4H),
        0.6 * l6H,           # behavioral nudge
    )
    med_fire = max(
        min(l2M, l4M),
        min(l2H, l4L),
        min(l2L, l4H),
    )
    low_fire = max(l2L, min(l2L, l4L))

    num = high_fire * 0.9 + med_fire * 0.55 + low_fire * 0.2
    den = high_fire + med_fire + low_fire
    return float((num / den) if den > 0 else (0.5 * l2 + 0.3 * l4 + 0.2 * l6))


def run_fis(candidates: List[dict]) -> List[dict]:
    """Compute composite fis_score for each candidate."""
    for c in candidates:
        l2 = c.get("l2_score", 0.0)
        l4 = c.get("l4_score", 0.0)
        l6 = c.get("l6_score", 0.0)
        l3_pen = c.get("l3_penalty", 1.0)

        fuzzy = _fuzzy_score(l2, l4, l6)
        # L3 seniority penalty applied on top
        c["fis_score"] = float(fuzzy * l3_pen)
    logger.info(f"FIS: scored {len(candidates)} candidates")
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
                row = fraud_kb.execute(
                    "SELECT founding_year FROM company_founding_dates WHERE LOWER(company_name)=?",
                    (comp,),
                ).fetchone()
                if row and row["founding_year"]:
                    oldest = min(oldest, int(row["founding_year"]))
    cid = str(c.get("candidate_id", c.get("id", "")))
    # negate score & exp for descending; oldest ascending; cid ascending
    return (-score, -exp, oldest, cid)


def rank_candidates(candidates: List[dict], fraud_kb) -> List[dict]:
    """Sort with full tie-breaking chain."""
    return sorted(candidates, key=lambda c: _tiebreak_key(c, fraud_kb))


# ──────────────────────────────────────────────────────────────────────────────
# FLASHRANK POLISH (top 50)
# ──────────────────────────────────────────────────────────────────────────────
def flashrank_polish(ranked: List[dict], jd: dict, ranker, fraud_kb) -> List[dict]:
    """
    Re-rank the top FLASHRANK_TOP_N candidates using a blend of FlashRank
    cross-encoder score and the L7 FIS score.

    Ranks 1-50:  final_score = FLASHRANK_FIS_WEIGHT * flashrank_norm + (1 - FLASHRANK_FIS_WEIGHT) * fis_score
                 (FIS weight = 0.75, FlashRank weight = 0.25)
    Ranks 51-100: final_score = fis_score  (pure FIS, no FlashRank)

    The original FIS score is preserved as 'l7_fis_raw'; 'fis_score' is updated
    to the blended value so downstream CSV output reflects the combined ranking.
    """
    if ranker is None or len(ranked) == 0:
        logger.warning("FlashRank unavailable; skipping polish")
        return ranked

    top_n = min(C.FLASHRANK_TOP_N, len(ranked))
    top = ranked[:top_n]
    mid = ranked[top_n:100]   # ranks 51-100: pure FIS, no FlashRank
    tail = ranked[100:]

    jd_text = " ".join(
        jd.get("explicit_required", []) + jd.get("inferred_required", [])
        + [jd.get("job_title", "")]
    )

    try:
        from flashrank import RerankRequest
        passages = [
            {"id": i, "text": utils.get_work_profile_text(c) or " ",
             "meta": {"orig_idx": i}}
            for i, c in enumerate(top)
        ]
        req = RerankRequest(query=jd_text, passages=passages)
        results = ranker.rerank(req)

        # Map candidate index → raw FlashRank score
        fr_raw: dict[int, float] = {r["id"]: float(r.get("score", 0.0)) for r in results}

        # Min-max normalize FlashRank scores to [0, 1]
        fr_values = list(fr_raw.values())
        fr_min, fr_max = min(fr_values), max(fr_values)
        fr_range = fr_max - fr_min if fr_max > fr_min else 1.0

        w = C.FLASHRANK_FIS_WEIGHT
        for i, c in enumerate(top):
            raw = fr_raw.get(i, fr_min)
            fr_norm = (raw - fr_min) / fr_range
            orig_fis = c.get("fis_score", 0.0)
            c["l7_fis_raw"] = round(orig_fis, 4)
            c["flashrank_score"] = round(fr_norm, 4)
            c["fis_score"] = float(w * fr_norm + (1.0 - w) * orig_fis)

        # Sort by blended score descending
        reordered = sorted(top, key=lambda c: c["fis_score"], reverse=True)
        for rank_pos, c in enumerate(reordered):
            c["flashrank_rank"] = rank_pos

        # Ranks 51-100: mark as pure FIS (no FlashRank blending)
        for c in mid:
            c["l7_fis_raw"] = round(c.get("fis_score", 0.0), 4)
            c["flashrank_score"] = None

        logger.info(
            f"FlashRank: blended top {top_n} "
            f"(weight={w} FR + {1-w} FIS); "
            f"fr_range=[{fr_min:.3f}, {fr_max:.3f}]; "
            f"ranks 51-100 use pure FIS"
        )
        return reordered + mid + tail
    except Exception as e:
        logger.warning(f"FlashRank polish failed ({e}); keeping FIS order")
        return ranked


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

    l2 = c.get("l2_score", 0.0)
    l4 = c.get("l4_score", 0.0)
    if l2 >= 0.6:
        bits.append("strong overall profile-JD match")
    elif l2 >= 0.4:
        bits.append("moderate profile-JD match")
    else:
        bits.append("limited profile-JD match")

    if l4 >= 0.5:
        bits.append("work history closely aligns with JD responsibilities")
    elif l4 > 0:
        bits.append("some work-history alignment")

    # honest concerns
    if c.get("l3_flag"):
        bits.append(c["l3_flag"])

    l6 = c.get("l6_score", 0.0)
    if l6 >= 0.7:
        bits.append("strong engagement signals")
    elif l6 <= 0.3:
        bits.append("weak engagement signals")

    text = "; ".join(bits)
    return text[0].upper() + text[1:] if text else "Candidate evaluated across all layers."
