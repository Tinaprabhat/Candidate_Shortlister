"""
tests/honeypot_stresstest.py — Stress tests for L1 honeypot / fraud detection.

Covers fictional companies, inverted salaries, impossible education timelines,
work-before-founding anomalies, batch processing, and edge cases.

Run:  python -m pytest tests/honeypot_stresstest.py -v
  or: python tests/honeypot_stresstest.py
"""

import sys
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import layers

TESTS_DIR = Path(__file__).parent


# ── fixtures ──────────────────────────────────────────────────────────────────
def mock_kb():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE fictional_companies (company_name TEXT)")
    conn.execute("CREATE TABLE company_founding_dates (company_name TEXT, founding_year INT)")
    conn.executemany("INSERT INTO fictional_companies VALUES (?)", [("hooli",), ("initech",)])
    conn.executemany("INSERT INTO company_founding_dates VALUES (?,?)",
                     [("realtech", 2010), ("youngco", 2022)])
    conn.commit()
    return conn


def good_candidate(cid="CAND_0000001", years=6):
    return {
        "candidate_id": cid, "total_experience_years": years,
        "skills": ["Python", "RAG", "LLM"],
        "work_experience": [{"title": "ML Engineer", "company": "realtech",
                             "start_year": 2016, "duration_years": years,
                             "description": "Built RAG and LLM systems with Python and FAISS."}],
        "projects": [{"description": "Semantic search using sentence transformers."}],
        "education": [{"degree": "Bachelor", "end_year": 2015}],
        "behavioral_signals": {
            "github_activity_score": 0.8, "profile_completeness": 0.9,
            "recruiter_response_rate": 0.7, "salary_fit": 0.6,
            "notice_period_score": 0.5,
        },
    }


# ── core honeypot cases ───────────────────────────────────────────────────────
def test_honeypot_fictional_company():
    kb = mock_kb()
    hp = {"candidate_id": "HP1", "skills": ["Python"],
          "work_experience": [{"title": "Eng", "company": "Hooli",
                               "start_year": 2015, "description": "x"}],
          "education": [{"degree": "Bachelor", "end_year": 2014}]}
    assert len(layers.l1_hard_reject([hp], kb)) == 0, "fictional-company honeypot leaked"
    print("  ✅ fictional company (Hooli) rejected")


def test_honeypot_salary_inverted():
    kb = mock_kb()
    hp = {"candidate_id": "HP2", "skills": ["Python"],
          "redrob_signals": {"expected_salary_range_inr_lpa": {"min": 200000, "max": 50000}},
          "work_experience": [{"title": "Dev", "company": "realtech",
                               "start_year": 2017, "description": "x"}],
          "education": [{"degree": "Bachelor", "end_year": 2016}]}
    assert len(layers.l1_hard_reject([hp], kb)) == 0, "inverted salary honeypot leaked"
    print("  ✅ inverted salary (min 200k > max 50k) rejected")


def test_honeypot_phd_before_bachelor():
    kb = mock_kb()
    hp = {"candidate_id": "HP3", "skills": ["ML"],
          "education": [{"degree": "PhD", "end_year": 2010},
                        {"degree": "Bachelor", "end_year": 2015}]}
    assert len(layers.l1_hard_reject([hp], kb)) == 0, "degree-timeline honeypot leaked"
    print("  ✅ PhD-before-Bachelor (2010 < 2015) rejected")


def test_honeypot_worked_before_founding():
    kb = mock_kb()
    hp = {"candidate_id": "HP4", "skills": ["Python"],
          "work_experience": [{"title": "Senior Engineer", "company": "youngco",
                               "start_year": 2018, "description": "x"}],
          "education": [{"degree": "Bachelor", "end_year": 2010}]}
    assert len(layers.l1_hard_reject([hp], kb)) == 0, "work-before-founding honeypot leaked"
    print("  ✅ worked-before-company-founded (2018 < 2022) rejected")


def test_good_candidate_survives_l1():
    kb = mock_kb()
    assert len(layers.l1_hard_reject([good_candidate()], kb)) == 1, \
        "legitimate candidate wrongly rejected by L1"
    print("  ✅ legitimate candidate survives L1")


# ── batch / JSONL stress tests ────────────────────────────────────────────────
def test_honeypot_batch_from_jsonl():
    """All 4 honeypot fixtures in honeypot_candidates.jsonl must be rejected."""
    jsonl_path = TESTS_DIR / "honeypot_candidates.jsonl"
    if not jsonl_path.exists():
        print("  ⚠️  honeypot_candidates.jsonl not found — skipping")
        return
    kb = mock_kb()
    cases = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    survivors = layers.l1_hard_reject(cases, kb)
    leaked = [c["candidate_id"] for c in survivors]
    assert len(survivors) == 0, f"{len(survivors)} JSONL honeypot(s) leaked: {leaked}"
    print(f"  ✅ all {len(cases)} JSONL honeypots rejected")


def test_honeypot_mixed_batch():
    """Honeypots mixed with legitimate candidates: only legit survive."""
    kb = mock_kb()
    legit = [good_candidate(f"GOOD_{i}", years=4 + i) for i in range(5)]
    honeypots = [
        {"candidate_id": "MIX_HP1", "skills": ["Python"],
         "work_experience": [{"title": "Eng", "company": "Hooli",
                              "start_year": 2015, "description": "x"}],
         "education": [{"degree": "Bachelor", "end_year": 2014}]},
        {"candidate_id": "MIX_HP2", "skills": ["Python"],
         "redrob_signals": {"expected_salary_range_inr_lpa": {"min": 300000, "max": 10000}},
         "work_experience": [{"title": "Dev", "company": "realtech",
                              "start_year": 2018, "description": "x"}],
         "education": [{"degree": "Bachelor", "end_year": 2016}]},
    ]
    survivors = layers.l1_hard_reject(legit + honeypots, kb)
    survivor_ids = {c["candidate_id"] for c in survivors}
    for c in legit:
        assert c["candidate_id"] in survivor_ids, \
            f"legit {c['candidate_id']} wrongly rejected"
    for hp in honeypots:
        assert hp["candidate_id"] not in survivor_ids, \
            f"honeypot {hp['candidate_id']} leaked through L1"
    print(f"  ✅ mixed batch: {len(legit)} legit survived, {len(honeypots)} honeypots rejected")


def test_honeypot_large_batch_all_rejected():
    """100 fictional-company honeypots processed as a batch — all must be rejected."""
    kb = mock_kb()
    batch = [
        {"candidate_id": f"SPAM_{i:03d}", "skills": ["Python"],
         "work_experience": [{"title": "Eng", "company": "Initech",
                              "start_year": 2014 + (i % 5), "description": "x"}],
         "education": [{"degree": "Bachelor", "end_year": 2013}]}
        for i in range(100)
    ]
    survivors = layers.l1_hard_reject(batch, kb)
    assert len(survivors) == 0, \
        f"{len(survivors)} of 100 fictional-company honeypots leaked"
    print("  ✅ large batch: all 100 fictional-company candidates rejected")


# ── edge cases ────────────────────────────────────────────────────────────────
def test_honeypot_company_case_insensitive():
    """Fictional company check must be case-insensitive (HOOLI, hooli, Hooli all rejected)."""
    kb = mock_kb()
    for variant in ["HOOLI", "hooli", "Hooli", "hOoLi"]:
        hp = {"candidate_id": f"CASE_{variant}", "skills": ["Python"],
              "work_experience": [{"title": "Eng", "company": variant,
                                   "start_year": 2015, "description": "x"}],
              "education": [{"degree": "Bachelor", "end_year": 2014}]}
        survivors = layers.l1_hard_reject([hp], kb)
        assert len(survivors) == 0, \
            f"fictional company '{variant}' not caught (case sensitivity bug)"
    print("  ✅ fictional company check is case-insensitive across 4 variants")


def test_honeypot_salary_equal_is_valid():
    """min == max salary is unusual but not fraudulent — candidate should survive L1."""
    kb = mock_kb()
    c = good_candidate("SALARY_EQ")
    c.setdefault("redrob_signals", {})["expected_salary_range_inr_lpa"] = {"min": 100000, "max": 100000}
    assert len(layers.l1_hard_reject([c], kb)) == 1, \
        "min == max salary incorrectly flagged as honeypot"
    print("  ✅ min == max salary correctly passes L1")


def test_honeypot_no_education_survives():
    """Candidate with no education entries must not fail the degree-timeline check."""
    kb = mock_kb()
    c = {
        "candidate_id": "NO_EDU", "skills": ["Python"],
        "work_experience": [{"title": "Dev", "company": "realtech",
                             "start_year": 2015, "description": "built stuff"}],
        "education": [],
    }
    assert len(layers.l1_hard_reject([c], kb)) == 1, \
        "candidate with empty education[] incorrectly rejected at L1"
    print("  ✅ candidate with no education entries survives L1")


def test_honeypot_unknown_company_not_flagged():
    """Company not in the KB (neither fictional nor with a founding date) must pass."""
    kb = mock_kb()
    c = good_candidate("UNKNOWN_CO")
    c["work_experience"][0]["company"] = "some_unknown_startup_xyz_9999"
    assert len(layers.l1_hard_reject([c], kb)) == 1, \
        "candidate at unknown company (not in KB) incorrectly rejected"
    print("  ✅ candidate at unknown company correctly passes L1")


def test_honeypot_multiple_violations_rejected_once():
    """Candidate with multiple L1 violations is rejected exactly once (no duplication)."""
    kb = mock_kb()
    hp = {
        "candidate_id": "MULTI_VIO", "skills": ["Python"],
        "redrob_signals": {"expected_salary_range_inr_lpa": {"min": 500000, "max": 10000}},
        "work_experience": [{"title": "Eng", "company": "Hooli",
                             "start_year": 2015, "description": "x"}],
        "education": [{"degree": "PhD", "end_year": 2008},
                      {"degree": "Bachelor", "end_year": 2013}],
    }
    survivors = layers.l1_hard_reject([hp], kb)
    assert len(survivors) == 0, "multi-violation honeypot leaked through L1"
    print("  ✅ multi-violation candidate rejected (not duplicated in output)")


def test_honeypot_inverted_salary_extreme():
    """Extremely inverted salary (min=1M, max=1) must be rejected."""
    kb = mock_kb()
    hp = {
        "candidate_id": "SAL_EXTREME", "skills": ["Python"],
        "redrob_signals": {"expected_salary_range_inr_lpa": {"min": 1_000_000, "max": 1}},
        "work_experience": [{"title": "Dev", "company": "realtech",
                             "start_year": 2018, "description": "x"}],
        "education": [{"degree": "Bachelor", "end_year": 2017}],
    }
    assert len(layers.l1_hard_reject([hp], kb)) == 0, \
        "extreme inverted salary honeypot leaked"
    print("  ✅ extreme inverted salary (1M > 1) rejected")


# ── runner ────────────────────────────────────────────────────────────────────
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} honeypot stress tests\n")
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {t.__name__}: ERROR {e}")
            failed += 1
    print(f"\n{'='*50}\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
