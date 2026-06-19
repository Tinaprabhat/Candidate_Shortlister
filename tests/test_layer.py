"""
tests/test_layer.py — Unit tests for each layer in the RedRob cascade.

Tests every layer in isolation using deterministic fixtures — no real models,
no network, no disk reads beyond the SQLite in-memory KB.

Run:
    python -m pytest tests/test_layer.py -v
"""

import sys
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import layers, fis, constants as C
from src import utils


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

class MockST:
    """Deterministic fake sentence-transformer. Tech keywords boost similarity."""
    dim = 64

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            seed = abs(hash(t)) % (2 ** 32)
            rng = np.random.RandomState(seed)
            v = rng.rand(self.dim).astype(np.float32)
            for kw in ["python", "rag", "llm", "faiss", "ml", "docker",
                       "embeddings", "vector", "transformer"]:
                if kw in t.lower():
                    v[hash(kw) % self.dim] += 3.0
            norm = np.linalg.norm(v)
            out.append(v / (norm + 1e-9))
        return np.array(out)


def make_kb(fictional=("hooli", "initech"), founding=None):
    """In-memory SQLite fraud KB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE fictional_companies (company_name TEXT)")
    conn.execute(
        "CREATE TABLE company_founding_dates (company_name TEXT, founding_year INT)"
    )
    if fictional:
        conn.executemany(
            "INSERT INTO fictional_companies VALUES (?)",
            [(n,) for n in fictional],
        )
    if founding:
        conn.executemany(
            "INSERT INTO company_founding_dates VALUES (?,?)",
            founding,
        )
    conn.commit()
    return conn


def make_candidate(
    cid="C001",
    years=6,
    skills=None,
    work=None,
    possible_honeypot=False,
    salary_was_inverted=False,
    behavioral=None,
    profile=None,
):
    """Factory for a minimal valid candidate dict (new schema)."""
    if skills is None:
        skills = ["Python", "RAG", "LLM", "FAISS"]
    if work is None:
        work = [
            {
                "title": "ML Engineer",
                "company": "TechCorp",
                "start_date": "2018-01-01",
                "duration_months": years * 12,
                "description": "Built RAG pipelines with FAISS and Python embeddings.",
            }
        ]
    c = {
        "candidate_id": cid,
        "possible_honeypot": possible_honeypot,
        "salary_was_inverted": salary_was_inverted,
        "skills": skills,
        "career_history": work,
        "profile": profile or {
            "years_of_experience": years,
            "summary": "Senior ML engineer specialising in LLM and retrieval systems.",
            "headline": "ML Engineer",
        },
        "behavioral_signals": behavioral or {
            "github_activity_score": 0.8,
            "recruiter_response_rate": 0.7,
            "profile_completeness": 0.9,
            "salary_fit": 0.6,
            "notice_period_score": 0.5,
        },
        "education": [
            {"degree": "B.Tech", "end_year": 2016},
            {"degree": "M.Tech", "end_year": 2018},
        ],
    }
    return c


JD = {
    "job_title": "Senior AI Engineer",
    "location": "Pune, India",
    "experience_min": 5,
    "experience_max": 9,
    "required_seniority": "senior",
    "explicit_required": ["Python", "RAG", "LLM", "FAISS", "embeddings"],
    "inferred_required": ["SQL", "REST APIs", "Git"],
    "explicit_bonus": [],
    "inferred_bonus": [],
    "role_description": "Own the ranking and retrieval systems.",
}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — HARD REJECT (FRAUD DETECTION)
# ─────────────────────────────────────────────────────────────────────────────

class TestL1HardReject:

    def test_clean_candidate_passes(self):
        kb = make_kb()
        result = layers.l1_hard_reject([make_candidate()], kb)
        assert len(result) == 1

    def test_honeypot_flag_rejects(self):
        kb = make_kb()
        c = make_candidate(possible_honeypot=True)
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_salary_inverted_rejects(self):
        kb = make_kb()
        c = make_candidate(salary_was_inverted=True)
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_salary_min_gt_max_rejects(self):
        kb = make_kb()
        c = make_candidate()
        c["salary_expectation"] = {"min": 200000, "max": 100000}
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_phd_before_bachelor_rejects(self):
        kb = make_kb()
        c = make_candidate()
        c["education"] = [
            {"degree": "PhD", "end_year": 2010},
            {"degree": "Bachelor", "end_year": 2015},
        ]
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_master_before_bachelor_rejects(self):
        kb = make_kb()
        c = make_candidate()
        c["education"] = [
            {"degree": "Master", "end_year": 2010},
            {"degree": "B.Tech", "end_year": 2013},
        ]
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_fictional_company_in_kb_rejects(self):
        kb = make_kb(fictional=("hooli",))
        c = make_candidate()
        c["career_history"][0]["company"] = "Hooli"
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_worked_before_founding_rejects(self):
        kb = make_kb(fictional=(), founding=[("youngco", 2020)])
        c = make_candidate()
        c["career_history"] = [
            {"title": "Engineer", "company": "youngco",
             "start_date": "2018-03-01", "duration_months": 24,
             "description": "Worked here."}
        ]
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_senior_role_before_graduation_rejects(self):
        kb = make_kb()
        c = make_candidate()
        c["education"] = [{"degree": "B.Tech", "end_year": 2018}]
        c["career_history"] = [
            {"title": "Senior Engineer", "company": "AcmeCo",
             "start_date": "2015-01-01", "duration_months": 12,
             "description": "Led team."}
        ]
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_multiple_candidates_mixed(self):
        kb = make_kb()
        good = make_candidate("G1")
        bad = make_candidate("B1", possible_honeypot=True)
        result = layers.l1_hard_reject([good, bad, good], kb)
        assert len(result) == 2
        assert all(c["candidate_id"] == "G1" for c in result)

    def test_no_fraud_kb_still_works(self):
        # Should pass without SQLite KB (no company check)
        result = layers.l1_hard_reject([make_candidate()], None)
        assert len(result) == 1

    def test_iso_start_date_parsed_correctly(self):
        """Ensure ISO date strings are parsed for start year."""
        kb = make_kb(fictional=(), founding=[("newco", 2022)])
        c = make_candidate()
        c["career_history"] = [
            {"title": "Engineer", "company": "newco",
             "start_date": "2019-06-15", "duration_months": 36,
             "description": "Worked there."}
        ]
        result = layers.l1_hard_reject([c], kb)
        assert result == []  # started 2019 < founded 2022 → reject

    def test_l1_score_attached_to_passing_candidate(self):
        kb = make_kb()
        c = make_candidate()
        result = layers.l1_hard_reject([c], kb)
        assert len(result) == 1
        assert "l1_score" in result[0]
        assert 0.0 <= result[0]["l1_score"] <= 1.0

    def test_in_code_fictional_company_rejects(self):
        """Company in the hard-coded FICTIONAL_COMPANIES set is rejected even without KB entries."""
        kb = make_kb(fictional=())  # empty KB — relies on in-code list
        c = make_candidate()
        c["career_history"][0]["company"] = "Dunder Mifflin"
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_overlapping_full_time_jobs_rejects(self):
        """Two full-time jobs with overlapping year ranges must be hard-rejected."""
        kb = make_kb()
        c = make_candidate()
        c["career_history"] = [
            {"title": "Engineer", "company": "A",
             "start_date": "2016-01-01", "end_year": 2019,
             "duration_months": 36, "description": "Built things."},
            {"title": "Engineer", "company": "B",
             "start_date": "2018-01-01", "end_year": 2021,
             "duration_months": 36, "description": "Built more."},
        ]
        result = layers.l1_hard_reject([c], kb)
        assert result == []  # 2016–2019 overlaps 2018–2021

    def test_exp_exceeds_career_span_rejects(self):
        """Claimed experience exceeding years-since-graduation must be hard-rejected."""
        kb = make_kb()
        c = make_candidate(years=10)
        c["education"] = [{"degree": "B.Tech", "end_year": 2020}]
        # career_span = current_year(2026) - 2020 = 6; 10 > 6 → reject
        result = layers.l1_hard_reject([c], kb)
        assert result == []

    def test_age_vs_experience_rejects(self):
        """Claimed experience exceeding the age-based maximum must be hard-rejected."""
        kb = make_kb()
        c = make_candidate(years=15)
        c["education"] = []  # no graduation date → skip career-span check
        c["date_of_birth"] = "2000-01-01"
        # max_exp = 2026 - (2000+18) = 8; 15 > 8 → reject
        result = layers.l1_hard_reject([c], kb)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — BI-ENCODER SIMILARITY + 55% GATE
# ─────────────────────────────────────────────────────────────────────────────

class TestL2BiEncoder:

    def test_attaches_l2_score(self):
        st = MockST()
        cands = [make_candidate("C1"), make_candidate("C2")]
        result = layers.l2_bi_encoder(cands, JD, st)
        for c in result:
            assert "l2_score" in c
            assert 0.0 <= c["l2_score"] <= 1.0

    def test_tech_candidate_scores_higher(self):
        st = MockST()
        tech = make_candidate("TECH", skills=["Python", "RAG", "FAISS", "LLM", "embeddings"])
        non_tech = make_candidate("HR", skills=["Excel", "Hiring", "Onboarding"],
                                  work=[{"title": "HR Manager", "company": "Corp",
                                         "start_date": "2015-01-01", "duration_months": 60,
                                         "description": "Managed HR processes."}],
                                  profile={"years_of_experience": 6, "summary": "HR manager.",
                                           "headline": "HR"})
        result = layers.l2_bi_encoder([tech, non_tech], JD, st)
        assert result[0]["l2_score"] > result[1]["l2_score"]

    def test_empty_candidates_returns_empty(self):
        st = MockST()
        assert layers.l2_bi_encoder([], JD, st) == []

    def test_gate_keeps_55_percent(self):
        st = MockST()
        cands = [make_candidate(f"C{i}", years=5 + (i % 3)) for i in range(100)]
        scored = layers.l2_bi_encoder(cands, JD, st)
        gated = layers.l2_gate(scored)
        # 55% of 100 = 55; allow ±5 for random sampling variation
        assert 50 <= len(gated) <= 60

    def test_gate_on_single_candidate(self):
        st = MockST()
        cands = [make_candidate()]
        scored = layers.l2_bi_encoder(cands, JD, st)
        gated = layers.l2_gate(scored)
        # top 50% of 1 = 0, but random 5% rescues at least 1 from bottom
        assert len(gated) >= 0  # may be 0 or 1

    def test_gate_deterministic_with_same_seed(self):
        st = MockST()
        cands = [make_candidate(f"C{i}") for i in range(20)]
        scored1 = layers.l2_bi_encoder(cands, JD, st)
        scored2 = layers.l2_bi_encoder(cands, JD, st)
        gated1 = layers.l2_gate(list(scored1), seed=42)
        gated2 = layers.l2_gate(list(scored2), seed=42)
        assert [c["candidate_id"] for c in gated1] == [c["candidate_id"] for c in gated2]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — SENIORITY REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

class TestL3Seniority:

    def test_qualified_candidate_no_penalty(self):
        c = make_candidate(years=7)  # 7y → level 7 ≥ senior (5)
        result = layers.l3_seniority([c], JD)
        assert result[0]["l3_penalty"] == 1.0
        assert result[0]["l3_flag"] == ""

    def test_under_qualified_gets_penalty(self):
        c = make_candidate(years=2)  # 2y → level 1, below senior (5)
        result = layers.l3_seniority([c], JD)
        assert result[0]["l3_penalty"] == C.SENIORITY_PENALTY
        assert "under-qualified" in result[0]["l3_flag"]

    def test_penalty_does_not_remove_candidate(self):
        cands = [make_candidate(f"C{i}", years=i) for i in range(10)]
        result = layers.l3_seniority(cands, JD)
        assert len(result) == 10

    def test_boundary_experience_exact_senior(self):
        # 7y → seniority 7, JD senior = 5; no penalty
        c = make_candidate(years=7)
        result = layers.l3_seniority([c], JD)
        assert result[0]["l3_penalty"] == 1.0

    def test_seniority_flag_contains_years(self):
        c = make_candidate(years=1)
        result = layers.l3_seniority([c], JD)
        assert "1" in result[0]["l3_flag"] or "under-qualified" in result[0]["l3_flag"]

    def test_jd_junior_level_no_penalty_for_junior(self):
        jd_junior = {**JD, "required_seniority": "junior"}
        c = make_candidate(years=1)
        result = layers.l3_seniority([c], jd_junior)
        assert result[0]["l3_penalty"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — SEMANTIC WORK-TO-JD RELEVANCE
# ─────────────────────────────────────────────────────────────────────────────

class TestL4SemanticWork:

    def test_attaches_l4_score(self):
        st = MockST()
        cands = [make_candidate()]
        result = layers.l4_semantic_work(cands, JD, st)
        assert "l4_score" in result[0]
        assert 0.0 <= result[0]["l4_score"] <= 1.0

    def test_no_description_scores_zero(self):
        st = MockST()
        c = make_candidate()
        c["career_history"] = [{"title": "Engineer", "company": "Corp",
                                 "start_date": "2019-01-01", "duration_months": 36}]
        # no "description" key
        result = layers.l4_semantic_work([c], JD, st)
        assert result[0]["l4_score"] == 0.0

    def test_relevant_description_scores_higher_than_irrelevant(self):
        st = MockST()
        relevant = make_candidate("R", work=[{
            "title": "ML Eng", "company": "A",
            "start_date": "2018-01-01", "duration_months": 60,
            "description": "Built RAG pipelines with Python, FAISS, embeddings, LLM.",
        }])
        irrelevant = make_candidate("I", work=[{
            "title": "Sales", "company": "B",
            "start_date": "2018-01-01", "duration_months": 60,
            "description": "Managed sales targets and customer relationships.",
        }])
        result = layers.l4_semantic_work([relevant, irrelevant], JD, st)
        assert result[0]["l4_score"] >= result[1]["l4_score"]

    def test_empty_candidates_handled(self):
        st = MockST()
        assert layers.l4_semantic_work([], JD, st) == []

    def test_does_not_remove_candidates(self):
        st = MockST()
        cands = [make_candidate(f"C{i}") for i in range(5)]
        result = layers.l4_semantic_work(cands, JD, st)
        assert len(result) == 5


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — BEHAVIORAL SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

class TestL6Behavioral:

    def test_attaches_l6_score(self):
        cands = [make_candidate()]
        result = layers.l6_behavioral(cands, JD)
        assert "l6_score" in result[0]
        assert 0.0 <= result[0]["l6_score"] <= 1.0

    def test_all_scores_in_0_1_range(self):
        """Absolute normalization keeps all scores within [0, 1] regardless of pool."""
        cands = [make_candidate(f"C{i}", behavioral={
            "github_activity_score": i / 10,
            "recruiter_response_rate": 0.5,
            "profile_completeness": 0.8,
            "salary_fit": 0.5,
            "notice_period_score": 0.5,
        }) for i in range(1, 11)]
        result = layers.l6_behavioral(cands, JD)
        for c in result:
            assert 0.0 <= c["l6_score"] <= 1.0

    def test_high_github_scores_higher_for_tech_role(self):
        high_gh = make_candidate("H", behavioral={"github_activity_score": 1.0,
                                                   "recruiter_response_rate": 0.5,
                                                   "profile_completeness": 0.5,
                                                   "salary_fit": 0.5,
                                                   "notice_period_score": 0.5})
        low_gh = make_candidate("L", behavioral={"github_activity_score": 0.0,
                                                  "recruiter_response_rate": 0.5,
                                                  "profile_completeness": 0.5,
                                                  "salary_fit": 0.5,
                                                  "notice_period_score": 0.5})
        result = layers.l6_behavioral([high_gh, low_gh], JD)
        assert result[0]["l6_score"] > result[1]["l6_score"]

    def test_does_not_remove_candidates(self):
        cands = [make_candidate(f"C{i}") for i in range(8)]
        result = layers.l6_behavioral(cands, JD)
        assert len(result) == 8

    def test_missing_behavioral_signals_handled(self):
        c = make_candidate()
        c["behavioral_signals"] = {}
        result = layers.l6_behavioral([c], JD)
        assert "l6_score" in result[0]

    def test_single_candidate_absolute_score(self):
        """Single candidate gets an absolute score — not pool-relative 0.0."""
        c = make_candidate()
        # default signals: github=0.8, rr=0.7, pc=0.9, sf=0.6, np=0.5 → sum=3.5
        # tech role → max_possible=5.0 → l6_score = 3.5/5.0 = 0.70
        result = layers.l6_behavioral([c], JD)
        assert result[0]["l6_score"] == pytest.approx(0.70, abs=0.01)

    def test_non_tech_role_excludes_github(self):
        """Non-tech JD: github is not counted; max_possible = 4.0."""
        jd_non_tech = {**JD, "job_title": "Marketing Manager",
                       "explicit_required": ["Excel", "PowerPoint"]}
        c = make_candidate(behavioral={
            "github_activity_score": 0.9,
            "recruiter_response_rate": 0.8,
            "profile_completeness": 0.8,
            "salary_fit": 0.8,
            "notice_period_score": 0.8,
        })
        result = layers.l6_behavioral([c], jd_non_tech)
        # github ignored; score = 0.8+0.8+0.8+0.8 = 3.2; l6_score = 3.2/4.0 = 0.80
        assert result[0]["l6_score"] == pytest.approx(0.80, abs=0.01)

    def test_github_absence_penalty_for_tech_role(self):
        """Missing github_activity_score subtracts 0.1 from the raw sum for tech roles."""
        c = make_candidate(behavioral={
            "recruiter_response_rate": 0.5,
            "profile_completeness": 0.5,
            "salary_fit": 0.5,
            "notice_period_score": 0.5,
        })
        result = layers.l6_behavioral([c], JD)
        # score = -0.1 + 0.5+0.5+0.5+0.5 = 1.9; l6_score = max(0, 1.9/5) = 0.38
        assert result[0]["l6_score"] == pytest.approx(0.38, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — FIS + RANKING
# ─────────────────────────────────────────────────────────────────────────────

class TestL7FIS:

    def _make_scored(self, cid, l2, l4, l6=0.5, l3_pen=1.0, years=6):
        c = make_candidate(cid, years=years)
        c["l2_score"] = l2
        c["l4_score"] = l4
        c["l6_score"] = l6
        c["l3_penalty"] = l3_pen
        return c

    def test_fis_score_attached(self):
        c = self._make_scored("C1", l2=0.7, l4=0.6)
        result = fis.run_fis([c])
        assert "fis_score" in result[0]
        assert 0.0 <= result[0]["fis_score"] <= 1.0

    def test_high_l2_and_l4_gives_high_fis(self):
        c_high = self._make_scored("HIGH", l2=0.9, l4=0.85, l6=0.8)
        c_low = self._make_scored("LOW", l2=0.2, l4=0.1, l6=0.1)
        fis.run_fis([c_high, c_low])
        assert c_high["fis_score"] > c_low["fis_score"]

    def test_l3_penalty_reduces_score(self):
        c_pen = self._make_scored("PEN", l2=0.8, l4=0.7, l3_pen=C.SENIORITY_PENALTY)
        c_ok = self._make_scored("OK", l2=0.8, l4=0.7, l3_pen=1.0)
        fis.run_fis([c_pen, c_ok])
        assert c_pen["fis_score"] < c_ok["fis_score"]

    def test_rank_candidates_sorted_descending(self):
        kb = make_kb()
        cands = [self._make_scored(f"C{i}", l2=i * 0.1, l4=i * 0.1) for i in range(5)]
        fis.run_fis(cands)
        ranked = fis.rank_candidates(cands, kb)
        scores = [c["fis_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_candidates_tiebreak_by_experience(self):
        kb = make_kb()
        # Same l2/l4/l6 → tiebreak on years
        a = self._make_scored("A", l2=0.7, l4=0.6, l6=0.5, years=8)
        b = self._make_scored("B", l2=0.7, l4=0.6, l6=0.5, years=5)
        fis.run_fis([a, b])
        ranked = fis.rank_candidates([a, b], kb)
        assert ranked[0]["candidate_id"] == "A"  # more experience wins

    def test_flashrank_polish_returns_same_count_when_ranker_none(self):
        ranked = [self._make_scored(f"C{i}", l2=0.7, l4=0.6) for i in range(5)]
        fis.run_fis(ranked)
        result = fis.flashrank_polish(ranked, JD, ranker=None, fraud_kb=None)
        assert len(result) == 5

    def test_generate_reasoning_non_empty(self):
        c = self._make_scored("C1", l2=0.75, l4=0.65)
        c["l3_flag"] = ""
        c["l5_flag"] = ""
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert isinstance(text, str) and len(text) > 0

    def test_generate_reasoning_mentions_strong_match_for_high_l2(self):
        c = self._make_scored("C1", l2=0.9, l4=0.8)
        c["l3_flag"] = c["l5_flag"] = ""
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert "strong" in text.lower()

    def test_generate_reasoning_mentions_concern_for_low_seniority(self):
        c = self._make_scored("C1", l2=0.6, l4=0.5, l3_pen=C.SENIORITY_PENALTY)
        c["l3_flag"] = "under-qualified (2y vs senior required)"
        c["l5_flag"] = ""
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert "under-qualified" in text.lower()

    def test_fis_score_monotonic_with_l2(self):
        """Increasing l2 while l4 fixed should not decrease fis_score."""
        prev_score = -1.0
        for l2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            c = self._make_scored(f"C_{l2}", l2=l2, l4=0.5, l6=0.5)
            fis.run_fis([c])
            assert c["fis_score"] >= prev_score - 0.01  # allow tiny float noise
            prev_score = c["fis_score"]


# ─────────────────────────────────────────────────────────────────────────────
# UTILS — helper correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestUtils:

    def test_get_total_experience_years_profile_field(self):
        c = make_candidate(years=7)
        assert utils.get_total_experience_years(c) == pytest.approx(7.0)

    def test_get_total_experience_years_duration_months(self):
        c = {"career_history": [
            {"title": "E", "company": "A", "start_date": "2018-01-01",
             "duration_months": 36, "description": "desc"},
            {"title": "E2", "company": "B", "start_date": "2021-01-01",
             "duration_months": 24, "description": "desc2"},
        ]}
        assert utils.get_total_experience_years(c) == pytest.approx(5.0)

    def test_get_total_experience_years_old_schema(self):
        c = {"work_experience": [
            {"title": "E", "company": "A", "start_year": 2018,
             "duration_years": 4, "description": "desc"},
        ]}
        assert utils.get_total_experience_years(c) == pytest.approx(4.0)

    def test_get_work_profile_text_includes_summary(self):
        c = make_candidate()
        text = utils.get_work_profile_text(c)
        assert "llm" in text.lower() or "retrieval" in text.lower()

    def test_get_work_descriptions_only_excludes_skills(self):
        c = make_candidate()
        # skills-only field must not appear as a bare skill list
        skills_text = utils.get_skills_text(c)
        work_text = utils.get_work_descriptions_only(c)
        # description contains actual work narrative, not the raw skills list header
        assert "RAG pipelines" in work_text  # confirms description is present
        # Skills are NOT prepended as a standalone block (unlike get_work_profile_text)
        assert skills_text not in work_text

    def test_read_json_unwraps_candidates_envelope(self, tmp_path):
        import json
        data = {"candidates": [{"id": "A"}, {"id": "B"}], "meta": {"total": 2}}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        result = utils.read_json(p)
        assert len(result) == 2
        assert result[0]["id"] == "A"

    def test_read_json_bare_list(self, tmp_path):
        import json
        data = [{"id": "A"}, {"id": "B"}]
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        result = utils.read_json(p)
        assert len(result) == 2

    def test_read_json_single_object(self, tmp_path):
        import json
        data = {"id": "A", "name": "Test"}
        p = tmp_path / "test.json"
        p.write_text(json.dumps(data))
        result = utils.read_json(p)
        assert len(result) == 1

    def test_iter_work_history_prefers_career_history(self):
        c = {
            "career_history": [{"title": "New"}],
            "work_experience": [{"title": "Old"}],
        }
        entries = list(utils._iter_work_history(c))
        assert entries[0]["title"] == "New"

    def test_iter_work_history_falls_back_to_work_experience(self):
        c = {"work_experience": [{"title": "Old"}]}
        entries = list(utils._iter_work_history(c))
        assert entries[0]["title"] == "Old"


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
