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

from pipeline import layers, fis, constants as C
from pipeline import utils
from pipeline.dictionary import expand_skill


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
        c.setdefault("redrob_signals", {})["expected_salary_range_inr_lpa"] = {"min": 200000, "max": 100000}
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
# LAYER 1b — PROFILE INTEGRITY FLAGS
# ─────────────────────────────────────────────────────────────────────────────

class TestL1bProfileIntegrity:

    def test_clean_candidate_passes_no_penalty(self):
        c = make_candidate()
        result = layers.l1b_profile_integrity([c])
        assert len(result) == 1
        assert result[0]["l1b_penalty"] == 1.0
        assert result[0]["l1b_flags"] == []
        assert result[0]["l1b_status"] == "pass"

    def test_hard_reject_reverse_degree_order(self):
        c = make_candidate()
        c["reverse_degree_order"] = True
        result = layers.l1b_profile_integrity([c])
        assert result == []

    def test_active_before_signup_passes_l1b(self):
        # active_before_signup is a soft-penalty signal handled by L3 condition f,
        # NOT a hard reject at L1b.
        c = make_candidate()
        c["active_before_signup"] = True
        result = layers.l1b_profile_integrity([c])
        assert len(result) == 1
        assert result[0]["l1b_status"] == "pass"

    def test_hard_reject_invalid_degree_field_combination(self):
        # invalid_degree_field_combination lives per education entry (not top-level).
        c = make_candidate()
        c["education"] = [{"degree": "MBA", "end_year": 2019,
                           "invalid_degree_field_combination": True}]
        result = layers.l1b_profile_integrity([c])
        assert result == []

    def test_hard_reject_all_descriptions_identical(self):
        c = make_candidate()
        c["all_descriptions_identical"] = True
        result = layers.l1b_profile_integrity([c])
        assert result == []

    # Soft-penalty flags are NO LONGER applied at L1b.
    # They pass through with penalty=1.0 and are stored as L2 table cols,
    # then unified into condition 'h' at the L3 FIS layer.

    def test_soft_flags_pass_through_with_no_penalty(self):
        """Soft-penalty flags pass L1b with penalty=1.0."""
        soft_flags = [
            "skill_career_domain_mismatch",
            "education_overlap",
            "second_undergrad_after_first",
            "education_career_gap_flag",
        ]
        for flag in soft_flags:
            c = make_candidate()
            c[flag] = True
            result = layers.l1b_profile_integrity([c])
            assert len(result) == 1, f"Expected pass for {flag}"
            assert result[0]["l1b_penalty"] == 1.0, f"Expected no penalty for {flag}"
            assert result[0]["l1b_flags"] == [], f"Expected empty flags for {flag}"
            assert result[0]["l1b_status"] == "pass", f"Expected status=pass for {flag}"

    def test_multiple_soft_flags_all_pass(self):
        """Multiple soft flags together still yield penalty=1.0 at L1b."""
        c = make_candidate()
        c["skill_career_domain_mismatch"] = True
        c["education_overlap"] = True
        result = layers.l1b_profile_integrity([c])
        assert len(result) == 1
        assert result[0]["l1b_penalty"] == 1.0
        assert result[0]["l1b_flags"] == []

    def test_false_flags_are_ignored(self):
        c = make_candidate()
        c["reverse_degree_order"] = False
        c["skill_career_domain_mismatch"] = False
        result = layers.l1b_profile_integrity([c])
        assert len(result) == 1
        assert result[0]["l1b_penalty"] == 1.0

    def test_hard_reject_ignores_soft_flags(self):
        """Hard reject fires even when other soft flags are also present."""
        c = make_candidate()
        c["reverse_degree_order"] = True      # hard reject
        c["education_career_gap_flag"] = True  # soft — irrelevant
        result = layers.l1b_profile_integrity([c])
        assert result == []

    def test_multiple_candidates_mixed(self):
        good = make_candidate("G")
        bad = make_candidate("B")
        bad["reverse_degree_order"] = True
        flagged = make_candidate("F")          # soft flag — passes with penalty=1.0
        flagged["education_overlap"] = True
        result = layers.l1b_profile_integrity([good, bad, flagged])
        assert len(result) == 2
        ids = {c["candidate_id"] for c in result}
        assert "B" not in ids
        assert all(c["l1b_penalty"] == 1.0 for c in result)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1c — SKILL MATCH (NLP STRING + SYNONYM MATCH)
# ─────────────────────────────────────────────────────────────────────────────

class TestL1cSkillMatch:

    def test_attaches_score_and_lists(self):
        c = make_candidate()
        result = layers.l1c_skill_match([c], JD)
        assert "l1c_score" in result[0]
        assert "l1c_matched_required" in result[0]
        assert "l1c_missing_required" in result[0]
        assert "l1c_matched_bonus" in result[0]
        assert 0.0 <= result[0]["l1c_score"] <= 1.0

    def test_tech_candidate_scores_high(self):
        c = make_candidate(skills=["Python", "RAG", "LLM", "FAISS", "SQL"])
        result = layers.l1c_skill_match([c], JD)
        assert result[0]["l1c_score"] > 0.5

    def test_unrelated_candidate_scores_low(self):
        c = make_candidate(
            skills=["Excel", "PowerPoint", "Word"],
            work=[{"title": "Accountant", "company": "Corp",
                   "start_date": "2018-01-01", "duration_months": 60,
                   "description": "Managed accounts and ledger reconciliation."}],
            profile={"years_of_experience": 6, "summary": "Finance professional.", "headline": "Accountant"},
        )
        result = layers.l1c_skill_match([c], JD)
        assert result[0]["l1c_score"] < 0.3

    def test_tech_scores_higher_than_unrelated(self):
        tech = make_candidate("T", skills=["Python", "RAG", "LLM"])
        non_tech = make_candidate("N", skills=["Accounts", "Tally", "Excel"],
                                  work=[{"title": "Accountant", "company": "Corp",
                                         "start_date": "2018-01-01", "duration_months": 60,
                                         "description": "Bookkeeping and tax filing."}],
                                  profile={"years_of_experience": 6, "summary": "Finance.",
                                           "headline": "Accountant"})
        result = layers.l1c_skill_match([tech, non_tech], JD)
        assert result[0]["l1c_score"] > result[1]["l1c_score"]

    def test_synonym_llm_matches_large_language_model(self):
        jd_ml = {**JD, "explicit_required": ["large language model"], "inferred_required": []}
        c = make_candidate(skills=["LLM", "Python"])
        result = layers.l1c_skill_match([c], jd_ml)
        assert "large language model" in result[0]["l1c_matched_required"]

    def test_synonym_ml_matches_machine_learning(self):
        jd_ml = {**JD, "explicit_required": ["machine learning", "Python"], "inferred_required": []}
        c = make_candidate(skills=["ML", "Python"],
                           work=[{"title": "ML Eng", "company": "A",
                                  "start_date": "2020-01-01", "duration_months": 36,
                                  "description": "Built ML models and pipelines."}])
        result = layers.l1c_skill_match([c], jd_ml)
        assert "machine learning" in result[0]["l1c_matched_required"]

    def test_skill_in_work_description_is_matched(self):
        c = make_candidate(
            skills=[],
            work=[{"title": "Engineer", "company": "Corp",
                   "start_date": "2019-01-01", "duration_months": 48,
                   "description": "Built RAG pipelines with FAISS and Python."}],
        )
        jd_simple = {**JD, "explicit_required": ["Python", "RAG", "FAISS"], "inferred_required": []}
        result = layers.l1c_skill_match([c], jd_simple)
        assert result[0]["l1c_score"] == pytest.approx(1.0, abs=0.01)

    def test_empty_jd_skills_gives_full_score(self):
        c = make_candidate()
        jd_empty = {**JD, "explicit_required": [], "inferred_required": [],
                    "explicit_bonus": [], "inferred_bonus": []}
        result = layers.l1c_skill_match([c], jd_empty)
        assert result[0]["l1c_score"] == 1.0

    def test_missing_skills_listed_correctly(self):
        jd_strict = {**JD, "explicit_required": ["Python", "Kubernetes"], "inferred_required": []}
        c = make_candidate(skills=["Python"],
                           work=[{"title": "Dev", "company": "X",
                                  "start_date": "2020-01-01", "duration_months": 36,
                                  "description": "Backend Python development."}])
        result = layers.l1c_skill_match([c], jd_strict)
        assert "Kubernetes" in result[0]["l1c_missing_required"]
        assert "Python" in result[0]["l1c_matched_required"]

    def test_does_not_remove_candidates_when_no_gate(self):
        old_val = C.L1C_MIN_SKILL_MATCH
        C.L1C_MIN_SKILL_MATCH = 0.0
        cands = [make_candidate(f"C{i}") for i in range(5)]
        result = layers.l1c_skill_match(cands, JD)
        assert len(result) == 5
        C.L1C_MIN_SKILL_MATCH = old_val

    def test_gate_filters_low_scoring_candidates(self):
        old_val = C.L1C_MIN_SKILL_MATCH
        C.L1C_MIN_SKILL_MATCH = 0.5
        tech = make_candidate("T", skills=["Python", "RAG", "LLM", "FAISS"])
        non_tech = make_candidate("N", skills=["Tally", "GST", "Accounting"],
                                  work=[{"title": "Accountant", "company": "Corp",
                                         "start_date": "2018-01-01", "duration_months": 60,
                                         "description": "Financial statements and audit."}],
                                  profile={"years_of_experience": 6, "summary": "Finance.",
                                           "headline": "Accountant"})
        result = layers.l1c_skill_match([tech, non_tech], JD)
        assert any(c["candidate_id"] == "T" for c in result)
        assert not any(c["candidate_id"] == "N" for c in result)
        C.L1C_MIN_SKILL_MATCH = old_val

    def test_score_is_between_zero_and_one(self):
        cands = [make_candidate(f"C{i}") for i in range(10)]
        result = layers.l1c_skill_match(cands, JD)
        for c in result:
            assert 0.0 <= c["l1c_score"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# DICTIONARY — expand_skill utility
# ─────────────────────────────────────────────────────────────────────────────

class TestL1cGate:
    """Tests for the global 65% gate applied after per-folder L1c scoring."""

    def _scored(self, cid: str, score: float):
        c = make_candidate(cid)
        c["l1c_score"] = score
        return c

    def test_keeps_approximately_65_percent(self):
        cands = [self._scored(f"C{i}", i / 100.0) for i in range(100)]
        gated = layers.l1c_gate(cands)
        # top 50 + random ceil(100*0.15)=15 = 65
        assert 60 <= len(gated) <= 70

    def test_top_50_percent_always_included(self):
        cands = [self._scored(f"C{i}", i / 100.0) for i in range(100)]
        gated = layers.l1c_gate(cands)
        gated_ids = {c["candidate_id"] for c in gated}
        # C50–C99 are the top 50 by score
        for i in range(50, 100):
            assert f"C{i}" in gated_ids

    def test_empty_returns_empty(self):
        assert layers.l1c_gate([]) == []

    def test_single_candidate_passes_through(self):
        c = self._scored("C1", 0.5)
        gated = layers.l1c_gate([c])
        assert len(gated) >= 1

    def test_deterministic_with_same_seed(self):
        cands = [self._scored(f"C{i}", i / 20.0) for i in range(20)]
        g1 = layers.l1c_gate(list(cands), seed=42)
        g2 = layers.l1c_gate(list(cands), seed=42)
        assert [c["candidate_id"] for c in g1] == [c["candidate_id"] for c in g2]

    def test_different_seeds_may_differ(self):
        cands = [self._scored(f"C{i}", i / 100.0) for i in range(100)]
        g1 = layers.l1c_gate(list(cands), seed=1)
        g2 = layers.l1c_gate(list(cands), seed=99)
        # The top-50 overlap is identical; the random-15 may differ
        ids1 = {c["candidate_id"] for c in g1}
        ids2 = {c["candidate_id"] for c in g2}
        assert ids1 != ids2  # nearly certain with 100 candidates

    def test_all_candidates_small_pool(self):
        # Pool of 3 → top ceil(1.5)=2 + random ceil(0.45)=1 = 3 (all pass)
        cands = [self._scored(f"C{i}", i / 3.0) for i in range(3)]
        gated = layers.l1c_gate(cands)
        assert len(gated) == 3


class TestDictionary:

    def test_canonical_term_expands_to_self_plus_synonyms(self):
        forms = expand_skill("python")
        assert "python" in forms
        assert "python3" in forms

    def test_synonym_resolves_to_canonical_and_all_forms(self):
        forms = expand_skill("LLM")
        assert "large language model" in forms
        assert "llm" in forms

    def test_ml_expands_to_machine_learning(self):
        forms = expand_skill("ML")
        assert "machine learning" in forms

    def test_unknown_term_returns_itself(self):
        forms = expand_skill("xyzfoobar")
        assert forms == ["xyzfoobar"]

    def test_empty_string_returns_empty(self):
        forms = expand_skill("")
        assert forms == []

    def test_case_insensitive_canonical_lookup(self):
        forms = expand_skill("Python")
        assert "python" in forms

    def test_figma_resolves_via_synonym(self):
        forms = expand_skill("figma")
        assert "ui design" in forms


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

    def _make_scored(self, cid, l1c, l4, l6=0.5, l3_pen=1.0, l1b_pen=1.0, years=6):
        c = make_candidate(cid, years=years)
        c["l1c_score"] = l1c
        c["l1c_matched_required"] = []
        c["l1c_missing_required"] = []
        c["l4_score"] = l4
        c["l6_score"] = l6
        c["l3_penalty"] = l3_pen
        c["l1b_penalty"] = l1b_pen
        return c

    def test_fis_score_attached(self):
        c = self._make_scored("C1", l1c=0.7, l4=0.6)
        result = fis.run_fis([c])
        assert "fis_score" in result[0]
        assert 0.0 <= result[0]["fis_score"] <= 1.0

    def test_high_l1c_and_l4_gives_high_fis(self):
        c_high = self._make_scored("HIGH", l1c=0.9, l4=0.85, l6=0.8)
        c_low = self._make_scored("LOW", l1c=0.2, l4=0.1, l6=0.1)
        fis.run_fis([c_high, c_low])
        assert c_high["fis_score"] > c_low["fis_score"]

    def test_l3_penalty_reduces_score(self):
        c_pen = self._make_scored("PEN", l1c=0.8, l4=0.7, l3_pen=C.SENIORITY_PENALTY)
        c_ok = self._make_scored("OK", l1c=0.8, l4=0.7, l3_pen=1.0)
        fis.run_fis([c_pen, c_ok])
        assert c_pen["fis_score"] < c_ok["fis_score"]

    def test_l1b_penalty_reduces_score(self):
        c_pen = self._make_scored("PEN", l1c=0.8, l4=0.7, l1b_pen=0.7)
        c_ok = self._make_scored("OK", l1c=0.8, l4=0.7, l1b_pen=1.0)
        fis.run_fis([c_pen, c_ok])
        assert c_pen["fis_score"] < c_ok["fis_score"]

    def test_rank_candidates_sorted_descending(self):
        kb = make_kb()
        cands = [self._make_scored(f"C{i}", l1c=i * 0.1, l4=i * 0.1) for i in range(5)]
        fis.run_fis(cands)
        ranked = fis.rank_candidates(cands, kb)
        scores = [c["fis_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_candidates_tiebreak_by_experience(self):
        kb = make_kb()
        a = self._make_scored("A", l1c=0.7, l4=0.6, l6=0.5, years=8)
        b = self._make_scored("B", l1c=0.7, l4=0.6, l6=0.5, years=5)
        fis.run_fis([a, b])
        ranked = fis.rank_candidates([a, b], kb)
        assert ranked[0]["candidate_id"] == "A"  # more experience wins

    def test_generate_reasoning_non_empty(self):
        c = self._make_scored("C1", l1c=0.75, l4=0.65)
        c["l3_flag"] = ""
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert isinstance(text, str) and len(text) > 0

    def test_generate_reasoning_mentions_strong_match_for_high_l1c(self):
        c = self._make_scored("C1", l1c=0.9, l4=0.8)
        c["l3_flag"] = ""
        c["l1c_matched_required"] = ["Python", "RAG", "LLM", "FAISS", "embeddings", "SQL"]
        c["l1c_missing_required"] = []
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert "strong" in text.lower()

    def test_generate_reasoning_mentions_concern_for_low_seniority(self):
        c = self._make_scored("C1", l1c=0.6, l4=0.5, l3_pen=C.SENIORITY_PENALTY)
        c["l3_flag"] = "under-qualified (2y vs senior required)"
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert "under-qualified" in text.lower()

    def test_generate_reasoning_mentions_missing_skills_for_low_l1c(self):
        c = self._make_scored("C1", l1c=0.1, l4=0.5)
        c["l3_flag"] = ""
        c["l1c_matched_required"] = ["Python"]
        c["l1c_missing_required"] = ["RAG", "LLM", "FAISS", "embeddings", "SQL", "REST APIs", "Git"]
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert "missing" in text.lower() or "weak" in text.lower()

    def test_fis_score_monotonic_with_l1c(self):
        """Increasing l1c while l4 fixed should not decrease fis_score."""
        prev_score = -1.0
        for l1c in [0.1, 0.3, 0.5, 0.7, 0.9]:
            c = self._make_scored(f"C_{l1c}", l1c=l1c, l4=0.5, l6=0.5)
            fis.run_fis([c])
            assert c["fis_score"] >= prev_score - 0.01  # allow tiny float noise
            prev_score = c["fis_score"]

    # ── Tier tests ────────────────────────────────────────────────────────────

    def test_very_good_tier_assigned(self):
        """Candidate meeting all excellence criteria gets l7_tier='very_good'."""
        c = self._make_scored("VG", l1c=0.9, l4=0.85, l6=0.75, years=8)
        # l1c≥0.65 ✓  l4≥0.60 ✓  l6≥0.60 ✓  exp≥7 ✓  no flags ✓  l3_pen=1.0 ✓
        fis.run_fis([c])
        assert c["l7_tier"] == "very_good"

    def test_eligible_tier_when_below_thresholds(self):
        """A clean candidate not meeting very_good thresholds is 'eligible'."""
        c = self._make_scored("EL", l1c=0.5, l4=0.4, l6=0.4, years=4)
        fis.run_fis([c])
        assert c["l7_tier"] == "eligible"

    def test_penalized_tier_when_l3_penalty(self):
        """L3 penalty → penalized tier regardless of other scores."""
        c = self._make_scored("PEN", l1c=0.9, l4=0.9, l3_pen=C.SENIORITY_PENALTY)
        fis.run_fis([c])
        assert c["l7_tier"] == "penalized"

    def test_penalized_tier_when_l1b_flags(self):
        """L1b flags → penalized tier regardless of other scores."""
        c = self._make_scored("PEN", l1c=0.9, l4=0.9)
        c["l1b_flags"] = ["duplicate_job_descriptions"]
        fis.run_fis([c])
        assert c["l7_tier"] == "penalized"

    def test_not_very_good_if_exp_low(self):
        """Low experience (< L7_VERY_GOOD_EXP_MIN) keeps candidate 'eligible'."""
        c = self._make_scored("LOW_EXP", l1c=0.9, l4=0.85, l6=0.75, years=4)
        fis.run_fis([c])
        assert c["l7_tier"] == "eligible"

    def test_not_very_good_if_l6_low(self):
        """L6 below threshold keeps candidate 'eligible'."""
        c = self._make_scored("LOW_L6", l1c=0.9, l4=0.85, l6=0.3, years=8)
        fis.run_fis([c])
        assert c["l7_tier"] == "eligible"

    def test_penalized_ranks_after_eligible_regardless_of_score(self):
        """A penalized candidate with higher FIS score must still rank after eligible."""
        kb = make_kb()
        penalized = self._make_scored("PEN", l1c=0.95, l4=0.95, l3_pen=C.SENIORITY_PENALTY)
        eligible = self._make_scored("ELG", l1c=0.4, l4=0.4)
        fis.run_fis([penalized, eligible])
        ranked = fis.rank_candidates([penalized, eligible], kb)
        assert ranked[0]["candidate_id"] == "ELG"
        assert ranked[1]["candidate_id"] == "PEN"

    def test_very_good_ranks_before_eligible_regardless_of_score(self):
        """A very_good candidate must rank above any eligible candidate."""
        kb = make_kb()
        eligible = self._make_scored("ELG", l1c=0.95, l4=0.95, l6=0.5, years=4)
        vg = self._make_scored("VG", l1c=0.7, l4=0.65, l6=0.7, years=8)
        fis.run_fis([eligible, vg])
        ranked = fis.rank_candidates([eligible, vg], kb)
        assert ranked[0]["candidate_id"] == "VG"

    def test_penalized_never_in_top_100_if_100_clean_exist(self):
        """Penalized candidates fill slots > 100 when 100+ clean candidates exist."""
        kb = make_kb()
        clean = [self._make_scored(f"CLEAN_{i}", l1c=0.6, l4=0.5) for i in range(105)]
        dirty = [self._make_scored(f"DIRTY_{i}", l1c=0.9, l4=0.9,
                                   l3_pen=C.SENIORITY_PENALTY) for i in range(5)]
        fis.run_fis(clean + dirty)
        ranked = fis.rank_candidates(clean + dirty, kb)
        top100_ids = {c["candidate_id"] for c in ranked[:100]}
        for i in range(5):
            assert f"DIRTY_{i}" not in top100_ids

    def test_tier_logged_in_fis_score_breakdown(self):
        """l7_tier field must be present on every candidate after run_fis."""
        cands = [self._make_scored(f"C{i}", l1c=0.5, l4=0.5) for i in range(10)]
        fis.run_fis(cands)
        for c in cands:
            assert "l7_tier" in c
            assert c["l7_tier"] in ("very_good", "eligible", "penalized")

    def test_generate_reasoning_mentions_exceptional_for_very_good(self):
        c = self._make_scored("VG", l1c=0.9, l4=0.85, l6=0.75, years=8)
        c["l3_flag"] = ""
        c["l1c_matched_required"] = ["Python", "RAG", "LLM"]
        c["l1c_missing_required"] = []
        fis.run_fis([c])
        text = fis.generate_reasoning(c, JD)
        assert "exceptional" in text.lower()


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
