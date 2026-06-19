"""
tests/modular_test.py — Unit tests for each pipeline layer in isolation.

Covers: JD parser wiring, L2 bi-encoder + gate, L3 seniority penalty,
L4 semantic work relevance, L5 project sanity, L6 behavioral signals,
heuristic folder pruning, and the L7 FIS + ranking step.

Run:  python -m pytest tests/modular_test.py -v
  or: python tests/modular_test.py
"""

import sys
import sqlite3
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import layers, fis, pruning, constants as C


# ── fixtures ──────────────────────────────────────────────────────────────────
class MockST:
    """Deterministic fake sentence transformer for offline layer tests."""
    dim = 64

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        out = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % (2**32))
            v = rng.rand(self.dim).astype(np.float32)
            for kw in ["python", "rag", "llm", "faiss", "ml", "docker"]:
                if kw in t.lower():
                    v[hash(kw) % self.dim] += 2.0
            v = v / (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.array(out)


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


JD = {
    "job_title": "Senior AI Engineer",
    "explicit_required": ["Python", "RAG", "LLM", "FAISS"],
    "inferred_required": ["semantic search", "Docker"],
    "explicit_bonus": ["Kubernetes"], "inferred_bonus": ["CI/CD"],
    "semantic_neighbors": {}, "required_seniority": "senior",
}


# ── JD parser wiring (mocked — no real Ollama call) ───────────────────────────
def test_jd_parser_validate_and_fill():
    from src import jd_parser
    partial = {"explicit_required": ["Python", "python", "  RAG  "]}
    filled = jd_parser.validate_and_fill(partial)
    for f in jd_parser.JD_SCHEMA_FIELDS:
        assert f in filled, f"missing schema field: {f}"
    assert len(filled["explicit_required"]) == 2, filled["explicit_required"]
    assert filled["semantic_neighbors"] == {}
    assert "job_title" in filled and "required_seniority" in filled
    print("  ✅ jd_parser.validate_and_fill: schema complete, dedupe correct")


def test_jd_parser_prompt_has_intent_fields():
    from src import jd_parser
    p = jd_parser.PROMPT_TEMPLATE
    for field in ["explicit_required", "inferred_required", "explicit_bonus",
                  "inferred_bonus", "semantic_neighbors"]:
        assert field in p, f"PROMPT_TEMPLATE missing intent field: {field}"
    print("  ✅ jd_parser prompt covers all 5 intent fields")


# ── L2: bi-encoder + gate ─────────────────────────────────────────────────────
def test_l2_scores_and_gate():
    st = MockST()
    cands = [good_candidate(f"C{i}", years=5) for i in range(20)]
    cands = layers.l2_bi_encoder(cands, JD, st)
    assert all("l2_score" in c for c in cands), "l2_score key missing after l2_bi_encoder"
    assert all(0.0 <= c["l2_score"] <= 1.0 for c in cands), "l2_score out of [0, 1]"
    gated = layers.l2_gate(cands, seed=42)
    assert 9 <= len(gated) <= 13, f"55% gate kept {len(gated)} of 20 (expected 9–13)"
    print(f"  ✅ L2: scored + gated ({len(cands)}→{len(gated)})")


def test_l2_relevant_candidate_scores_higher():
    """Candidate matching JD keywords must outscore one with no relevant skills."""
    st = MockST()
    relevant = good_candidate("REL", years=5)
    irrelevant = good_candidate("IRREL", years=5)
    irrelevant["skills"] = ["Accounting", "Marketing"]
    irrelevant["work_experience"] = [{"title": "Accountant", "company": "realtech",
                                      "start_year": 2016, "duration_years": 5,
                                      "description": "managed spreadsheets and annual budgets"}]
    out = layers.l2_bi_encoder([relevant, irrelevant], JD, st)
    scores = {c["candidate_id"]: c["l2_score"] for c in out}
    assert scores["REL"] >= scores["IRREL"], (
        f"relevant ({scores['REL']:.3f}) should score >= irrelevant ({scores['IRREL']:.3f})"
    )
    print(f"  ✅ L2: relevant ({scores['REL']:.3f}) >= irrelevant ({scores['IRREL']:.3f})")


# ── L3: seniority penalty ─────────────────────────────────────────────────────
def test_l3_seniority_penalty():
    junior = good_candidate("JR", years=1)
    senior = good_candidate("SR", years=8)
    out = layers.l3_seniority([junior, senior], JD)
    jr = next(c for c in out if c["candidate_id"] == "JR")
    sr = next(c for c in out if c["candidate_id"] == "SR")
    assert jr["l3_penalty"] < 1.0, "under-qualified junior should be penalized"
    assert sr["l3_penalty"] == 1.0, "qualified senior should not be penalized"
    print("  ✅ L3: junior penalized, senior unaffected")


def test_l3_penalty_always_in_range():
    """l3_penalty must be in (0, 1] for any experience level."""
    for years in [0, 1, 3, 6, 10, 20]:
        c = good_candidate(f"EXP_{years}", years=years)
        out = layers.l3_seniority([c], JD)
        p = out[0]["l3_penalty"]
        assert 0 < p <= 1.0, f"l3_penalty={p} out of (0, 1] for {years} yrs experience"
    print("  ✅ L3: penalty in (0, 1] across 0–20 years experience")


# ── L4: semantic work relevance ───────────────────────────────────────────────
def test_l4_independent_score():
    st = MockST()
    cands = [good_candidate("C1"),
             {"candidate_id": "EMPTY", "skills": [], "work_experience": [], "projects": []}]
    out = layers.l4_semantic_work(cands, JD, st)
    empty = next(c for c in out if c["candidate_id"] == "EMPTY")
    assert empty["l4_score"] == 0.0, "no-work candidate should score 0 (not be penalized)"
    assert all("l4_score" in c for c in out), "l4_score key missing after l4_semantic_work"
    print("  ✅ L4: no-work candidate scores 0 (no spurious penalty)")


def test_l4_scores_in_range():
    """l4_score must be in [0, 1] for all candidates."""
    st = MockST()
    cands = [good_candidate(f"C{i}", years=3 + i) for i in range(10)]
    out = layers.l4_semantic_work(cands, JD, st)
    for c in out:
        assert 0.0 <= c["l4_score"] <= 1.0, \
            f"l4_score={c['l4_score']} out of [0, 1] for {c['candidate_id']}"
    print("  ✅ L4: all scores in [0, 1]")


# ── L6: behavioral signals ────────────────────────────────────────────────────
def test_l6_behavioral_normalized():
    cands = [good_candidate(f"C{i}") for i in range(5)]
    for i, c in enumerate(cands):
        c["behavioral_signals"]["github_activity_score"] = i / 4.0
    out = layers.l6_behavioral(cands, JD)
    scores = [c["l6_score"] for c in out]
    assert all(0.0 <= s <= 1.0 for s in scores), "L6 scores must all be in [0, 1]"
    print("  ✅ L6: behavioral signals normalized to [0, 1]")


def test_l6_higher_activity_scores_higher():
    """Candidate with higher github activity should produce a higher or equal l6_score."""
    low = good_candidate("LOW_ACT")
    high = good_candidate("HIGH_ACT")
    low["behavioral_signals"]["github_activity_score"] = 0.0
    high["behavioral_signals"]["github_activity_score"] = 1.0
    out = layers.l6_behavioral([low, high], JD)
    scores = {c["candidate_id"]: c["l6_score"] for c in out}
    assert scores["HIGH_ACT"] >= scores["LOW_ACT"], (
        f"high activity ({scores['HIGH_ACT']:.3f}) should score >= low activity ({scores['LOW_ACT']:.3f})"
    )
    print(f"  ✅ L6: high activity ({scores['HIGH_ACT']:.3f}) >= low activity ({scores['LOW_ACT']:.3f})")


# ── folder pruning ────────────────────────────────────────────────────────────
def test_pruning_drops_irrelevant():
    folders = ["AI_Engineer", "ML_Engineer", "HR_Recruiter", "Marketing"]
    kept = pruning.prune_folders(folders, JD)
    kept_names = [k for k, _ in kept]
    assert "AI_Engineer" in kept_names, "relevant folder AI_Engineer was dropped"
    assert "Marketing" not in kept_names or len(kept_names) <= 2, \
        "irrelevant Marketing folder kept alongside relevant ones"
    print(f"  ✅ pruning: relevant folders kept {kept_names}")


def test_pruning_expands_swe_abbreviation():
    """SWE folder abbreviation (→ 'software engineer') must survive for an AI Engineer JD."""
    folders = ["SWE", "Finance", "Legal"]
    kept = pruning.prune_folders(folders, JD)
    kept_names = [k for k, _ in kept]
    assert "SWE" in kept_names, \
        f"SWE (expands to 'software engineer') dropped; kept={kept_names}"
    print(f"  ✅ pruning: SWE abbreviation expanded and kept ({kept_names})")


# ── L7: FIS + ranking ─────────────────────────────────────────────────────────
def test_fis_and_ranking():
    st = MockST()
    kb = mock_kb()
    cands = [good_candidate(f"C{i}", years=3 + i) for i in range(6)]
    cands = layers.l2_bi_encoder(cands, JD, st)
    cands = layers.l3_seniority(cands, JD)
    cands = layers.l4_semantic_work(cands, JD, st)
    cands = layers.l6_behavioral(cands, JD)
    cands = fis.run_fis(cands)
    assert all("fis_score" in c for c in cands), "fis_score missing after run_fis"
    ranked = fis.rank_candidates(cands, kb)
    assert len(ranked) == len(cands), "rank_candidates dropped candidates unexpectedly"
    print(f"  ✅ FIS + ranking: {len(ranked)} ranked, top score {ranked[0]['fis_score']:.3f}")


def test_fis_scores_in_range():
    """fis_score must be in [0, 1] for all candidates."""
    st = MockST()
    cands = [good_candidate(f"C{i}", years=2 + i) for i in range(10)]
    cands = layers.l2_bi_encoder(cands, JD, st)
    cands = layers.l3_seniority(cands, JD)
    cands = layers.l4_semantic_work(cands, JD, st)
    cands = layers.l6_behavioral(cands, JD)
    cands = fis.run_fis(cands)
    for c in cands:
        assert 0.0 <= c["fis_score"] <= 1.0, \
            f"fis_score={c['fis_score']} out of [0, 1] for {c['candidate_id']}"
    print("  ✅ FIS: all fis_scores in [0, 1]")


def test_reasoning_is_specific():
    c = good_candidate("C1", years=7)
    c["l2_score"] = 0.7
    c["l4_score"] = 0.6
    c["l6_score"] = 0.8
    r = fis.generate_reasoning(c, JD)
    assert "7y" in r, f"reasoning should cite years of experience; got: '{r}'"
    assert len(r) > 20, f"reasoning too short (len={len(r)}): '{r}'"
    print(f"  ✅ reasoning cites experience: '{r[:60]}...'")


# ── runner ────────────────────────────────────────────────────────────────────
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} modular layer tests\n")
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
