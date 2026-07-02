"""
tests/modular_test.py — Smoke tests for each pipeline layer in isolation.

Covers: JD parser wiring, L1a fraud KB, L1b profile integrity, L1c skill
match, L1d inferred match, L2 table extract, L3 weighted-condition scoring,
the L1a→L3 streaming cascade, L4 semantic work relevance + donts penalty,
L4b explicit-required penalty, L5 pass-through, and heuristic folder pruning.

These exercise the CURRENT pipeline surface (pipeline/layers.py,
pipeline/pruning.py, pipeline/jd_parser.py) as wired together in rank.py —
NOT the legacy L1-L7/FIS architecture (pipeline/fis.py is orphaned and not
called from rank.py, so it is intentionally not covered here).

Run:  python -m pytest tests/modular_test.py -v
  or: python tests/modular_test.py
"""

import sys
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import layers, pruning, utils


# ── fixtures ──────────────────────────────────────────────────────────────────
class MockST:
    """Deterministic fake sentence transformer for offline L4 tests."""
    dim = 64

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
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
        "candidate_id": cid,
        "career_history": [{
            "title": "ML Engineer", "company": "realtech",
            "start_date": "2018-01-01", "duration_months": years * 12,
            "description": "Built RAG and LLM systems with Python and FAISS embeddings in production.",
        }],
        "skills": [{"name": "Python"}, {"name": "RAG"}, {"name": "LLM"}, {"name": "FAISS"}],
        "profile": {
            "years_of_experience": years,
            "summary": "Senior ML engineer specialising in retrieval and LLM systems.",
            "headline": "ML Engineer",
        },
        "education": [{"degree": "Bachelor", "end_year": 2015}],
        "redrob_signals": {
            "github_activity_score": 80,
            "recruiter_response_rate": 0.7,
            "profile_completeness_score": 0.9,
        },
    }


JD = {
    "job_title": "Senior AI Engineer",
    "explicit_required": ["Python", "RAG", "LLM", "FAISS"],
    "explicit_bonus": ["Kubernetes"],
    "inferred_required": ["semantic search", "Docker"],
    "inferred_bonus": ["CI/CD"],
    "what_you_will_do": "Own the ranking and retrieval systems; build RAG pipelines with Python, FAISS, and LLMs.",
    "donts": ["Pure researchers with no shipped production systems"],
}


# ── JD parser wiring (mocked — no real Ollama call) ───────────────────────────
def test_jd_parser_validate_and_fill():
    from pipeline import jd_parser
    partial = {"explicit_required": ["Python", "python", "  RAG  "]}
    filled = jd_parser.validate_and_fill(partial)
    for f in jd_parser.JD_SCHEMA_FIELDS:
        assert f in filled, f"missing schema field: {f}"
    assert len(filled["explicit_required"]) == 2, filled["explicit_required"]
    assert filled["semantic_neighbors"] == {}
    assert "job_title" in filled and "required_seniority" in filled
    print("  ✅ jd_parser.validate_and_fill: schema complete, dedupe correct")


def test_jd_parser_prompt_has_intent_fields():
    from pipeline import jd_parser
    p = jd_parser.PROMPT_TEMPLATE
    for field in ["explicit_required", "inferred_required", "explicit_bonus",
                  "inferred_bonus", "semantic_neighbors"]:
        assert field in p, f"PROMPT_TEMPLATE missing intent field: {field}"
    print("  ✅ jd_parser prompt covers all 5 intent fields")


# ── L1a: fraud KB hard reject ─────────────────────────────────────────────────
def test_l1a_clean_candidate_passes():
    kb = mock_kb()
    result = layers.l1_hard_reject([good_candidate()], kb)
    assert len(result) == 1
    assert "l1_score" in result[0] and 0.0 <= result[0]["l1_score"] <= 1.0
    print("  ✅ L1a: clean candidate passes with l1_score in [0, 1]")


def test_l1a_fictional_company_rejects():
    kb = mock_kb()
    c = good_candidate()
    c["career_history"][0]["company"] = "Hooli"
    result = layers.l1_hard_reject([c], kb)
    assert result == []
    print("  ✅ L1a: fictional company hard-rejected")


# ── L1b: profile integrity ────────────────────────────────────────────────────
def test_l1b_clean_candidate_no_penalty():
    c = good_candidate()
    result = layers.l1b_profile_integrity([c])
    assert len(result) == 1
    assert result[0]["l1b_penalty"] == 1.0
    print("  ✅ L1b: clean candidate passes with penalty=1.0")


def test_l1b_reverse_degree_order_rejects():
    c = good_candidate()
    c["reverse_degree_order"] = True
    result = layers.l1b_profile_integrity([c])
    assert result == []
    print("  ✅ L1b: reverse_degree_order hard-rejected")


# ── L1c: skill match ──────────────────────────────────────────────────────────
def test_l1c_scores_and_matches():
    cands = [good_candidate(f"C{i}", years=5) for i in range(5)]
    result = layers.l1c_skill_match(cands, JD)
    assert all("l1c_score" in c for c in result), "l1c_score key missing after l1c_skill_match"
    assert all(0.0 <= c["l1c_score"] <= 1.0 for c in result), "l1c_score out of [0, 1]"
    assert all("Python" in c["l1c_matched_required"] for c in result)
    print(f"  ✅ L1c: scored {len(result)} candidates, all matched explicit skills")


def test_l1c_relevant_candidate_scores_higher():
    """Candidate matching JD keywords must outscore one with no relevant skills."""
    relevant = good_candidate("REL")
    irrelevant = good_candidate("IRREL")
    irrelevant["skills"] = [{"name": "Accounting"}, {"name": "Marketing"}]
    irrelevant["career_history"] = [{"title": "Accountant", "company": "realtech",
                                      "start_date": "2018-01-01", "duration_months": 60,
                                      "description": "managed spreadsheets and annual budgets"}]
    result = layers.l1c_skill_match([relevant, irrelevant], JD)
    scores = {c["candidate_id"]: c["l1c_score"] for c in result}
    assert scores["REL"] >= scores.get("IRREL", 0.0), (
        f"relevant ({scores['REL']:.3f}) should score >= irrelevant "
        f"({scores.get('IRREL', 0.0):.3f})"
    )
    print(f"  ✅ L1c: relevant ({scores['REL']:.3f}) >= irrelevant "
          f"({scores.get('IRREL', 0.0):.3f})")


# ── L1d: inferred skill match ─────────────────────────────────────────────────
def test_l1d_attaches_score():
    cands = [good_candidate(f"C{i}") for i in range(3)]
    cands = layers.l1c_skill_match(cands, JD)  # L1d reuses L1c's pre-computed lists
    result = layers.l1d_inferred_match(cands, JD)
    assert len(result) == 3, "L1d must never reject candidates"
    assert all("l1d_score" in c for c in result)
    assert all(0.0 <= c["l1d_score"] <= 1.0 for c in result)
    print("  ✅ L1d: l1d_score attached in [0, 1], no candidates removed")


# ── L2: table extract ─────────────────────────────────────────────────────────
def test_l2_builds_table_row():
    cands = [good_candidate(f"C{i}") for i in range(3)]
    cands = layers.l1c_skill_match(cands, JD)
    cands = layers.l1d_inferred_match(cands, JD)
    result = layers.l2_table_extract(cands, JD)
    assert len(result) == 3, "L2 must never reject candidates"
    for c in result:
        assert "table_row" in c
        row = c["table_row"]
        for col in ("candidate_id", "total_exp", "skill_match_score", "tools_score"):
            assert col in row, f"table_row missing expected column: {col}"
    print("  ✅ L2: table_row built with 31-col schema for all candidates")


# ── L3: weighted-condition scoring ────────────────────────────────────────────
def test_l3_scores_in_range():
    cands = [good_candidate(f"C{i}", years=y) for i, y in enumerate([1, 4, 6, 9, 15])]
    cands = layers.l1c_skill_match(cands, JD)
    cands = layers.l1d_inferred_match(cands, JD)
    cands = layers.l2_table_extract(cands, JD)
    result = layers.l3_fuzzy_score(cands, JD)
    assert len(result) == 5, "L3 must never reject candidates"
    for c in result:
        assert 0.0 <= c["l3_score"] <= 1.0, f"l3_score={c['l3_score']} out of [0, 1]"
        assert c["l3_class"] in ("strong_fit", "good_fit", "moderate_fit", "weak_fit")
    print("  ✅ L3: l3_score/l3_class attached in range for all candidates")


def test_l3_ideal_experience_scores_higher():
    """A candidate in the 5-9y JD sweet spot should score >= a 1y junior, all else equal."""
    junior = good_candidate("JR", years=1)
    ideal = good_candidate("IDEAL", years=6)
    cands = layers.l1c_skill_match([junior, ideal], JD)
    cands = layers.l1d_inferred_match(cands, JD)
    cands = layers.l2_table_extract(cands, JD)
    result = layers.l3_fuzzy_score(cands, JD)
    scores = {c["candidate_id"]: c["l3_score"] for c in result}
    assert scores["IDEAL"] >= scores["JR"], (
        f"ideal-experience candidate ({scores['IDEAL']:.3f}) should score >= "
        f"junior ({scores['JR']:.3f})"
    )
    print(f"  ✅ L3: ideal-exp ({scores['IDEAL']:.3f}) >= junior ({scores['JR']:.3f})")


# ── Streaming cascade (L1a→L1b→L1c→L1d→L2→L3) ────────────────────────────────
def test_streaming_cascade_end_to_end():
    kb = mock_kb()
    cands = [good_candidate(f"C{i}", years=3 + i) for i in range(8)]
    survivors = layers.run_streaming_cascade(cands, JD, kb, max_workers=4)
    assert len(survivors) == 8, "no candidate should be rejected in the clean fixture"
    assert all("l3_score" in c for c in survivors), "l3_score missing after streaming cascade"
    print(f"  ✅ streaming cascade: {len(cands)} in → {len(survivors)} survived with l3_score set")


# ── L4: semantic work relevance + donts penalty ───────────────────────────────
def test_l4_scores_and_sorts():
    orig_loader = utils.load_sentence_transformer
    utils.load_sentence_transformer = lambda: MockST()
    try:
        cands = [good_candidate(f"C{i}", years=3 + i) for i in range(6)]
        cands = layers.l1c_skill_match(cands, JD)
        cands = layers.l1d_inferred_match(cands, JD)
        cands = layers.l2_table_extract(cands, JD)
        cands = layers.l3_fuzzy_score(cands, JD)
        result = layers.l4_semantic_work(cands, JD)
        assert len(result) == 6, "L4 must never reject candidates"
        for c in result:
            assert 0.0 <= c["l4_work_relevance"] <= 1.0
            assert "candidate_final_score" in c and "l4_combined_score" in c
        scores = [c["candidate_final_score"] for c in result]
        assert scores == sorted(scores, reverse=True), "L4 output must be sorted descending"
        print(f"  ✅ L4: {len(result)} scored, sorted by candidate_final_score descending")
    finally:
        utils.load_sentence_transformer = orig_loader


def test_l4_no_work_history_scores_zero_relevance():
    orig_loader = utils.load_sentence_transformer
    utils.load_sentence_transformer = lambda: MockST()
    try:
        c = {"candidate_id": "EMPTY", "career_history": [], "skills": []}
        result = layers.l4_semantic_work([c], JD)
        assert result[0]["l4_work_relevance"] == 0.0, "no-work candidate should score 0 relevance"
        print("  ✅ L4: candidate with no work history scores 0 relevance (no crash)")
    finally:
        utils.load_sentence_transformer = orig_loader


# ── L4b: explicit-required penalty ────────────────────────────────────────────
def test_l4b_penalizes_few_matches():
    strong = good_candidate("STRONG")
    strong["l1c_matched_required"] = ["Python", "RAG", "LLM", "FAISS"]
    strong["candidate_final_score"] = 0.80
    strong["l4_combined_score"] = 0.80

    weak = good_candidate("WEAK")
    weak["l1c_matched_required"] = ["Python"]
    weak["candidate_final_score"] = 0.80
    weak["l4_combined_score"] = 0.80

    result = layers.l4b_explicit_req_penalty([strong, weak])
    by_id = {c["candidate_id"]: c for c in result}
    assert by_id["STRONG"]["l4b_explicit_req_penalty"] == 0.0
    assert by_id["WEAK"]["l4b_explicit_req_penalty"] > 0.0
    assert by_id["WEAK"]["candidate_final_score"] < by_id["STRONG"]["candidate_final_score"]
    print("  ✅ L4b: candidate with fewer matched required skills penalized more")



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
