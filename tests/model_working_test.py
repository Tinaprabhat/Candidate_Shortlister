"""
tests/model_working_test.py — Verifies all models load correctly or degrade
gracefully when artifacts are absent.

Does NOT import heavy models at module level — each test loads its target
lazily so a missing model does not abort the whole suite.

Run:  python -m pytest tests/model_working_test.py -v
  or: python tests/model_working_test.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import constants as C


# ── Fraud KB (L1) ─────────────────────────────────────────────────────────────
def test_fraud_kb_loads_or_none():
    """Fraud KB should return a sqlite3 connection or None — never raise."""
    from pipeline import utils
    kb = utils.load_fraud_kb()
    assert kb is None or hasattr(kb, "execute"), \
        f"load_fraud_kb() returned unexpected type: {type(kb)}"
    status = "loaded" if kb is not None else "gracefully absent"
    print(f"  ✅ fraud KB: {status}")


def test_fraud_kb_schema_when_present():
    """If the fraud KB loads, both expected tables must exist."""
    from pipeline import utils
    kb = utils.load_fraud_kb()
    if kb is None:
        print("  ⚠️  fraud KB absent — skipping schema check")
        return
    tables = {row[0] for row in kb.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "fictional_companies" in tables, \
        f"fraud KB missing 'fictional_companies' table; got {sorted(tables)}"
    assert "company_founding_dates" in tables, \
        f"fraud KB missing 'company_founding_dates' table; got {sorted(tables)}"
    print(f"  ✅ fraud KB schema: required tables present {sorted(tables)}")


def test_fraud_kb_fictional_companies_populated():
    """fictional_companies table must contain at least one row."""
    from pipeline import utils
    kb = utils.load_fraud_kb()
    if kb is None:
        print("  ⚠️  fraud KB absent — skipping population check")
        return
    count = kb.execute("SELECT COUNT(*) FROM fictional_companies").fetchone()[0]
    assert count > 0, "fictional_companies table is empty — fraud KB may be corrupt"
    print(f"  ✅ fraud KB: {count} fictional company entries loaded")


# ── Sentence Transformer (L2 + L4) ────────────────────────────────────────────
def test_sentence_transformer_loads():
    """Sentence transformer must load (ONNX or PyTorch) — it is required for ranking."""
    from pipeline import utils
    try:
        st = utils.load_sentence_transformer()
    except Exception as e:
        raise AssertionError(
            f"Sentence transformer failed to load (required for ranking): {e}"
        ) from e
    assert st is not None, "load_sentence_transformer() returned None"
    assert hasattr(st, "encode"), "ST object missing .encode() method"
    print("  ✅ sentence transformer: loaded with .encode()")


def test_sentence_transformer_encodes_normalized_vectors():
    """ST must produce L2-normalized float32 vectors with the correct dimension."""
    import numpy as np
    from pipeline import utils
    try:
        st = utils.load_sentence_transformer()
    except Exception:
        print("  ⚠️  sentence transformer unavailable — skipping encode test")
        return
    sentences = ["hello world", "python machine learning FAISS"]
    vecs = st.encode(sentences, normalize_embeddings=True, convert_to_numpy=True)
    assert vecs.ndim == 2, f"encode() should return 2-D array; got shape {vecs.shape}"
    assert vecs.shape[0] == len(sentences), "one vector per sentence expected"
    assert vecs.dtype in (np.float32, np.float64), f"unexpected dtype: {vecs.dtype}"
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), \
        f"vectors not L2-normalized; norms={norms}"
    print(f"  ✅ sentence transformer: encodes {vecs.shape[1]}-dim L2-normalized vectors")


def test_sentence_transformer_consistent_embeddings():
    """Same input must produce the same embedding on repeated calls."""
    import numpy as np
    from pipeline import utils
    try:
        st = utils.load_sentence_transformer()
    except Exception:
        print("  ⚠️  sentence transformer unavailable — skipping consistency test")
        return
    text = ["deterministic embedding test sentence"]
    v1 = st.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    v2 = st.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    assert np.allclose(v1, v2, atol=1e-5), \
        "sentence transformer produces different embeddings for the same input"
    print("  ✅ sentence transformer: embeddings are deterministic")


# ── FAISS index (L4, optional) ────────────────────────────────────────────────
def test_faiss_index_loads_or_none():
    """FAISS index is optional — must return an index or None, never raise."""
    from pipeline import utils
    try:
        idx = utils.load_faiss_index()
    except Exception as e:
        print(f"  ⚠️  FAISS not installed or index error: {e} — acceptable in dev")
        return
    status = "loaded" if idx is not None else "gracefully absent"
    print(f"  ✅ FAISS index: {status}")


# ── FlashRank (L7 polish) ─────────────────────────────────────────────────────
def test_flashrank_loads_or_error():
    """FlashRank must load when installed; ImportError is acceptable if not installed."""
    from pipeline import utils
    try:
        fr = utils.load_flashrank()
        assert hasattr(fr, "rerank"), \
            f"FlashRank object missing .rerank() method; got {type(fr)}"
        print("  ✅ FlashRank: loaded with .rerank()")
    except ImportError as e:
        print(f"  ⚠️  FlashRank not installed: {e} — install with: pip install flashrank")


# ── spaCy ────────────────────────────────────────────────────────────────────
def test_spacy_loads_or_error():
    """spaCy must load when installed; ImportError / OSError are acceptable if not."""
    from pipeline import utils
    try:
        nlp = utils.load_spacy()
        assert callable(nlp), f"spaCy model is not callable; got {type(nlp)}"
        print("  ✅ spaCy: loaded and callable")
    except (ImportError, OSError) as e:
        print(f"  ⚠️  spaCy unavailable: {e} — install with: pip install spacy && python -m spacy download en_core_web_sm")


# ── constants sanity check ────────────────────────────────────────────────────
def test_model_path_constants_defined():
    """All model-path constants expected by utils must exist in constants.py."""
    required = [
        "SENTENCE_TRANSFORMER_DIR", "FAISS_INDEX_DIR", "FLASHRANK_DIR",
        "SPACY_DIR", "FRAUD_KB_PATH", "MODELS_DIR",
    ]
    missing = [name for name in required if not hasattr(C, name)]
    assert not missing, f"constants.py missing path constants: {missing}"
    print(f"  ✅ all {len(required)} model-path constants present in constants.py")


def test_model_dirs_exist_after_setup():
    """models/decompressed/ must exist (created by setup.sh / build_kb.py)."""
    assert C.MODELS_DIR.exists(), (
        f"models/decompressed/ not found at {C.MODELS_DIR}. "
        "Run: bash setup.sh  (or python build_kb.py for a fresh build)"
    )
    print(f"  ✅ models/decompressed/ exists at {C.MODELS_DIR}")


# ── runner ────────────────────────────────────────────────────────────────────
def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"Running {len(tests)} model-working tests\n")
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
