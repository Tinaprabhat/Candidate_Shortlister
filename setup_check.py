#!/usr/bin/env python3
"""
setup_check.py — Pre-flight check.

Verifies that everything needed to run RedRob is present and working,
BEFORE you start a ranking run. Run this after setup.sh.

    python setup_check.py

Three required model artifacts:
  1. sentence_transformer/   — all-MiniLM-L6-v2 (ONNX INT8 + PyTorch fallback)
  2. ms-marco-MiniLM-L-12-v2/ — FlashRank cross-encoder (ONNX INT8)
  3. fraud_kb/fraud_kb.db    — SQLite fraud knowledge base

FAISS is built in-memory by L4 — no on-disk artifact required.

Checks (in order):
  1. Python version
  2. Required Python packages importable
  3. Model artifacts on disk
  4. Sentence-transformer live encode test
  5. Fraud KB SQLite opens
  6. JD parsing prerequisites — warning only
  7. data/jd.json present — warning only

Exit code 0 = ready to rank. Non-zero = blocking problem found.
"""

import sys
import shutil
import importlib
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; NC = "\033[0m"
def ok(m):   print(f"  {GREEN}✅ {m}{NC}")
def warn(m): print(f"  {YEL}⚠️  {m}{NC}")
def err(m):  print(f"  {RED}❌ {m}{NC}")

blocking = 0
warnings = 0


def check_python():
    print("1. Python version")
    global blocking
    if sys.version_info >= (3, 9):
        ok(f"Python {sys.version.split()[0]}")
    else:
        err(f"Python {sys.version.split()[0]} — need ≥ 3.9")
        blocking += 1


def check_packages():
    print("\n2. Required packages")
    global blocking, warnings
    required = ["numpy", "sentence_transformers"]
    optional = {
        "faiss": "L4 FAISS similarity search (in-memory; built each run)",
        "flashrank": "L7 cross-encoder polish (falls back to FIS order)",
        "fitz": "PDF parsing (only needed for JD pre-step)",
        "groq": "Groq API integration (only needed if GROQ_API_KEY is set)",
        "streamlit": "Web UI (only needed for app.py)",
    }
    for pkg in required:
        try:
            importlib.import_module(pkg)
            ok(f"{pkg}")
        except ImportError:
            err(f"{pkg} missing — pip install -r requirements.txt")
            blocking += 1
    for pkg, why in optional.items():
        try:
            importlib.import_module(pkg)
            ok(f"{pkg}")
        except ImportError:
            warn(f"{pkg} missing — {why}")
            warnings += 1


def check_models():
    print("\n3. Model artifacts on disk (3 required)")
    global warnings
    from src import constants as C
    artifacts = {
        "MiniLM sentence-transformer (ONNX INT8)": C.SENTENCE_TRANSFORMER_DIR / "onnx" / "model_quantized.onnx",
        "MiniLM sentence-transformer (PyTorch)":   C.SENTENCE_TRANSFORMER_DIR,
        "FlashRank cross-encoder (ONNX INT8)":     C.FLASHRANK_DIR,
        "Fraud KB (SQLite)":                        C.FRAUD_KB_PATH,
    }
    for name, path in artifacts.items():
        if Path(path).exists():
            ok(f"{name}: {path}")
        else:
            warn(f"{name} not found at {path}")
            warnings += 1


def check_biencoder():
    print("\n4. Bi-encoder live test")
    global blocking
    try:
        from src import utils
        model = utils.load_sentence_transformer()
        v = model.encode(["hello world"], convert_to_numpy=True, normalize_embeddings=True)
        if v is not None and len(v) == 1:
            ok(f"Bi-encoder encodes OK (dim={len(v[0])})")
        else:
            err("Bi-encoder returned unexpected shape")
            blocking += 1
    except Exception as e:
        err(f"Bi-encoder failed to load/encode: {e}")
        err("  (If offline, ensure models/decompressed/sentence_transformer exists)")
        blocking += 1


def check_fraud_kb():
    print("\n5. Fraud KB")
    global warnings
    try:
        from src import utils
        kb = utils.load_fraud_kb()
        if kb is None:
            warn("Fraud KB not found — L1 will use in-code blacklist only")
            warnings += 1
        else:
            n = kb.execute("SELECT COUNT(*) FROM fictional_companies").fetchone()[0]
            ok(f"Fraud KB open ({n} fictional companies)")
    except Exception as e:
        warn(f"Fraud KB check skipped: {e}")
        warnings += 1


def check_jd_parsing_prerequisites():
    print("\n6. JD parsing prerequisites")
    global warnings

    if shutil.which("ollama") is not None:
        ok("Ollama CLI available")
    else:
        warn("Ollama CLI not found — JD parsing requires local Ollama or an existing jd.json")
        warnings += 1

    env = ROOT / ".env"
    if env.exists() and "OLLAMA_MODEL" in env.read_text(encoding="utf-8-sig", errors="ignore"):
        ok("OLLAMA_MODEL found in .env")
    else:
        warn("OLLAMA_MODEL not set in .env — defaulting to mistral:latest")
        warnings += 1

    from src import constants as C
    if Path(C.JD_JSON_PATH).exists():
        ok(f"jd.json present: {C.JD_JSON_PATH}")
    else:
        warn(f"jd.json not found at {C.JD_JSON_PATH} — "
             f"run: python -m src.jd_parser --jd <job_description> --out {C.JD_JSON_PATH}")
        warnings += 1


def main():
    print("=" * 60)
    print("RedRob — Setup Check")
    print("=" * 60)
    check_python()
    check_packages()
    check_models()
    check_biencoder()
    check_fraud_kb()
    check_jd_parsing_prerequisites()

    print("\n" + "=" * 60)
    if blocking == 0:
        print(f"{GREEN}✅ READY TO RANK{NC} "
              f"({warnings} warnings — fallbacks available)")
        print("Run: python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv")
        sys.exit(0)
    else:
        print(f"{RED}❌ {blocking} blocking problem(s) found.{NC} "
              f"Fix these before ranking.")
        sys.exit(1)


if __name__ == "__main__":
    main()
