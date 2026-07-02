#!/usr/bin/env python3
"""
build_kb.py — One-time artifact builder (run BEFORE submission).

This downloads, quantizes, and compresses every model the pipeline needs,
plus builds the SQLite fraud knowledge base. It may exceed the 5-minute
ranking budget — that is fine. This is pre-computation, NOT the ranking step.

Outputs land in:
  models/decompressed/   (ready-to-load artifacts)
  models/compressed/     (.tar.gz archives committed to Git)

Usage:
    python build_kb.py --all
    python build_kb.py --sentence-transformer --spacy   # selective

After running, commit models/compressed/*.tar.gz (via Git LFS).
On a fresh clone, setup.sh decompresses them back into models/decompressed/.
"""

import json
import time
import tarfile
import hashlib
import sqlite3
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_kb")

ROOT = Path(__file__).parent
DECOMP = ROOT / "models" / "decompressed"
COMP = ROOT / "models" / "compressed"
DECOMP.mkdir(parents=True, exist_ok=True)
COMP.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. SENTENCE-TRANSFORMER (bi-encoder, L2/L4) → ONNX INT8
# ──────────────────────────────────────────────────────────────────────────────
def build_sentence_transformer():
    logger.info("Building sentence-transformer (ONNX INT8)...")
    out_dir = DECOMP / "sentence_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sentence_transformers import SentenceTransformer
        # Download + save in ST format (works on CPU offline after this).
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        model.save(str(out_dir))
        # Optional ONNX INT8 export via optimum (if installed)
        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from optimum.onnxruntime.configuration import AutoQuantizationConfig
            from optimum.onnxruntime import ORTQuantizer
            onnx_dir = out_dir / "onnx"
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2", export=True)
            ort_model.save_pretrained(onnx_dir)
            quantizer = ORTQuantizer.from_pretrained(onnx_dir)
            qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
            quantizer.quantize(save_dir=onnx_dir, quantization_config=qconfig)
            logger.info("  ONNX INT8 export complete")
        except Exception as e:
            logger.warning(f"  ONNX INT8 export skipped ({e}); ST format saved (still CPU-fast)")
        _compress("sentence_transformer")
        return True
    except Exception as e:
        logger.error(f"sentence-transformer build failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 2. spaCy (L5 helper)
# ──────────────────────────────────────────────────────────────────────────────
def build_spacy():
    logger.info("Building spaCy model...")
    out_dir = DECOMP / "spacy_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import spacy
        from spacy.cli import download as spacy_download
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy_download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        nlp.to_disk(out_dir / "en_core_web_sm")
        _compress("spacy_model")
        return True
    except Exception as e:
        logger.error(f"spaCy build failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# 3. FRAUD KB (L1) — SQLite from public sources / curated lists
# ──────────────────────────────────────────────────────────────────────────────
FICTIONAL = [
    ("dunder mifflin", "TV fiction (The Office)"),
    ("hooli", "TV fiction (Silicon Valley)"),
    ("pied piper", "TV fiction (Silicon Valley)"),
    ("acme corp", "cartoon fiction"),
    ("acme corporation", "cartoon fiction"),
    ("initech", "film fiction (Office Space)"),
    ("globex", "TV fiction (Simpsons)"),
    ("soylent corp", "film fiction"),
    ("umbrella corporation", "game fiction (Resident Evil)"),
    ("stark industries", "comic fiction (Marvel)"),
    ("wayne enterprises", "comic fiction (DC)"),
    ("cyberdyne systems", "film fiction (Terminator)"),
    ("weyland-yutani", "film fiction (Alien)"),
    ("tyrell corporation", "film fiction (Blade Runner)"),
    ("oscorp", "comic fiction (Marvel)"),
    ("aperture science", "game fiction (Portal)"),
    ("black mesa", "game fiction (Half-Life)"),
    ("vault-tec", "game fiction (Fallout)"),
    ("wonka industries", "film fiction"),
    ("massive dynamic", "TV fiction (Fringe)"),
]

# A tiny seed of real company founding years. In production, populate this from
# MCA data.gov.in / DPIIT / public registries via build scripts.
COMPANY_FOUNDING_SEED = [
    ("google", 1998), ("microsoft", 1975), ("amazon", 1994), ("apple", 1976),
    ("meta", 2004), ("facebook", 2004), ("netflix", 1997), ("tesla", 2003),
    ("openai", 2015), ("anthropic", 2021), ("nvidia", 1993), ("infosys", 1981),
    ("tcs", 1968), ("wipro", 1945), ("flipkart", 2007), ("zomato", 2008),
    ("paytm", 2010), ("ola", 2010), ("swiggy", 2014), ("razorpay", 2014),
]

SKILL_ALIASES = [
    ("large language models", "llm", "ml"),
    ("machine learning", "ml", "ml"),
    ("natural language processing", "nlp", "nlp"),
    ("deep learning", "dl", "ml"),
    ("retrieval augmented generation", "rag", "nlp"),
    ("python", "python3", "lang"),
    ("kubernetes", "k8s", "infra"),
]


def build_fraud_kb():
    """Build fraud KB using the enhanced multi-source builder (rebuild_fraud_kb.py).

    Sources included:
      - Indian Companies   : MCA data.gov.in
      - Indian Startups    : DPIIT Startup India
      - Indian Universities: AICTE + UGC
      - Global Companies   : Kaggle PDL 7M
      - Global Universities: WHED UNESCO
      - Research Venues    : ArXiv bulk metadata
    """
    logger.info("Building fraud KB (SQLite) — enhanced multi-source build...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rebuild_fraud_kb", ROOT / "rebuild_fraud_kb.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok = mod.build()
        if ok:
            logger.info("  Fraud KB (enhanced) built successfully")
        return ok
    except Exception as e:
        logger.error(f"Enhanced fraud KB build failed: {e}; falling back to seed-only build")
        # Minimal fallback so the pipeline isn't broken
        out_dir = DECOMP / "fraud_kb"
        out_dir.mkdir(parents=True, exist_ok=True)
        db_path = out_dir / "fraud_kb.db"
        if db_path.exists():
            db_path.unlink()
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE fictional_companies (company_name TEXT PRIMARY KEY, reason TEXT)")
        conn.execute("CREATE TABLE company_founding_dates (company_name TEXT PRIMARY KEY, founding_year INT)")
        conn.execute("CREATE TABLE skill_aliases (skill_canonical TEXT, alias TEXT, category TEXT)")
        conn.executemany("INSERT OR IGNORE INTO fictional_companies VALUES (?,?)", FICTIONAL)
        conn.executemany("INSERT OR IGNORE INTO company_founding_dates VALUES (?,?)", COMPANY_FOUNDING_SEED)
        conn.executemany("INSERT INTO skill_aliases VALUES (?,?,?)", SKILL_ALIASES)
        conn.execute("CREATE INDEX idx_fic ON fictional_companies(company_name)")
        conn.execute("CREATE INDEX idx_found ON company_founding_dates(company_name)")
        conn.execute("CREATE INDEX idx_alias ON skill_aliases(skill_canonical)")
        conn.commit()
        conn.close()
        logger.info(f"  Fraud KB (fallback): {len(FICTIONAL)} fictional, "
                    f"{len(COMPANY_FOUNDING_SEED)} real companies")
        _compress("fraud_kb")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# COMPRESSION + CHECKSUMS
# ──────────────────────────────────────────────────────────────────────────────
def _compress(name: str):
    src = DECOMP / name
    if not src.exists():
        logger.warning(f"  nothing to compress for {name}")
        return
    tar_path = COMP / f"{name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src, arcname=name)
    size_mb = tar_path.stat().st_size / 1e6
    logger.info(f"  compressed → {tar_path.name} ({size_mb:.1f} MB)")
    _update_checksum(tar_path)


def _update_checksum(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    checksum_file = COMP / "checksums.sha256"
    lines = {}
    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            if "  " in line:
                cs, fn = line.split("  ", 1)
                lines[fn] = cs
    lines[path.name] = h.hexdigest()
    checksum_file.write_text("".join(f"{cs}  {fn}\n" for fn, cs in sorted(lines.items())))


def main():
    ap = argparse.ArgumentParser(description="Build all RedRob model artifacts")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--sentence-transformer", action="store_true")
    ap.add_argument("--spacy", action="store_true")
    ap.add_argument("--fraud-kb", action="store_true")
    args = ap.parse_args()

    if not any(vars(args).values()):
        ap.error("Pass --all or specific flags (e.g. --fraud-kb)")

    t0 = time.time()
    results = {}
    if args.all or args.fraud_kb:            results["fraud_kb"] = build_fraud_kb()
    if args.all or args.sentence_transformer: results["sentence_transformer"] = build_sentence_transformer()
    if args.all or args.spacy:               results["spacy"] = build_spacy()

    print("\n" + "=" * 50)
    print("BUILD SUMMARY")
    print("=" * 50)
    for name, ok_ in results.items():
        print(f"  {'OK' if ok_ else 'FAIL'} {name}")
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Compressed artifacts in: {COMP}")
    print("Next: commit models/compressed/*.tar.gz (Git LFS), then run setup_check.py")


if __name__ == "__main__":
    main()
