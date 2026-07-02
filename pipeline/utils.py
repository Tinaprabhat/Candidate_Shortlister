"""
utils.py — Model loading & shared helpers for rank.py.

Three models only (all in models/decompressed/ after setup.sh):
  1. sentence_transformer/  — all-MiniLM-L6-v2 (ONNX INT8 + PyTorch fallback)
  2. ms-marco-MiniLM-L-12-v2/ — FlashRank cross-encoder (ONNX INT8)
  3. fraud_kb/fraud_kb.db    — SQLite fraud knowledge base

L4 computes cosine similarity directly via numpy (no FAISS index — the
"faiss" weight key elsewhere refers to a scoring signal name, not this
library). No quantization at runtime; the happy path (models already
decompressed) needs no network, but load_sentence_transformer() falls
back to a by-name download if the local model directory is missing.
"""

import json
import logging
import sqlite3
import tarfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from . import constants as C

logger = logging.getLogger(__name__)

FRAUD_KB_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DECOMPRESSION — runs once per machine; decompressed/ persists after that
# ──────────────────────────────────────────────────────────────────────────────
def ensure_models_decompressed() -> None:
    """
    Decompress models/compressed/*.tar.gz into models/decompressed/ if not
    already present. No-op (just a directory-exists check) once decompressed —
    models/decompressed/ is meant to persist between runs, not be rebuilt.
    """
    comp_dir = C.MODELS_COMPRESSED_DIR
    decomp_dir = C.MODELS_DIR

    if not comp_dir.exists():
        return  # nothing to decompress from (e.g. dev box without LFS archives)

    decomp_dir.mkdir(parents=True, exist_ok=True)

    for archive in sorted(comp_dir.glob("*.tar.gz")):
        target = decomp_dir / archive.stem.removesuffix(".tar")
        if target.exists():
            continue  # already decompressed (some archives, e.g. flashrank_onnx, are legitimately empty)
        logger.info(f"[model-setup] Decompressing {archive.name} → {decomp_dir}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(decomp_dir)


class _Cache:
    store: Dict[str, Any] = {}


def _get(name: str, loader):
    if name not in _Cache.store:
        t = time.perf_counter()
        _Cache.store[name] = loader()
        elapsed = round(time.perf_counter() - t, 3)
        logger.info(f"[model-load] '{name}' loaded in {elapsed}s")
        _Cache.store[f"{name}__load_elapsed_s"] = elapsed
    return _Cache.store[name]


# ──────────────────────────────────────────────────────────────────────────────
# SENTENCE-TRANSFORMER — all-MiniLM-L6-v2 (ONNX INT8; PyTorch fallback)
# ──────────────────────────────────────────────────────────────────────────────
def load_sentence_transformer():
    """Load the bi-encoder. Prefers ONNX INT8; falls back to PyTorch ST."""
    def _load():
        from sentence_transformers import SentenceTransformer
        model_dir = C.SENTENCE_TRANSFORMER_DIR
        onnx_path = model_dir / "onnx" / "model_quantized.onnx"
        if onnx_path.exists():
            try:
                logger.info(f"Loading ONNX INT8 sentence-transformer from {onnx_path}")
                return SentenceTransformer(str(model_dir), device="cpu", backend="onnx")
            except Exception as e:
                logger.warning(f"ONNX INT8 load failed ({e}); falling back to PyTorch")
        if model_dir.exists():
            logger.info(f"Loading sentence-transformer (PyTorch) from {model_dir}")
            return SentenceTransformer(str(model_dir), device="cpu")
        logger.warning(f"{model_dir} not found; loading {C.ST_MODEL_NAME} by name (requires network)")
        return SentenceTransformer(C.ST_MODEL_NAME, device="cpu")
    return _get("st", _load)


# ──────────────────────────────────────────────────────────────────────────────
# FLASHRANK — ms-marco-MiniLM-L-12-v2 (ONNX INT8 cross-encoder, L7 polish)
# ──────────────────────────────────────────────────────────────────────────────
def load_flashrank():
    """Load FlashRank cross-encoder from the local ms-marco-MiniLM-L-12-v2 directory."""
    def _load():
        try:
            from flashrank import Ranker
        except ImportError:
            raise ImportError("flashrank required: pip install flashrank")
        logger.info(f"Loading FlashRank ({C.FLASHRANK_MODEL_NAME}) from {C.FLASHRANK_DIR}")
        # Ranker(model_name, cache_dir) looks for cache_dir/model_name/
        # FLASHRANK_DIR IS the model directory, so cache_dir = FLASHRANK_DIR.parent
        return Ranker(model_name=C.FLASHRANK_MODEL_NAME, cache_dir=str(C.FLASHRANK_DIR.parent))
    return _get("flashrank", _load)


# ──────────────────────────────────────────────────────────────────────────────
# FRAUD KB — SQLite knowledge base (L1 verification)
# ──────────────────────────────────────────────────────────────────────────────
def load_fraud_kb():
    def _load():
        if not C.FRAUD_KB_PATH.exists():
            logger.warning(f"Fraud KB not found at {C.FRAUD_KB_PATH}; L1 uses in-code blacklist only")
            return None
        logger.info(f"Loading fraud KB from {C.FRAUD_KB_PATH}")
        conn = sqlite3.connect(str(C.FRAUD_KB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    return _get("fraud_kb", _load)


# ──────────────────────────────────────────────────────────────────────────────
# JD JSON
# ──────────────────────────────────────────────────────────────────────────────
def load_jd_json(path: Path = None) -> dict:
    """
    Load jd.json and normalize it via jd_parser.validate_and_fill() — every
    schema field is guaranteed present (correct type/default), skill lists
    are deduped, and what_you_will_do is synthesised from candidate_work if
    the file left it empty. jd.json itself is expected to already exist on
    disk; this function does not parse or generate it.
    """
    path = path or C.JD_JSON_PATH
    if not path.exists():
        raise FileNotFoundError(f"jd.json not found at {path}.")
    t = time.perf_counter()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    from . import jd_parser
    data = jd_parser.validate_and_fill(data)
    logger.info(f"[io] load_jd_json: {path.name} read in {round(time.perf_counter()-t, 3)}s")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# CANDIDATE LOADING
# ──────────────────────────────────────────────────────────────────────────────
def read_json(path: Path) -> List[dict]:
    """
    Read a .json file into a list of candidate dicts.

    Handles three formats:
      - Bare array:    [{...}, {...}]
      - Wrapped obj:   {"candidates": [...], "meta": {...}}
                       (also "data", "results", "items", "records")
      - Single object: {...}  → treated as a one-element list
    """
    t = time.perf_counter()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        result = [item for item in data if isinstance(item, dict)]
    elif isinstance(data, dict):
        result = []
        for key in ("candidates", "data", "results", "items", "records"):
            val = data.get(key)
            if isinstance(val, list):
                result = [item for item in val if isinstance(item, dict)]
                break
        if not result:
            result = [data]
    else:
        result = []
    logger.info(f"[io] read_json: {path.name} → {len(result)} records in {round(time.perf_counter()-t, 3)}s")
    return result


def read_jsonl(path: Path) -> List[dict]:
    """Read a .jsonl file into a list of dicts."""
    t = time.perf_counter()
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed line in {path}")
    logger.info(f"[io] read_jsonl: {path.name} → {len(out)} records in {round(time.perf_counter()-t, 3)}s")
    return out


def cleanup():
    """Close any open resources (e.g. SQLite)."""
    fk = _Cache.store.get("fraud_kb")
    if fk is not None:
        try:
            fk.close()
        except Exception:
            pass
    _Cache.store.clear()


# ──────────────────────────────────────────────────────────────────────────────
# PROFILE TEXT HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _iter_work_history(candidate: dict):
    """
    Yield work-history entries from career_history (new schema) or
    work_experience (old schema), preferring career_history when both present.
    """
    for key in ("career_history", "work_experience"):
        entries = candidate.get(key) or []
        if isinstance(entries, list) and entries:
            yield from entries
            return  # use whichever key is populated first


def get_skills_text(candidate: dict) -> str:
    """Extract skills section as a flat string."""
    skills = candidate.get("skills", [])
    if isinstance(skills, list):
        parts = []
        for s in skills:
            if isinstance(s, dict):
                parts.append(str(s.get("name", "")))
            else:
                parts.append(str(s))
        return " ".join(parts)
    return str(skills)


def get_work_profile_text(candidate: dict) -> str:
    """
    Full skills + profile summary + work descriptions — used for L2.

    Works with both schemas:
      Old: work_experience[].description
      New: profile.summary + profile.headline + career_history[].description
    """
    parts = [get_skills_text(candidate)]

    # Profile-level summary / headline (new schema)
    profile = candidate.get("profile") or {}
    if isinstance(profile, dict):
        for field in ("summary", "headline"):
            val = profile.get(field, "")
            if val:
                parts.append(str(val))

    # Work-history descriptions (both schemas)
    for exp in _iter_work_history(candidate):
        if isinstance(exp, dict):
            d = exp.get("description", "")
            if d:
                parts.append(str(d))

    # Projects (old schema; new schema has no projects field)
    for proj in (candidate.get("projects") or []):
        if isinstance(proj, dict):
            d = proj.get("description", "")
            if d:
                parts.append(str(d))

    return " ".join(p for p in parts if p).strip()


def get_work_descriptions_only(candidate: dict) -> str:
    """
    L4 input: ONLY work / project descriptions.
    NO titles, company names, or skill lists.
    """
    parts = []
    for exp in _iter_work_history(candidate):
        if isinstance(exp, dict):
            d = exp.get("description", "")
            if d:
                parts.append(str(d))
    for proj in (candidate.get("projects") or []):
        if isinstance(proj, dict):
            d = proj.get("description", "")
            if d:
                parts.append(str(d))
    return " ".join(parts).strip()


def get_total_experience_years(candidate: dict) -> float:
    """
    Best-effort total years of experience.

    Priority:
      1. Explicit top-level field: total_experience_years
      2. profile.years_of_experience (new schema)
      3. Sum duration_years / duration_months from work history
    """
    # 1. Top-level explicit field
    for key in ("total_experience_years", "years_of_experience"):
        val = candidate.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # 2. Nested under profile (new schema)
    profile = candidate.get("profile") or {}
    if isinstance(profile, dict):
        val = profile.get("years_of_experience")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # 3. Sum from work history entries
    total = 0.0
    for exp in _iter_work_history(candidate):
        if not isinstance(exp, dict):
            continue
        if "duration_years" in exp:
            try:
                total += float(exp["duration_years"])
            except (ValueError, TypeError):
                pass
        elif "duration_months" in exp:
            try:
                total += float(exp["duration_months"]) / 12.0
            except (ValueError, TypeError):
                pass
    return total
