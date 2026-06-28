"""
pruning.py — Structured + heuristic folder pruning, then timed thread dispatch.

Pruning is applied in five phases (hard → soft):

  Phase 1a — Code/No-code: drop root-level buckets whose code/no-code type
                           contradicts the JD requirement (e.g. no_code/ when
                           JD is a software-engineering role).
  Phase 1b — Inactive    : drop folders whose name identifies them as an
                           inactive/unavailable candidate bucket.
  Phase 1c — Location    : drop folders that are explicitly outside the JD city
                           (e.g. india_outside when JD is Bangalore).
  Phase 1d — Experience  : drop folders whose experience ceiling is below the
                           JD minimum (e.g. 0-3/ when JD requires 5+ yrs).
  Phase 1e — Role        : drop folders labelled as a different role family
                           (e.g. other_role when JD is an engineering position).
  Phase 2  — Token-overlap: score surviving folders against JD title + skills;
                           all Phase-1 survivors are kept, sorted by relevance.

Tree layout expected on disk (new structure):
  <code|no_code>/
    <available|unavailable>/
      <0-3|4-9|10+>/
        <engineering|cloud|devops|…>/
          <role_bucket>.json   ← e.g. swe_.json, ai_engineer.json

Each leaf JSON file is treated as its own candidate bucket (keyed by its full
relative path including the file stem), so multiple files in the same domain
folder are all discovered and pruned independently.

After folder selection, individual inactive candidates are also filtered out
inside each kept folder before the cascade layers run.

NO AI model used here.  All decisions are deterministic rule-based logic.
"""

import re
import time
import logging
import threading
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Optional

from . import constants as C
from . import utils

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# INTERNAL REGEX (experience range parsing)
# ──────────────────────────────────────────────────────────────────────────────
_EXP_RANGE_RE = re.compile(
    r'(\d{1,2})\s*[-_–]\s*(\d{1,2})\s*(?:yr|yrs|year|years)?\b', re.I
)
_EXP_RANGE_TO_RE = re.compile(
    r'(\d{1,2})\s+to\s+(\d{1,2})\s*(?:yr|yrs|year|years)?\b', re.I
)
_EXP_PLUS_RE = re.compile(
    r'(\d{1,2})\s*\+\s*(?:yr|yrs|year|years)?(?:\b|$)', re.I
)
_EXP_PLUS_WORD_RE = re.compile(
    r'(\d{1,2})\s*plus\b', re.I
)

# Leaf-level tokens that carry no role signal; excluded from Phase-2 scoring
_STATUS_TOKENS: frozenset = frozenset({
    "active", "inactive", "moderate", "open", "closed",
    "available", "unavailable", "eligible",
})


# ──────────────────────────────────────────────────────────────────────────────
# TOKEN-OVERLAP HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _normalize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, expand abbreviations."""
    text = text.lower().replace("_", " ").replace("-", " ")
    tokens = re.findall(r"[a-z0-9]+", text)
    expanded = []
    for t in tokens:
        if t in C.FOLDER_ABBREVIATIONS:
            expanded.extend(C.FOLDER_ABBREVIATIONS[t].split())
        else:
            expanded.append(t)
    return expanded


def build_jd_signal(jd: dict) -> List[str]:
    """JD title + top explicit skills → normalized token list."""
    title = jd.get("job_title", "")
    top_skills = jd.get("explicit_required", [])[:10]
    signal_text = title + " " + " ".join(top_skills)
    tokens = _normalize(signal_text)
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score_folder(folder_name: str, jd_signal: List[str]) -> float:
    """
    Token-overlap score between folder path and JD signal (0..1).

    Scores ALL path components except known status-bucket words (active,
    inactive, moderate …).  This lets role components like "ai_ml" or
    "software_engineering" contribute tokens that overlap with the JD signal.
    """
    if not jd_signal:
        return 0.0
    parts = folder_name.replace("\\", "/").split("/")
    # Keep only non-status components for scoring
    role_parts = [
        p for p in parts
        if p.lower().replace("_", " ").strip() not in _STATUS_TOKENS
    ]
    combined = " ".join(role_parts)
    folder_tokens = set(_normalize(combined))
    if not folder_tokens:
        return 0.0
    overlap = sum(1 for t in jd_signal if t in folder_tokens)
    return overlap / len(jd_signal)


# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED PRUNING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _path_parts_normalized(path_str: str) -> List[str]:
    """Split folder path into components and normalise each (lowercase, _/- → space)."""
    return [
        p.lower().replace("_", " ").replace("-", " ")
        for p in path_str.replace("\\", "/").split("/")
    ]


def _classify_jd_role(jd: dict) -> str:
    """Return 'engineering' if the JD role belongs to the engineering family, else 'other'."""
    title = jd.get("job_title", "").lower()
    skills = " ".join(
        jd.get("explicit_required", []) + jd.get("inferred_required", [])
    ).lower()
    combined = title + " " + skills
    for keyword in C.ENGINEERING_ROLE_KEYWORDS:
        if keyword in combined:
            return "engineering"
    return "other"


def _classify_jd_coding(jd: dict) -> str:
    """
    Return 'code' if the JD requires coding/technical skills, 'no_code' otherwise.
    """
    for field in ("requires_coding", "coding_required", "is_technical"):
        val = jd.get(field)
        if val is not None:
            return "code" if bool(val) else "no_code"

    job_type = str(jd.get("job_type") or "").lower()
    if job_type:
        for token in ("no_code", "no code", "nocode", "non tech", "non_tech", "non-tech",
                      "no coding", "non coding"):
            if token in job_type:
                return "no_code"
        for token in ("code", "tech", "technical", "software", "engineering"):
            if token in job_type:
                return "code"

    return "code" if _classify_jd_role(jd) == "engineering" else "no_code"


def _is_coding_type_excluded(path_str: str, coding_type: str) -> bool:
    """
    Return True if the folder's root-level bucket contradicts the JD coding type.
    """
    parts = _path_parts_normalized(path_str)
    root = parts[0] if parts else ""
    if coding_type == "code":
        return root in C.NO_CODE_FOLDER_ROOTS
    if coding_type == "no_code":
        return root in C.CODE_FOLDER_ROOTS
    return False


def _parse_exp_from_path(path_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse an experience range from a folder path string.
    Returns (min_years, max_years) or None if no pattern found.
    """
    raw_parts = path_str.replace("\\", "/").split("/")
    candidates = []
    for p in raw_parts:
        candidates.append(p)
        norm = p.replace("_", " ")
        if norm != p:
            candidates.append(norm)
    candidates.append(path_str)

    for part in candidates:
        m = _EXP_RANGE_RE.search(part) or _EXP_RANGE_TO_RE.search(part)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = _EXP_PLUS_RE.search(part) or _EXP_PLUS_WORD_RE.search(part)
        if m:
            return int(m.group(1)), 99
    return None


def _is_location_excluded(path_str: str, jd_location: str) -> bool:
    """
    Return True if this folder path should be dropped because it explicitly
    represents a location that does not match the JD's target.
    """
    if not jd_location:
        return False
    jd_loc = jd_location.lower().strip()
    parts = _path_parts_normalized(path_str)
    root = parts[0] if parts else ""

    if "india" in jd_loc:
        if "outside" in root and "india" in root:
            return True

    if any(alias in jd_loc for alias in C.BANGALORE_ALIASES):
        for part in parts:
            for pattern in C.INDIA_OUTSIDE_PATTERNS:
                if pattern in part:
                    return True

    return False


def _is_experience_excluded(path_str: str, exp_min: int) -> bool:
    """
    Return True if this folder's maximum experience ceiling is below the JD's
    minimum requirement.
    """
    if exp_min <= 0:
        return False
    exp_range = _parse_exp_from_path(path_str)
    if exp_range is None:
        return False
    return exp_range[1] < exp_min


def _is_role_excluded(path_str: str, role_category: str) -> bool:
    """
    Return True if this folder explicitly represents a role family that doesn't
    match the JD.
    """
    if role_category != "engineering":
        return False
    parts = _path_parts_normalized(path_str)
    for part in parts:
        for pattern in C.OTHER_ROLE_FOLDER_PATTERNS:
            if pattern in part:
                return True
        if part == "other":
            return True
    return False


def _is_inactive_folder(path_str: str) -> bool:
    """
    Return True if ANY component of this folder path is a known inactive-bucket
    name. Uses exact match on each normalised path component.
    """
    parts = _path_parts_normalized(path_str)
    for part in parts:
        if part in C.INACTIVE_FOLDER_NAMES:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# INACTIVE CANDIDATE FILTER (within a kept folder)
# ──────────────────────────────────────────────────────────────────────────────
def filter_inactive_candidates(candidates: List[dict]) -> List[dict]:
    """
    Remove individual candidates whose status field marks them as inactive.
    """
    active, n_pruned = [], 0
    for c in candidates:
        raw_status = (
            c.get("status")
            or c.get("candidate_status")
            or c.get("active_status")
            or ""
        )
        if str(raw_status).lower().strip() in C.INACTIVE_CANDIDATE_STATUSES:
            n_pruned += 1
        else:
            active.append(c)
    if n_pruned:
        logger.info(f"[inactive-filter] removed {n_pruned} inactive candidate(s)")
    return active


# ──────────────────────────────────────────────────────────────────────────────
# TREE-STRUCTURE DISPLAY
# ──────────────────────────────────────────────────────────────────────────────
def _render_folder_tree(folder_keys: List[str], decisions: Dict[str, str]) -> str:
    lines = ["Folder tree:"]
    for i, key in enumerate(sorted(folder_keys)):
        connector = "└──" if i == len(folder_keys) - 1 else "├──"
        decision = decisions.get(key, "KEEP")
        lines.append(f"  {connector} {key}  [{decision}]")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PRUNING ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def prune_folders(
    folder_names: List[str],
    jd: dict,
) -> List[Tuple[str, float]]:
    """
    Score and filter candidate buckets in two phases.

    Phase 1 (structured hard-prune) — code/no-code → inactive → location
                                      → experience → role
    Phase 2 (token-overlap soft-prune): sort survivors by relevance score.

    Returns kept buckets sorted by token-overlap score descending:
      [(bucket_key, score), ...]
    """
    jd_location  = jd.get("location", "").strip()
    exp_min      = int(jd.get("experience_min") or 0)
    role_category = _classify_jd_role(jd)
    coding_type  = _classify_jd_coding(jd)

    logger.info(
        f"Pruning meta — coding='{coding_type}', location='{jd_location}', "
        f"exp_min={exp_min}yrs, role='{role_category}'"
    )

    # ── Phase 1: structured hard-prune ────────────────────────────────────────
    phase1a_excluded: set = set()
    remaining: List[str] = []
    decisions: Dict[str, str] = {}

    for name in folder_names:
        if _is_coding_type_excluded(name, coding_type):
            phase1a_excluded.add(name)
            reason = f"coding-type mismatch (JD wants '{coding_type}')"
            logger.info(f"  '{name}': hard-DROP [1a] → {reason}")
            decisions[name] = f"DROP:{reason}"
        elif _is_inactive_folder(name):
            reason = "inactive/unavailable folder"
            logger.info(f"  '{name}': hard-DROP [1b] → {reason}")
            decisions[name] = f"DROP:{reason}"
        elif _is_location_excluded(name, jd_location):
            reason = f"location outside '{jd_location}'"
            logger.info(f"  '{name}': hard-DROP [1c] → {reason}")
            decisions[name] = f"DROP:{reason}"
        elif _is_experience_excluded(name, exp_min):
            reason = f"experience ceiling below {exp_min}yrs"
            logger.info(f"  '{name}': hard-DROP [1d] → {reason}")
            decisions[name] = f"DROP:{reason}"
        elif _is_role_excluded(name, role_category):
            reason = f"role mismatch (not {role_category})"
            logger.info(f"  '{name}': hard-DROP [1e] → {reason}")
            decisions[name] = f"DROP:{reason}"
        else:
            remaining.append(name)

    if not remaining:
        eligible = [n for n in folder_names if n not in phase1a_excluded]
        if eligible:
            logger.warning(
                f"Phase 1b–1e pruned all {len(folder_names) - len(phase1a_excluded)} "
                f"post-1a folders; reverting heuristic drops "
                f"({len(phase1a_excluded)} Phase-1a exclusions kept)"
            )
            remaining = eligible
        else:
            logger.error(
                f"All {len(folder_names)} folders are Phase-1a excluded "
                f"(coding_type='{coding_type}').  "
                "Check that the ZIP contains the correct code/no_code root bucket."
            )
            remaining = []

    # ── Phase 2: token-overlap ranking ────────────────────────────────────────
    jd_signal = build_jd_signal(jd)
    logger.info(f"JD signal tokens: {jd_signal}")

    scored = [(name, score_folder(name, jd_signal)) for name in remaining]
    scored.sort(key=lambda x: x[1], reverse=True)
    kept = scored

    for name, score in kept:
        decisions[name] = f"KEEP (score={score:.3f})"
        logger.info(f"  '{name}': score={score:.3f} → KEEP")

    logger.info("\n" + _render_folder_tree(folder_names, decisions))

    return kept


# ──────────────────────────────────────────────────────────────────────────────
# ZIP HANDLING
# ──────────────────────────────────────────────────────────────────────────────
def extract_zip(zip_path: Path) -> Path:
    """Extract ZIP to a temp dir, return the root path."""
    tmp = Path(tempfile.mkdtemp(prefix="redrob_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    logger.info(f"Extracted ZIP → {tmp}")
    return tmp


def discover_folders(root: Path) -> Dict[str, Path]:
    """
    Find all candidate data buckets inside the extracted ZIP root.

    Key format: "<relative_dir>/<file_stem>"  (e.g.
      "code/available/0-3/engineering/swe_"   or
      "code/available/4-9/cloud/cloud_engineer")

    Multiple JSON files in the same domain folder are each registered as
    independent buckets.
    """
    folders: Dict[str, Path] = {}

    for pattern in ("*.json", "*.jsonl"):
        for data_file in root.rglob(pattern):
            rel_dir = data_file.parent.relative_to(root)
            rel_dir_str = str(rel_dir).replace("\\", "/")
            if rel_dir_str == ".":
                folder_key = data_file.stem
            else:
                folder_key = f"{rel_dir_str}/{data_file.stem}"
            folders[folder_key] = data_file

    if not folders:
        for data_file in root.glob("*.json"):
            folders[data_file.stem] = data_file
        for data_file in root.glob("*.jsonl"):
            if data_file.stem not in folders:
                folders[data_file.stem] = data_file

    logger.info(
        f"Discovered {len(folders)} candidate bucket(s):\n"
        + "\n".join(f"  {k}  →  {v.name}" for k, v in sorted(folders.items()))
    )
    return folders


# ──────────────────────────────────────────────────────────────────────────────
# CANDIDATE LOADING (with format auto-detection)
# ──────────────────────────────────────────────────────────────────────────────
def _load_candidates(data_path: Path) -> List[dict]:
    """Load candidates from .json (array) or .jsonl (one-per-line)."""
    if data_path.suffix.lower() == ".jsonl":
        return utils.read_jsonl(data_path)
    return utils.read_json(data_path)


# ──────────────────────────────────────────────────────────────────────────────
# TIMED THREAD DISPATCH
# ──────────────────────────────────────────────────────────────────────────────
def dispatch_folders_staggered(
    kept_folders: List[Tuple[str, float]],
    folder_paths: Dict[str, Path],
    process_fn: Callable[[str, List[dict]], None],
    stagger_sec: int = None,
    tracer=None,
):
    """
    Dispatch each folder's candidates into `process_fn` with a stagger.
    """
    stagger_sec = stagger_sec if stagger_sec is not None else C.FOLDER_DISPATCH_STAGGER_SEC
    threads = []

    def _worker(name: str, data_path: Path):
        logger.info(f"[dispatch] loading '{name}' from {data_path.name}")
        try:
            t_load = time.perf_counter()
            candidates = _load_candidates(data_path)
            load_elapsed = round(time.perf_counter() - t_load, 3)
        except Exception as exc:
            logger.error(f"[dispatch] failed to load '{name}': {exc}")
            if tracer:
                tracer.record_error(f"load:{name}", str(exc))
            return

        n_loaded = len(candidates)
        logger.info(f"[dispatch] '{name}': file read in {load_elapsed}s")

        for c in candidates:
            c["_folder"] = name

        candidates = filter_inactive_candidates(candidates)
        n_active = len(candidates)

        logger.info(
            f"[dispatch] '{name}': loaded={n_loaded}, "
            f"active={n_active} → cascade"
        )
        if tracer:
            tracer.record_folder_load(name, n_loaded, n_active)

        if not candidates:
            logger.warning(f"[dispatch] '{name}': no active candidates, skipping")
            return

        process_fn(name, candidates)

    for i, (name, score) in enumerate(kept_folders):
        data_path = folder_paths[name]
        t = threading.Thread(target=_worker, args=(name, data_path), daemon=True)
        threads.append(t)
        t.start()
        logger.info(
            f"[dispatch] '{name}' thread started (score={score:.3f}, "
            f"stagger={i * stagger_sec}s)"
        )
        if i < len(kept_folders) - 1:
            time.sleep(stagger_sec)

    for t in threads:
        t.join()
    logger.info("[dispatch] all folder threads complete")
