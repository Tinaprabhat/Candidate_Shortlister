"""
pruning.py — Structured folder pruning + timed thread dispatch.

Fixed tree layout (5 levels):
  <code|no_code>/
    <available|unavailable>/
      <0_to_3|4_to_9|10_plus>/        ← experience tier
        <engineering|devops_and_cloud|hr_and_people|…>/   ← domain
          <role_stem>.json             ← e.g. ai_engineer.json

Pruning phases (hard → soft):

  Phase 1a — Code/No-code : drop buckets whose code/no-code type
                            contradicts the JD (e.g. no_code/ when JD
                            is a software-engineering role).
  Phase 1b — Inactive     : drop folders explicitly marked unavailable /
                            inactive.
  Phase 1c — Experience   : drop folders whose experience ceiling is
                            below the JD minimum (e.g. 0_to_3/ when JD
                            requires 5+ yrs).
  Phase 1d — Role         : drop folders whose domain or role stem is
                            incompatible with the JD role family
                            (e.g. hr_and_people/ or net_developer for an
                            AI/ML JD).

  Phase 2  — Token-overlap: score surviving folders against JD title +
                            skills; all Phase-1 survivors are kept,
                            sorted by relevance.

JD signals consumed:
  • role_type  (derived) → Phase 1a
  • experience_min       → Phase 1c
  • role family (derived)→ Phase 1d  domain + stem filtering
  • job_title + skills   → Phase 2   token-overlap ranking

For AI roles (ai_ml family): ML, SWE, Data, DL, NLP, CV buckets are
open; java, .net, mobile, frontend, cloud/devops, QA are excluded.

NO AI model used. All decisions are deterministic rule-based logic.
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
# INTERNAL REGEX (experience tier parsing)
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

# Tokens that carry no role signal; skipped in Phase-2 scoring
_STATUS_TOKENS: frozenset = frozenset({
    "active", "inactive", "moderate", "open", "closed",
    "available", "unavailable", "eligible",
})

# Short-form non-engineering domain folder names.
# These are exact matches against a single normalised path component —
# handles folders named "HR", "hr_and_people", "Sales", etc. that would
# not be caught by the longer substring patterns in C.OTHER_ROLE_FOLDER_PATTERNS.
_NON_ENG_DOMAIN_PARTS: frozenset = frozenset({
    "hr", "hr and people", "human resources",
    "sales", "marketing", "finance", "legal",
    "operations", "admin", "administration",
    "talent", "recruitment", "accounting", "accounts",
    "customer support", "customer success", "support",
    "business", "business development",
    "product and design", "design",
    "non tech engineering",
    "other", "unclassified",
})

# Engineering sub-family names — used to guard Phase 1d
_ENGINEERING_FAMILIES: frozenset = frozenset({
    "ai_ml", "cloud_devops", "qa", "mobile", "java_net", "swe_general",
})


# ──────────────────────────────────────────────────────────────────────────────
# JD CLASSIFICATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _classify_jd_role(jd: dict) -> str:
    """
    Return the JD's role family:
      'ai_ml' | 'cloud_devops' | 'qa' | 'mobile' | 'java_net' |
      'swe_general' | 'other'

    Fine-grained families from C.ROLE_FAMILY_KEYWORDS are checked first
    (insertion order). If none match but a broad engineering keyword
    (C.ENGINEERING_ROLE_KEYWORDS) does, returns 'swe_general'. Non-
    engineering JDs return 'other'.
    """
    title  = jd.get("job_title", "").lower()
    skills = " ".join(
        jd.get("explicit_required", []) + jd.get("inferred_required", [])
    ).lower()
    combined = title + " " + skills

    for family, keywords in C.ROLE_FAMILY_KEYWORDS.items():
        if any(kw in combined for kw in keywords):
            return family

    for kw in C.ENGINEERING_ROLE_KEYWORDS:
        if kw in combined:
            return "swe_general"

    return "other"


def _classify_jd_coding(jd: dict) -> str:
    """
    Return 'code' if the JD requires coding/technical skills, else 'no_code'.

    Detection order (first match wins):
      1. Explicit boolean flags: requires_coding / coding_required / is_technical
      2. job_type string containing a no-code or code keyword
      3. Fall back to engineering role classification (engineering → code)
    """
    for field in ("requires_coding", "coding_required", "is_technical"):
        val = jd.get(field)
        if val is not None:
            return "code" if bool(val) else "no_code"

    job_type = str(jd.get("job_type") or "").lower()
    if job_type:
        for token in ("no_code", "no code", "nocode", "non tech", "non_tech",
                      "non-tech", "no coding", "non coding"):
            if token in job_type:
                return "no_code"
        for token in ("code", "tech", "technical", "software", "engineering"):
            if token in job_type:
                return "code"

    return "code" if _classify_jd_role(jd) in _ENGINEERING_FAMILIES else "no_code"


# ──────────────────────────────────────────────────────────────────────────────
# TOKEN-OVERLAP HELPERS (Phase 2)
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
    """JD title + top explicit skills → normalised, deduplicated token list."""
    title      = jd.get("job_title", "")
    top_skills = jd.get("explicit_required", [])[:10]
    tokens     = _normalize(title + " " + " ".join(top_skills))
    seen, out  = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score_folder(folder_name: str, jd_signal: List[str]) -> float:
    """
    Token-overlap score between folder path and JD signal [0, 1].

    Scores all path components except known status-bucket tokens (active,
    inactive, available …) so role components like "ai_ml" or
    "software_engineering" contribute to the JD title/skills match.
    """
    if not jd_signal:
        return 0.0
    parts = folder_name.replace("\\", "/").split("/")
    role_parts = [
        p for p in parts
        if p.lower().replace("_", " ").strip() not in _STATUS_TOKENS
    ]
    folder_tokens = set(_normalize(" ".join(role_parts)))
    if not folder_tokens:
        return 0.0
    overlap = sum(1 for t in jd_signal if t in folder_tokens)
    return overlap / len(jd_signal)


# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED PRUNING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _path_parts_normalized(path_str: str) -> List[str]:
    """Split path into components; lowercase each and replace _/- with space."""
    return [
        p.lower().replace("_", " ").replace("-", " ")
        for p in path_str.replace("\\", "/").split("/")
    ]


# ── floor sets for code/no-code and inactive bucket tokens ───────────────────
_NO_CODE_ROOT_FLOOR: frozenset = frozenset({"no_code", "nocode", "no-code"})
_CODE_ROOT_FLOOR:    frozenset = frozenset({"code"})
_INACTIVE_FLOOR:     frozenset = frozenset({"unavailable", "inactive", "rejected", "closed"})


def _root_token_set(configured: set, floor: frozenset) -> set:
    """Union configured tokens (raw + space-normalised) with the floor set."""
    out = set(floor)
    for t in configured:
        t = str(t).lower().strip()
        out.add(t)
        out.add(t.replace("_", " ").replace("-", " "))
    return out


# ── Phase 1a ─────────────────────────────────────────────────────────────────
def _is_coding_type_excluded(path_str: str, coding_type: str) -> bool:
    """
    Return True if the folder's code/no-code bucket contradicts the JD.

    Scans ALL path components for the first recognised code/no-code token
    (handles a Dataset/ wrapper prefix without breaking).
    """
    raw_parts = [p.strip().lower() for p in path_str.replace("\\", "/").split("/") if p.strip()]

    no_code_tokens = _root_token_set(getattr(C, "NO_CODE_FOLDER_ROOTS", set()), _NO_CODE_ROOT_FLOOR)
    code_tokens    = _root_token_set(getattr(C, "CODE_FOLDER_ROOTS",    set()), _CODE_ROOT_FLOOR)

    for part in raw_parts:
        norm       = part.replace("_", " ").replace("-", " ")
        is_no_code = part in no_code_tokens or norm in no_code_tokens
        is_code    = part in code_tokens    or norm in code_tokens

        if is_no_code or is_code:
            if coding_type == "code":
                return is_no_code
            if coding_type == "no_code":
                return is_code
            return False

    return False


# ── Phase 1b ─────────────────────────────────────────────────────────────────
def _is_inactive_folder(path_str: str) -> bool:
    """
    Return True if ANY path component names an inactive/unavailable bucket.

    Checks both raw (underscore) and space-normalised forms against the
    configured + floor inactive token sets.
    """
    raw_parts  = [p.lower().strip() for p in path_str.replace("\\", "/").split("/")]
    norm_parts = _path_parts_normalized(path_str)
    inactive_tokens = _root_token_set(getattr(C, "INACTIVE_FOLDER_NAMES", set()), _INACTIVE_FLOOR)
    for part in raw_parts + norm_parts:
        if part in inactive_tokens:
            return True
    return False


# ── Phase 1c ─────────────────────────────────────────────────────────────────
def _parse_exp_from_path(path_str: str) -> Optional[Tuple[int, int]]:
    """
    Parse an experience range from a folder path.
    Returns (min_years, max_years) or None.

    Recognised formats (tried on both raw and _→space normalised forms):
      "0-3", "4-9", "0_to_3", "4 to 9"  → dash/word range   (lo, hi)
      "10+", "10_plus", "10 plus"         → open-ended bound  (10, 99)
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


def _is_experience_excluded(path_str: str, exp_min: int) -> bool:
    """
    Return True if folder's experience ceiling is below the JD minimum.
    e.g. 0_to_3/ (ceiling=3) excluded when JD requires 5+ years.
    """
    if exp_min <= 0:
        return False
    exp_range = _parse_exp_from_path(path_str)
    if exp_range is None:
        return False
    return exp_range[1] < exp_min


# ── Phase 1d ─────────────────────────────────────────────────────────────────
def _is_role_excluded(path_str: str, jd_role_family: str) -> bool:
    """
    Return True if the folder's domain or role stem is incompatible with
    the JD role family.

    Phase 1d-i  — non-engineering domain detection:
      Checks BOTH short-form exact matches (_NON_ENG_DOMAIN_PARTS, e.g.
      "hr", "hr and people") AND longer substring patterns
      (C.OTHER_ROLE_FOLDER_PATTERNS, e.g. "human resources").
      This correctly handles folder names like "HR", "hr_and_people",
      "Sales", "product_and_design", etc.

    Phase 1d-ii — cross-engineering sub-family exclusion:
      Uses C.ROLE_FAMILY_EXCLUSIONS to drop role stems incompatible with
      the JD family (e.g. "net developer", "frontend engineer" for AI JD).
    """
    if jd_role_family not in _ENGINEERING_FAMILIES:
        return False

    parts = _path_parts_normalized(path_str)

    # 1d-i: non-engineering domains — exact match on short names first,
    # then substring match on longer patterns
    for part in parts:
        if part in _NON_ENG_DOMAIN_PARTS:
            return True
        for pattern in C.OTHER_ROLE_FOLDER_PATTERNS:
            if pattern in part:
                return True

    # 1d-ii: incompatible engineering sub-family (substring match)
    exclusions = C.ROLE_FAMILY_EXCLUSIONS.get(jd_role_family, frozenset())
    for part in parts:
        for pattern in exclusions:
            if pattern in part:
                return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# INACTIVE CANDIDATE FILTER (within a kept folder)
# ──────────────────────────────────────────────────────────────────────────────
def filter_inactive_candidates(candidates: List[dict]) -> List[dict]:
    """
    Remove individual candidates whose status field marks them inactive.
    Checks: status, candidate_status, active_status (first non-empty wins).
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
        decision  = decisions.get(key, "KEEP")
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

    Phase 1 (structured hard-prune):
      1a code/no-code → 1b inactive → 1c experience → 1d role/domain

    Phase 2 (token-overlap soft-rank):
      All Phase-1 survivors kept, sorted by JD-signal overlap score.

    Returns kept buckets sorted by score descending:
      [(bucket_key, score), ...]
    """
    exp_min       = int(jd.get("experience_min") or 0)
    role_family   = _classify_jd_role(jd)
    coding_type   = _classify_jd_coding(jd)

    logger.info(
        f"Pruning meta — coding='{coding_type}', exp_min={exp_min}yrs, "
        f"role='{role_family}'"
    )

    # ── Phase 1: structured hard-prune ───────────────────────────────────────
    # Phase 1a drops are tracked separately; they are CATEGORICAL exclusions
    # (wrong candidate type entirely) and must never be restored by the
    # fallback that handles over-aggressive Phase 1b–1d heuristics.
    phase1a_excluded: set          = set()
    remaining:        List[str]    = []
    decisions:        Dict[str, str] = {}

    for name in folder_names:
        if _is_coding_type_excluded(name, coding_type):      # 1a — categorical
            phase1a_excluded.add(name)
            reason = f"coding-type mismatch (JD wants '{coding_type}')"
            logger.info(f"  '{name}': hard-DROP [1a] → {reason}")
            decisions[name] = f"DROP:{reason}"

        elif _is_inactive_folder(name):                      # 1b — availability
            reason = "inactive/unavailable folder"
            logger.info(f"  '{name}': hard-DROP [1b] → {reason}")
            decisions[name] = f"DROP:{reason}"

        elif _is_experience_excluded(name, exp_min):         # 1c — experience
            reason = f"experience ceiling below {exp_min}yrs"
            logger.info(f"  '{name}': hard-DROP [1c] → {reason}")
            decisions[name] = f"DROP:{reason}"

        elif _is_role_excluded(name, role_family):           # 1d — role/domain
            reason = f"role mismatch (JD family: {role_family})"
            logger.info(f"  '{name}': hard-DROP [1d] → {reason}")
            decisions[name] = f"DROP:{reason}"

        else:
            remaining.append(name)

    # Fallback: if Phase 1b–1d were over-restrictive (e.g. no exp range in
    # any folder name), re-admit Phase 1b–1d drops. Phase 1a drops are
    # NEVER re-admitted — wrong coding type is a hard categorical exclusion.
    if not remaining:
        eligible = [n for n in folder_names if n not in phase1a_excluded]
        if eligible:
            logger.warning(
                f"Phase 1b–1d pruned all {len(folder_names) - len(phase1a_excluded)} "
                f"post-1a folders; reverting heuristic drops "
                f"({len(phase1a_excluded)} Phase-1a exclusions kept)"
            )
            remaining = eligible
        else:
            logger.error(
                f"All {len(folder_names)} folders are Phase-1a excluded "
                f"(coding_type='{coding_type}'). "
                "Check that the ZIP contains the correct code/no_code root bucket."
            )
            remaining = []

    # ── Phase 2: token-overlap ranking ───────────────────────────────────────
    jd_signal = build_jd_signal(jd)
    logger.info(f"JD signal tokens: {jd_signal}")

    scored = [(name, score_folder(name, jd_signal)) for name in remaining]
    scored.sort(key=lambda x: x[1], reverse=True)

    for name, score in scored:
        decisions[name] = f"KEEP (score={score:.3f})"
        logger.info(f"  '{name}': score={score:.3f} → KEEP")

    logger.info("\n" + _render_folder_tree(folder_names, decisions))
    return scored


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

    Returns {bucket_key: path_to_data_file}.

    Key format: "<relative_dir>/<file_stem>"
      e.g. "Dataset/code/available/4_to_9/engineering/ai_engineer"

    Multiple JSON files in the same domain folder are each registered as
    independent buckets. Both .json (array) and .jsonl (one-per-line) are
    supported.
    """
    folders: Dict[str, Path] = {}

    for pattern in ("*.json", "*.jsonl"):
        for data_file in root.rglob(pattern):
            rel_dir     = data_file.parent.relative_to(root)
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
# CANDIDATE LOADING
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
    Dispatch each folder's candidates into process_fn with a stagger.

    Steps per folder:
      1. Load candidates from .json / .jsonl file.
      2. Filter out individually inactive candidates.
      3. Tag each candidate with its originating folder path.
      4. Hand off to process_fn on a dedicated thread.

    Threads are released stagger_sec apart (default: C.FOLDER_DISPATCH_STAGGER_SEC).
    All threads are joined before returning.
    process_fn(folder_name, candidates) must be thread-safe.
    """
    stagger_sec = stagger_sec if stagger_sec is not None else C.FOLDER_DISPATCH_STAGGER_SEC
    threads = []

    def _worker(name: str, data_path: Path):
        logger.info(f"[dispatch] loading '{name}' from {data_path.name}")
        try:
            t_load     = time.perf_counter()
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
        n_active   = len(candidates)

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
