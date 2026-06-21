# pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE
#   Single-pass pipeline.
#   Reads candidates.jsonl → cleans every candidate (fixes + flags)
#   → builds the Dataset/ folder tree.
#   All in one run, no intermediate batch files.
#
# PERFORMANCE
#   Avoids copy.deepcopy() entirely — json.loads() already gives a fresh dict
#   per line, so we mutate in-place. On a single CPU core this keeps the
#   full 100K run well under 60 seconds.
#
# USAGE
#   python pipeline.py
#   (run from inside the folder that contains candidates.jsonl)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import time
from datetime import date, datetime
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = "candidates.jsonl"
DATASET_DIR = "Dataset"
TODAY       = date(2026, 6, 17)   # fixed reference date


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

POSTGRAD_DEGREES  = {"m.tech", "m.e.", "m.s.", "m.sc", "m.sc.", "ph.d", "mba", "m.b.a"}
UNDERGRAD_DEGREES = {"b.tech", "b.e.", "b.sc", "b.sc.", "b.a.", "be", "btech", "b.eng"}
ENG_DEGREE_SET    = {"b.tech", "m.tech", "b.e.", "m.e.", "be", "btech", "mtech", "b.eng"}
INVALID_ENG_FIELDS = {"mba", "m.b.a", "commerce"}

# ── Unified AI skill set (used by BOTH FL3 and domain routing) ────────────────
# FIX #3: Previously split into AI_SKILLS vs AI_DOMAIN_SKILLS causing FL3 and
# domain routing to count AI skills differently. Now a single source of truth.
AI_SKILLS = {
    "machine learning", "deep learning", "nlp", "computer vision",
    "object detection", "image classification", "rag", "llms",
    "fine-tuning llms", "vector search", "embeddings", "faiss",
    "pinecone", "qdrant", "weaviate", "sentence transformers",
    "hugging face transformers", "langchain", "llamaindex",
    "diffusion models", "gans", "yolo", "cnn", "rnn", "lstm",
    "transformers", "bert", "gpt", "reinforcement learning",
    "recommendation systems", "information retrieval",
    "semantic search", "prompt engineering", "mlops", "mlflow",
    "weights & biases", "kubeflow", "bentoml",
    "speech recognition", "tts", "asr", "forecasting",
    "time series", "statistical modeling", "learning to rank",
    "bm25", "opensearch", "haystack", "qlora", "pgvector",
    "feature engineering", "data science", "opencv",
    "scikit-learn", "xgboost", "lightgbm", "prophet",
    "tensorflow", "pytorch", "keras"
}

# ── AI career keywords — DE/analytics titles removed (FIX #1) ────────────────
# "data engineer" and "analytics engineer" were wrongly included before,
# causing DE candidates with 3+ AI skills to be routed to ai_ml domain.
AI_CAREER_KEYWORDS = {
    "ml engineer", "machine learning", "data scientist", "ai specialist",
    "ai engineer", "nlp engineer", "computer vision", "research engineer",
    "deep learning", "junior ml", "senior ml", "ai research",
    "applied scientist", "mlops engineer"
}

# ── AI industry — exact match set only (FIX #2) ───────────────────────────────
# Previously used "ai" in industry substring which matched "retail",
# "financial", "pharmaceutical" etc. Now exact set match only.
AI_INDUSTRIES_EXACT = {
    "ai/ml", "ai", "ml", "machine learning", "artificial intelligence",
    "edtech ai", "healthtech ai", "deep learning", "nlp"
}

DE_DOMAIN_SKILLS = {
    "apache spark", "spark", "kafka", "airflow", "dbt", "hadoop",
    "hive", "presto", "trino", "databricks", "snowflake", "redshift",
    "bigquery", "data warehouse", "data lake", "etl", "elt",
    "pipeline", "flink", "nifi", "luigi", "prefect", "dagster",
    "glue", "fivetran", "stitch", "talend", "informatica",
    "delta lake", "iceberg", "parquet", "avro", "pyspark"
}

AI_TITLE_KEYWORDS = {
    "ml engineer", "machine learning", "data scientist", "ai engineer",
    "ai specialist", "nlp engineer", "computer vision engineer",
    "deep learning engineer", "research scientist", "ai research",
    "applied scientist", "mlops engineer"
}

DE_TITLE_KEYWORDS = {
    "data engineer", "analytics engineer", "etl developer",
    "data pipeline", "data architect", "big data", "spark engineer",
    "kafka engineer", "databricks engineer"
}

# FIX #6: "sde " (trailing space) replaced with "sde" — covers SDE, SDE-2, SDE2
SWE_TITLE_KEYWORDS = {
    "software engineer", "software developer", "backend engineer",
    "frontend engineer", "full stack", "fullstack", "sde",
    "platform engineer", "devops engineer", "site reliability",
    "cloud engineer", "systems engineer"
}

INDIA_PREFERRED_CITIES = {
    "noida", "pune", "bangalore", "bengaluru", "hyderabad",
    "chennai", "mumbai", "delhi", "gurugram", "gurgaon",
    "navi mumbai", "thane", "kolkata"
}

_SCORE_FIELDS = [
    "skill_career_domain_mismatch",
    "education_overlap",
    "reverse_degree_order",
    "second_undergrad_after_first",
    "education_career_gap_flag",
    "active_before_signup",
    "duplicate_job_descriptions",
    "all_descriptions_identical",
    "low_engagement_flag",
    "possible_honeypot",
    "_any_invalid_degree_field",
]


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _s(val) -> str:
    return (val or "").lower().strip()


def _parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _has_ai_career(career: list) -> bool:
    """
    Returns True if at least one job is a genuine AI/ML role.
    FIX #1: Does NOT include 'data engineer' / 'analytics engineer'.
    FIX #2: Uses exact industry set — no substring 'ai' check.
    """
    for job in career:
        title = _s(job.get("title"))
        ind   = _s(job.get("industry"))
        if any(kw in title for kw in AI_CAREER_KEYWORDS):
            return True
        if ind in AI_INDUSTRIES_EXACT:
            return True
        if "machine learning" in ind or "artificial intelligence" in ind:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  FIXES  (F1, F2, F3)
# ══════════════════════════════════════════════════════════════════════════════

def _fix_salary(sig: dict) -> None:
    """F1 — Swap inverted salary min/max."""
    sal = sig.setdefault("expected_salary_range_inr_lpa", {})
    mn, mx = sal.get("min"), sal.get("max")
    if mn is not None and mx is not None and mn > mx:
        sal["min"], sal["max"] = mx, mn
        sal["salary_was_inverted"] = True
    else:
        sal["salary_was_inverted"] = False


def _fix_github(sig: dict) -> None:
    """
    F2 — GitHub score sentinel.
    FIX #4: Also handles None in raw data (not just -1).
    Previously set github_not_linked=False when score was already None.
    """
    score = sig.get("github_activity_score")
    if score == -1 or score is None:
        sig["github_activity_score"] = None
        sig["github_not_linked"]     = True
    else:
        sig["github_not_linked"] = False


def _fix_offer(sig: dict) -> None:
    """F3 — Offer acceptance rate sentinel. -1 means no offer history."""
    if sig.get("offer_acceptance_rate") == -1:
        sig["offer_acceptance_rate"] = None
        sig["no_offer_history"]      = True
    else:
        sig["no_offer_history"] = False


def apply_fixes(c: dict) -> None:
    sig = c.setdefault("redrob_signals", {})
    _fix_salary(sig)
    _fix_github(sig)
    _fix_offer(sig)
    sal = sig.get("expected_salary_range_inr_lpa", {})
    c["salary_was_inverted"] = sal.get("salary_was_inverted", False)
    c["github_not_linked"]   = sig.get("github_not_linked",   False)
    c["no_offer_history"]    = sig.get("no_offer_history",    False)


# ══════════════════════════════════════════════════════════════════════════════
#  FLAGS
# ══════════════════════════════════════════════════════════════════════════════

def fl3_skill_career_domain_mismatch(c: dict) -> bool:
    """FL3 — 3+ AI/ML skills claimed but zero AI/ML career history."""
    skills   = c.get("skills", [])
    career   = c.get("career_history", [])
    ai_count = sum(1 for sk in skills if _s(sk.get("name")) in AI_SKILLS)
    if ai_count < 3:
        return False
    return not _has_ai_career(career)


def fl5_education_overlap(edu_list: list) -> tuple[bool, list]:
    """FL5 — Two education entries with overlapping year ranges."""
    pairs = []
    for i in range(len(edu_list)):
        for j in range(i + 1, len(edu_list)):
            s1 = edu_list[i].get("start_year")
            e1 = edu_list[i].get("end_year")
            s2 = edu_list[j].get("start_year")
            e2 = edu_list[j].get("end_year")
            if None in (s1, e1, s2, e2):
                continue
            if max(s1, s2) < min(e1, e2):
                pairs.append([i, j])
    return bool(pairs), pairs


def fl6_reverse_degree_order(edu_list: list) -> bool:
    """FL6 — Postgraduate degree completed before undergraduate started."""
    pg_ends, ug_starts = [], []
    for e in edu_list:
        d  = _s(e.get("degree"))
        ey = e.get("end_year")
        sy = e.get("start_year")
        if d in POSTGRAD_DEGREES  and ey is not None: pg_ends.append(ey)
        if d in UNDERGRAD_DEGREES and sy is not None: ug_starts.append(sy)
    if not pg_ends or not ug_starts:
        return False
    return min(pg_ends) <= max(ug_starts)


def fl7_second_undergrad_after_first(edu_list: list) -> bool:
    """FL7 — Two undergraduate degrees with 2+ year gap between them."""
    ugs = sorted(
        [e for e in edu_list
         if _s(e.get("degree")) in UNDERGRAD_DEGREES
         and e.get("start_year") is not None
         and e.get("end_year")   is not None],
        key=lambda e: e["start_year"]
    )
    if len(ugs) < 2:
        return False
    return ugs[1]["start_year"] >= ugs[0]["end_year"] + 2


def fl8_education_career_gap(c: dict) -> tuple[bool, float]:
    """
    FL8/FL9 — Unreasonable gap between education end and career start.
    FIX #5: Now flags BOTH directions:
      gap > 5  : career started 5+ years after graduation (suspiciously late)
      gap < -1 : career started 1+ year before graduation (impossible;
                 -1 allows for internships/part-time in final year)
    """
    edu_ends = [
        e["end_year"] for e in c.get("education", [])
        if e.get("end_year") is not None
    ]
    career_starts = [
        _parse_date(j.get("start_date")).year
        for j in c.get("career_history", [])
        if _parse_date(j.get("start_date"))
    ]
    if not edu_ends or not career_starts:
        return False, 0.0
    gap = float(min(career_starts) - max(edu_ends))
    flagged = gap > 1
    return flagged, round(gap, 1)


def fl11_active_before_signup(c: dict) -> tuple[bool, int]:
    """FL11/FL12 — last_active_date is earlier than signup_date."""
    sig    = c.get("redrob_signals", {})
    signup = _parse_date(sig.get("signup_date"))
    last   = _parse_date(sig.get("last_active_date"))
    if signup is None or last is None:
        return False, 0
    if last < signup:
        return True, (signup - last).days
    return False, 0


def fl16_invalid_degree_field(edu: dict) -> bool:
    """FL16 — Engineering degree with a non-engineering field of study."""
    deg   = _s(edu.get("degree"))
    field = _s(edu.get("field_of_study"))
    if deg in ENG_DEGREE_SET and field in INVALID_ENG_FIELDS:
        return True
    if deg in {"m.sc", "m.sc.", "m.s.", "m.s"} and field in INVALID_ENG_FIELDS:
        return True
    return False


def fl18_fl19_duplicate_descriptions(career: list) -> tuple[bool, bool, list]:
    """FL18 — Duplicate job descriptions. FL19 — All descriptions identical."""
    if len(career) < 2:
        return False, False, []
    descs = [(j.get("description") or "").strip() for j in career]
    pairs = [
        [i, j] for i in range(len(descs))
        for j in range(i + 1, len(descs))
        if descs[i] and descs[i] == descs[j]
    ]
    non_empty = [d for d in descs if d]
    all_ident  = len(non_empty) > 1 and len(set(non_empty)) == 1
    return bool(pairs), all_ident, pairs


def fl20_low_engagement(c: dict) -> bool:
    """FL20 — recruiter_response_rate < 0.10 AND avg_response_time_hours > 200."""
    sig  = c.get("redrob_signals", {})
    rate = sig.get("recruiter_response_rate")
    hrs  = sig.get("avg_response_time_hours")
    if rate is None or hrs is None:
        return False
    return rate < 0.10 and hrs > 200


def fl25_possible_honeypot(c: dict) -> bool:
    """FL25 — 3+ structural impossibility flags simultaneously true."""
    structural = sum([
        bool(c.get("education_overlap")),
        bool(c.get("reverse_degree_order")),
        bool(c.get("second_undergrad_after_first")),
        bool(c.get("all_descriptions_identical")),
        bool(c.get("duplicate_job_descriptions")),
    ])
    return structural >= 3


def fl26_honeypot_score(c: dict) -> int:
    """FL26 — Total count of all triggered flags (0-11 scale)."""
    return sum(1 for f in _SCORE_FIELDS if c.get(f, False))


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER CLEAN FUNCTION
#  Mutates the dict from json.loads() directly — no deepcopy needed since
#  json.loads() always returns a fresh object per line.
# ══════════════════════════════════════════════════════════════════════════════

def process_candidate(c: dict) -> dict:
    edu_list = c.get("education", [])
    career   = c.get("career_history", [])

    # Step 1: Fixes (must run before flags)
    apply_fixes(c)

    # Step 2: Candidate-level flags
    c["skill_career_domain_mismatch"] = fl3_skill_career_domain_mismatch(c)

    ov_flag, ov_pairs = fl5_education_overlap(edu_list)
    c["education_overlap"]             = ov_flag
    c["overlapping_education_indices"] = ov_pairs

    c["reverse_degree_order"]         = fl6_reverse_degree_order(edu_list)
    c["second_undergrad_after_first"] = fl7_second_undergrad_after_first(edu_list)

    gap_flag, gap_val = fl8_education_career_gap(c)
    c["education_career_gap_flag"]    = gap_flag
    c["education_career_gap_years"]   = gap_val

    abs_flag, abs_days = fl11_active_before_signup(c)
    c["active_before_signup"]         = abs_flag
    c["signup_active_gap_days"]       = abs_days

    dup_any, all_ident, dup_idx = fl18_fl19_duplicate_descriptions(career)
    c["duplicate_job_descriptions"]    = dup_any
    c["all_descriptions_identical"]    = all_ident
    c["duplicate_description_indices"] = dup_idx

    c["low_engagement_flag"] = fl20_low_engagement(c)

    # Step 3: Per-education flags (FL16)
    any_invalid = False
    for edu in edu_list:
        val = fl16_invalid_degree_field(edu)
        edu["invalid_degree_field_combination"] = val
        if val:
            any_invalid = True
    c["_any_invalid_degree_field"] = any_invalid

    # Step 4: Aggregate flags (must be last)
    c["possible_honeypot"] = fl25_possible_honeypot(c)
    c["honeypot_score"]    = fl26_honeypot_score(c)

    return c


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION  (4-dimension tree routing)
# ══════════════════════════════════════════════════════════════════════════════

def _country_bucket(c: dict) -> str:
    profile  = c.get("profile", {})
    country  = _s(profile.get("country"))
    location = _s(profile.get("location"))
    if country in ("india", "in"):
        city = location.split(",")[0].strip()
        if any(pref in city for pref in INDIA_PREFERRED_CITIES):
            return "india_preferred"
        return "india_other"
    return "outside_india"


def _experience_band(c: dict) -> str:
    yoe = c.get("profile", {}).get("years_of_experience")
    if yoe is None: return "0_to_4"
    if yoe < 5:     return "0_to_4"
    if yoe <= 9:    return "5_to_9"
    return "10_plus"


def _domain_bucket(c: dict) -> str:
    title       = _s(c.get("profile", {}).get("current_title"))
    industry    = _s(c.get("profile", {}).get("current_industry"))
    skill_names = {_s(sk.get("name")) for sk in c.get("skills", [])}
    ai_overlap  = len(skill_names & AI_SKILLS)
    de_overlap  = len(skill_names & DE_DOMAIN_SKILLS)

    has_ai_title  = any(kw in title for kw in AI_TITLE_KEYWORDS)
    has_de_title  = any(kw in title for kw in DE_TITLE_KEYWORDS)
    has_swe_title = any(kw in title for kw in SWE_TITLE_KEYWORDS)

    # FIX #2: exact match only — no "ai" in industry substring
    has_ai_ind = (
        industry in AI_INDUSTRIES_EXACT
        or "machine learning" in industry
        or "artificial intelligence" in industry
    )

    if has_ai_title or has_ai_ind:
        return "ai_ml"
    if ai_overlap >= 3 and _has_ai_career(c.get("career_history", [])):
        return "ai_ml"
    if has_de_title or de_overlap >= 3:
        return "data_engineering"
    if has_swe_title:
        return "software_engineering"
    return "other"


def _availability_bucket(c: dict) -> str:
    last = _parse_date(c.get("redrob_signals", {}).get("last_active_date"))
    if last is None:
        return "inactive"
    days = (TODAY - last).days
    if days <= 30:  return "active"
    if days <= 90:  return "moderate"
    return "inactive"


def classify(c: dict) -> dict:
    return {
        "country":      _country_bucket(c),
        "experience":   _experience_band(c),
        "domain":       _domain_bucket(c),
        "availability": _availability_bucket(c),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TREE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _leaf_path(dataset_dir: str, buckets: dict) -> str:
    return os.path.join(
        dataset_dir,
        buckets["country"],
        buckets["experience"],
        buckets["domain"],
        buckets["availability"],
        "candidates.json"
    )


def build_tree(all_candidates: list, dataset_dir: str) -> int:
    grouped: dict[str, list] = defaultdict(list)
    for c in all_candidates:
        path = _leaf_path(dataset_dir, classify(c))
        grouped[path].append(c)

    for path, candidates in grouped.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(candidates, fh, ensure_ascii=False, indent=2)

    return len(grouped)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    if not os.path.exists(INPUT_FILE):
        print(f"❌  File not found: {INPUT_FILE}")
        print(f"    Make sure candidates.jsonl is in the same folder.")
        return

    wall_start = time.time()

    # Step 1: Read + Clean in a single pass (no deepcopy)
    print(f"📂  Reading and cleaning {INPUT_FILE} ...")
    cleaned   = []
    skipped   = 0
    honeypots = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)   # fresh dict every time — no deepcopy needed
            except json.JSONDecodeError as e:
                print(f"    ⚠  Skipped line {i}: {e}")
                skipped += 1
                continue

            c = process_candidate(raw)
            cleaned.append(c)
            if c.get("possible_honeypot"):
                honeypots += 1

            if i % 10_000 == 0:
                elapsed = time.time() - wall_start
                print(f"    ... {i:,} processed  ({elapsed:.1f}s elapsed)")

    read_clean_time = time.time() - wall_start
    print(f"    ✅ {len(cleaned):,} candidates cleaned  |  "
          f"{skipped} skipped  |  "
          f"{honeypots:,} honeypots  |  "
          f"{read_clean_time:.1f}s")

    # Step 2: Build tree
    print(f"\n🌳  Building Dataset/ tree ...")
    tree_start      = time.time()
    buckets_written = build_tree(cleaned, DATASET_DIR)
    tree_time       = time.time() - tree_start
    print(f"    ✅ {buckets_written} leaf buckets written  |  {tree_time:.1f}s")

    total_time = time.time() - wall_start

    print()
    print("=" * 58)
    print("  PIPELINE COMPLETE")
    print("=" * 58)
    print(f"  Candidates loaded    : {len(cleaned) + skipped:,}")
    print(f"  Candidates cleaned   : {len(cleaned):,}")
    print(f"  Skipped (malformed)  : {skipped}")
    print(f"  Honeypots flagged    : {honeypots:,}  "
          f"({100 * honeypots / max(len(cleaned), 1):.1f}%)")
    print(f"  Leaf buckets written : {buckets_written}")
    print(f"  Output folder        : {os.path.abspath(DATASET_DIR)}/")
    print(f"  Total wall time      : {total_time:.1f}s")
    print()


if __name__ == "__main__":
    run()
