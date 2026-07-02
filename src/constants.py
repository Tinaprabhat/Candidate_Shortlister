"""
constants.py — Central configuration for RedRob pipeline.
All weights, thresholds, paths, and lookup tables live here.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "decompressed"
DATA_DIR = PROJECT_ROOT / "data"
JD_JSON_PATH = DATA_DIR / "jd.json"

# Model sub-paths (after setup.sh decompresses)
# Three models only: MiniLM sentence-transformer, FlashRank, Fraud KB
SENTENCE_TRANSFORMER_DIR = MODELS_DIR / "sentence_transformer"
FLASHRANK_DIR = MODELS_DIR / "ms-marco-MiniLM-L-12-v2"   # actual dir from tar extraction
FRAUD_KB_PATH = MODELS_DIR / "fraud_kb" / "fraud_kb.db"

# Embedding model name (downloaded by build_kb.py)
ST_MODEL_NAME = "all-MiniLM-L6-v2"
FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
EMBED_DIM = 384

# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ──────────────────────────────────────────────────────────────────────────────
FOLDER_DISPATCH_STAGGER_SEC = 0   # 0 = all folder threads start simultaneously

HARD_POOL_CAP = 60_000         # hard cap before expensive layers

# Concurrent workers for the L1a→L3 streaming pipeline (run_streaming_cascade).
# Each worker streams one candidate continuously through every pre-gate stage;
# multiple candidates are in-flight (at different stages) simultaneously.
PIPELINE_MAX_WORKERS = 32

# Soft penalty magnitudes
SENIORITY_PENALTY = 0.85       # multiply score if under-qualified (L3)

# ──────────────────────────────────────────────────────────────────────────────
# BATCH SIZES
# ──────────────────────────────────────────────────────────────────────────────
L1C_MIN_SKILL_MATCH = 0.0    # per-folder gate inside l1c_skill_match (0.0 = off)
L4_BATCH_SIZE = 64
FLASHRANK_TOP_N = 50           # polish only top 50
FLASHRANK_FIS_WEIGHT = 0.25    # blend weight: final = W*flashrank + (1-W)*fis  → FIS weight = 0.75

# ──────────────────────────────────────────────────────────────────────────────
# L7 TIER THRESHOLDS — "very good" candidate criteria
# A candidate meeting ALL of these (with no L1b flags and no L3 penalty)
# is promoted to tier "very_good" and guaranteed placement in the top 10.
# ──────────────────────────────────────────────────────────────────────────────
L7_VERY_GOOD_L1C_MIN = 0.65       # skill match score
L7_VERY_GOOD_L4_MIN = 0.60        # semantic work relevance
L7_VERY_GOOD_L6_MIN = 0.60        # behavioral signals
L7_VERY_GOOD_EXP_MIN = 7.0        # years of experience
L7_VERY_GOOD_COMPANY_AGE_MIN = 10.0  # oldest company must be ≥ 10 yrs old (0 = no data, skip)

# ──────────────────────────────────────────────────────────────────────────────
# FOLDER NAME ABBREVIATION EXPANSION (pruning)
# ──────────────────────────────────────────────────────────────────────────────
FOLDER_ABBREVIATIONS = {
    "swe": "software engineer",
    "sde": "software development engineer",
    "ml": "machine learning",
    "mlops": "machine learning operations ml ops",
    "ai": "artificial intelligence",
    "ds": "data scientist data science",
    "de": "data engineer",
    "hr": "human resources recruiter",
    "pm": "product manager",
    "qa": "quality assurance testing",
    "devops": "development operations devops",
    "cloud": "cloud computing infrastructure",
    "fe": "frontend front end",
    "be": "backend back end",
    "fullstack": "full stack",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "sre": "site reliability engineer",
    "ux": "user experience design",
    "ui": "user interface design",
    "sec": "security",
}

# ──────────────────────────────────────────────────────────────────────────────
# STRUCTURED FOLDER PRUNING RULES
# ──────────────────────────────────────────────────────────────────────────────

# Keywords that classify a JD role as "engineering".
# Matched against the normalised job_title + required skills text.
ENGINEERING_ROLE_KEYWORDS: frozenset = frozenset({
    "engineer", "engineering", "developer", "dev", "sde", "swe",
    "backend", "frontend", "fullstack", "full stack",
    "data", "ml", "ai", "aiml", "machine learning", "deep learning",
    "cloud", "devops", "mlops", "platform", "infrastructure", "sre",
    "python", "java", "golang", "scala", "rust", "typescript",
    "nlp", "computer vision", "data scientist", "data science", "software",
})

# Folder-path substrings that mean "non-engineering candidates".
# If the JD is engineering, folders matching any of these are hard-dropped.
OTHER_ROLE_FOLDER_PATTERNS: tuple = (
    "other role", "other_role", "non engineering", "non_engineering",
    "non tech", "non_tech", "business", "sales", "marketing",
    "human resources", "finance", "legal", "operations",
    "customer success", "customer_success",
)

# Candidate-level status values that mean the profile is inactive.
# Checked (lowercased) against any of: status, candidate_status, active_status.
INACTIVE_CANDIDATE_STATUSES: frozenset = frozenset({
    "inactive", "not active", "not_active", "closed", "unavailable",
    "on hold", "hold", "deactivated", "blacklisted", "rejected",
    "not available", "passive", "archived",
})

# Folder names (normalised: lowercase, _ / - → space) where the ENTIRE FOLDER
# is a bucket of inactive candidates.  The branch is hard-dropped in Phase 1
# before any candidate is loaded.
INACTIVE_FOLDER_NAMES: frozenset = frozenset({
    # All entries must be the POST-normalisation form (underscores already
    # replaced with spaces by _path_parts_normalized before comparison).
    "inactive", "not active", "closed", "unavailable",
    "deactivated", "blacklisted", "rejected", "archived",
})

# ── New tree: root-level code/no-code bucket identifiers ─────────────────────
# Used by Phase 1a to prune the entire branch when the JD's coding requirement
# contradicts the folder's root-level bucket.

# Normalised root folder names that mean "no coding required" candidates.
NO_CODE_FOLDER_ROOTS: frozenset = frozenset({
    "no code", "no_code", "nocode",
    "non code", "non_code",
    "non tech", "non_tech",
    "no coding", "non coding", "non_coding",
})

# Normalised root folder names that mean "coding / technical" candidates.
CODE_FOLDER_ROOTS: frozenset = frozenset({
    "code", "coding", "tech", "technical",
})

# ──────────────────────────────────────────────────────────────────────────────
# ROLE-FAMILY PRUNING (Phase 1e extended)
# ──────────────────────────────────────────────────────────────────────────────

# Fine-grained JD role-family detection.
# Iterated in insertion order; the FIRST family whose ANY keyword appears
# (as a substring) in the normalised "job_title + required_skills" text wins.
# All families here are engineering sub-families; a JD that matches none of
# these (and none of ENGINEERING_ROLE_KEYWORDS) falls back to "other".
ROLE_FAMILY_KEYWORDS: dict = {
    "ai_ml": frozenset({
        "ai engineer", "ml engineer", "machine learning", "deep learning",
        "nlp", "natural language processing", "computer vision", "llm",
        "large language model", "recommendation", "search engineer",
        "data scientist", "generative ai", "gen ai", "aiml", "ai/ml",
        "research scientist", "reinforcement learning",
    }),
    "cloud_devops": frozenset({
        "cloud engineer", "cloud architect", "devops engineer",
        "devops", "infrastructure engineer", "platform engineer",
        "site reliability engineer", "sre", "devsecops",
    }),
    "qa": frozenset({
        "qa engineer", "quality assurance", "quality engineer",
        "test engineer", "testing engineer", "sdet",
        "automation engineer", "software tester",
    }),
    "mobile": frozenset({
        "mobile engineer", "mobile developer", "android developer",
        "ios developer", "react native", "flutter developer",
        "mobile application",
    }),
    "java_net": frozenset({
        "java developer", "java engineer", ".net developer", ".net engineer",
        "dotnet developer", "dotnet engineer", "spring boot developer",
        "c# developer", "c# engineer",
    }),
    # "swe_general" is the implicit fallback for any other engineering role
}

# Folder-path substrings (normalised: lowercase, _ / - → space) that identify
# candidate buckets INCOMPATIBLE with each JD role family.
# Any match in ANY normalised path component triggers a Phase-1e hard-DROP.
ROLE_FAMILY_EXCLUSIONS: dict = {
    "ai_ml": frozenset({
        # cloud / infrastructure
        "cloud engineer", "cloud architect", "devops", "devops engineer",
        "systems engineer",
        # quality / test
        "qa engineer", "quality engineer", "quality assurance", "sdet",
        # mobile
        "mobile engineer", "mobile developer", "android", "ios developer",
        # java / .net  (both "engineer" and "developer" variants)
        "java developer", "java engineer",
        "net engineer", "net developer", "dotnet", "dotnet developer",
        # frontend (not relevant for backend AI/ML work)
        "frontend engineer", "frontend developer",
        "front end engineer", "front end developer",
    }),
    "cloud_devops": frozenset({
        "qa engineer", "quality engineer", "quality assurance", "sdet",
        "mobile engineer", "mobile developer", "android", "ios developer",
        "java developer", "java engineer",
        "net engineer", "net developer", "dotnet", "dotnet developer",
        "frontend engineer", "frontend developer",
    }),
    "qa": frozenset({
        "cloud engineer", "cloud architect", "devops", "mobile engineer",
        "mobile developer", "android", "ios developer",
        "java developer", "java engineer",
        "net engineer", "net developer", "dotnet", "dotnet developer",
    }),
    "mobile": frozenset({
        "cloud engineer", "cloud architect", "devops", "devops engineer",
        "qa engineer", "quality engineer", "quality assurance",
        "java developer", "java engineer",
        "net engineer", "net developer", "dotnet", "dotnet developer",
    }),
    "java_net": frozenset({
        "cloud engineer", "cloud architect", "devops", "devops engineer",
        "qa engineer", "quality engineer", "quality assurance",
        "mobile engineer", "mobile developer", "android", "ios developer",
    }),
    "swe_general": frozenset(),   # broad SWE role: accept all engineering sub-families
}
